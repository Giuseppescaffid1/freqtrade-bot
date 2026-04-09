#!/usr/bin/env python3
"""Generate FTMO dashboard JSON — called by cron every 5 min."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import requests

BASE = Path("/home/giuseppe/freqtrade-bot")
DB_URL = "postgresql://spi_user:spi_secure_2024@localhost:5432/spi"
OUTPUT = Path("/var/www/showcase/ftmo_dashboard.json")

BOTS = [
    {"name": "FTMO_Blitz",      "port": 8086, "user": "ftmoblitz",  "password": "ftmoblitz2026",
     "pair": "ETH/USDT:USDT", "display_pair": "ETH/USDT", "tf": "15m",
     "desc": "Aggressive ETH breakout (EMA cross + RSI + ADX + MACD + BB + Volume)",
     "backtest": {"period": "Jan 2024-Mar 2026", "return_pct": 8.53, "trades": 82, "win_rate": 47.6, "max_dd": 4.66},
     "params": {
         "EMA": "10 / 56",
         "RSI zone": "43-55",
         "ADX min": 29,
         "MACD": "histogram > 0",
         "BB": "price < upper band",
         "Volume": "> 0.9x avg",
         "SL": "trailing",
         "Risk": "1% per trade",
     }},
    {"name": "FTMO_CryptoEdge", "port": 8087, "user": "ftmoedge",   "password": "ftmoedge2026",
     "pair": "ETH/USDT:USDT", "display_pair": "ETH/USDT", "tf": "15m",
     "desc": "Multi-layer ETH edge (StochRSI + BB squeeze + ADX + RSI slope)",
     "backtest": {"period": "Jan 2024-Mar 2026", "return_pct": None, "trades": None, "win_rate": None, "max_dd": None},
     "params": {
         "StochRSI": "K/D crossover",
         "BB squeeze": "width < threshold",
         "ADX": "trend filter",
         "RSI slope": "momentum confirmation",
         "Risk": "1% per trade",
     }},
    {"name": "FTMO_SMC_SOL",    "port": 8088, "user": "ftmosmcsol", "password": "ftmosmcsol2026",
     "pair": "SOL/USDT:USDT", "display_pair": "SOL/USDT", "tf": "15m",
     "desc": "Smart Money Concepts + custom trailing (FBB + swing structure + volume)",
     "backtest": {"period": "Jan 2024-Mar 2026", "return_pct": 14.59, "trades": 87, "win_rate": 52.9, "max_dd": 0.35},
     "params": {
         "EMA": "10 / 56",
         "RSI zone": "43-55",
         "ADX min": 29,
         "FBB": "Fisher BB breakout",
         "Volume": "> 0.9x avg",
         "SL": "custom trailing",
         "Risk": "1% per trade",
     }},
]

STANDALONE = [
    {"name": "Gold", "display_pair": "XAUUSD", "tf": "15m",
     "desc": "4-phase pullback breakout",
     "health_file": BASE / "ftmo_gold_health.json",
     "backtest": {"period": "N/A (live-tuned)", "return_pct": None, "trades": None, "win_rate": None, "max_dd": None},
     "params": {
         "EMA": "3 / 14 / 24 / 100",
         "ATR period": 10,
         "ADX min": 20,
         "Pullback": "2 counter-trend candles, 6-bar window",
         "SL": "2.5x ATR",
         "TP": "10x ATR (long) / 6.5x ATR (short)",
         "R:R": "~1:4",
         "Risk": "1.5% per trade",
         "Skip hours": "22-01 UTC",
     }},
    {"name": "Oil",  "display_pair": "USOIL", "tf": "15m",
     "desc": "News-driven oil scalping",
     "health_file": BASE / "ftmo_oil_health.json",
     "backtest": {"period": "N/A (live-tuned)", "return_pct": None, "trades": None, "win_rate": None, "max_dd": None},
     "params": {
         "EMA": "20 / 50 / 200",
         "ATR period": 14,
         "RSI period": 14,
         "ADX period": 14,
         "Sentiment": "strong ≥3.0 / weak ≥1.5",
         "SL": "1.5x ATR",
         "TP": "4.5x ATR",
         "R:R": "1:3",
         "Risk": "1% per trade",
         "Max hold": "24 candles (6h)",
         "Cooldown": "4 candles (1h)",
     }},
]


def ft_get(bot, endpoint, params=None):
    try:
        r = requests.get(
            f"http://127.0.0.1:{bot['port']}/api/v1/{endpoint}",
            auth=(bot["user"], bot["password"]),
            params=params, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_bot_heartbeat(bot):
    """Get real-time heartbeat: last candle, indicators, signal proximity, open trades."""
    heartbeat = {
        "alive": False,
        "last_analyzed": None,
        "last_candle_age_s": None,
        "open_trades": [],
        "indicators": {},
        "signal_summary": "offline",
    }

    # Check if bot responds
    profit = ft_get(bot, "profit")
    if not profit:
        return heartbeat
    heartbeat["alive"] = True

    # Open trades
    status = ft_get(bot, "status")
    if isinstance(status, list):
        for t in status:
            heartbeat["open_trades"].append({
                "pair": t.get("pair", ""),
                "direction": "short" if t.get("is_short") else "long",
                "profit_pct": round(t.get("profit_pct", 0), 2),
                "profit_abs": round(t.get("profit_abs", 0), 2),
                "duration": t.get("trade_duration", ""),
                "stoploss": t.get("stop_loss_abs"),
                "current_rate": t.get("current_rate"),
            })

    # Last candle indicators
    candles = ft_get(bot, "pair_candles", {
        "pair": bot["pair"], "timeframe": bot["tf"], "limit": 1
    })
    if candles and candles.get("data"):
        cols = candles.get("columns", [])
        row = candles["data"][-1]
        data = dict(zip(cols, row))

        candle_time = data.get("date", "")
        heartbeat["last_analyzed"] = candle_time
        if candle_time:
            try:
                ct = datetime.fromisoformat(candle_time.replace("Z", "+00:00"))
                heartbeat["last_candle_age_s"] = int((datetime.now(timezone.utc) - ct).total_seconds())
            except Exception:
                pass

        # Extract key indicators
        ind = {}
        price = data.get("close")
        if price:
            ind["price"] = round(price, 2)

        for key in ("rsi", "adx", "macd_hist", "bb_upper", "bb_lower", "bb_mid",
                     "stoch_rsi_k", "stoch_rsi_d", "bb_width_pct", "bb_squeeze",
                     "rsi_slope", "volume"):
            if key in data and data[key] is not None:
                ind[key] = round(float(data[key]), 2) if isinstance(data[key], (int, float)) else data[key]

        # EMA fast/slow for trend
        for key in ("ema_fast_line", "ema_slow_line", "ema_10", "ema_56", "ema_24", "ema_27"):
            if key in data and data[key] is not None:
                ind[key] = round(float(data[key]), 2)

        # Check entry signals
        enter_long = data.get("enter_long")
        enter_short = data.get("enter_short")
        ind["enter_long"] = bool(enter_long) if enter_long is not None else False
        ind["enter_short"] = bool(enter_short) if enter_short is not None else False

        heartbeat["indicators"] = ind

        # Build human-readable signal summary
        heartbeat["signal_summary"] = _build_signal_summary(bot["name"], ind, price)

    return heartbeat


def _build_signal_summary(name, ind, price):
    """Build a human-readable summary of why a signal is or isn't firing."""
    if not ind or not price:
        return "no data"

    if ind.get("enter_long") or ind.get("enter_short"):
        direction = "LONG" if ind.get("enter_long") else "SHORT"
        return f"SIGNAL ACTIVE: {direction} entry triggered"

    reasons = []

    rsi = ind.get("rsi")
    adx = ind.get("adx")
    macd_hist = ind.get("macd_hist")

    if "FTMO_Blitz" in name or "CryptoEdge" in name:
        # EMA cross + RSI + ADX strategies
        ema_fast = ind.get("ema_fast_line") or ind.get("ema_10")
        ema_slow = ind.get("ema_slow_line") or ind.get("ema_56")
        if ema_fast and ema_slow:
            if ema_fast > ema_slow:
                reasons.append("EMA: bullish cross")
            else:
                reasons.append("EMA: bearish (no long)")

        if rsi is not None:
            if 43 <= rsi <= 55:
                reasons.append(f"RSI: {rsi:.0f} (in zone)")
            elif rsi > 70:
                reasons.append(f"RSI: {rsi:.0f} overbought")
            elif rsi < 30:
                reasons.append(f"RSI: {rsi:.0f} oversold")
            else:
                reasons.append(f"RSI: {rsi:.0f} (waiting)")

        if adx is not None:
            if adx >= 29:
                reasons.append(f"ADX: {adx:.0f} (strong trend)")
            else:
                reasons.append(f"ADX: {adx:.0f} (weak, need >29)")

        if macd_hist is not None:
            if macd_hist > 0:
                reasons.append("MACD: bullish")
            else:
                reasons.append("MACD: bearish")

    elif "SMC_SOL" in name:
        if rsi is not None:
            reasons.append(f"RSI: {rsi:.0f}")
        if adx is not None:
            reasons.append(f"ADX: {adx:.0f}")

    if not reasons:
        return "scanning for opportunities"

    return " | ".join(reasons)


def get_standalone_heartbeat(s):
    """Get heartbeat from standalone health file."""
    try:
        data = json.loads(s["health_file"].read_text())
        updated = data.get("updated_at", "")
        age_s = None
        if updated:
            try:
                ut = datetime.fromisoformat(updated)
                age_s = int((datetime.now(timezone.utc) - ut).total_seconds())
            except Exception:
                pass

        phase = data.get("phase", "")
        detail = data.get("detail", "")
        position = data.get("position", False)

        # Use detailed signal_explanation from health file if available
        sig_expl = data.get("signal_explanation", "")

        if position:
            summary = f"IN POSITION — hold candles: {data.get('hold_candles', '?')}"
            if sig_expl:
                summary += f"\n{sig_expl}"
        elif sig_expl:
            phase_prefix = f"[{phase}] " if phase else ""
            summary = f"{phase_prefix}{sig_expl}"
        elif phase:
            summary = f"{phase}" + (f" — {detail}" if detail else "")
        else:
            summary = "active"

        return {
            "alive": data.get("status") == "ok",
            "last_analyzed": data.get("last_candle"),
            "last_candle_age_s": age_s,
            "health_age_s": age_s,
            "open_trades": [{"pair": s["display_pair"], "direction": "active", "profit_pct": 0}] if position else [],
            "indicators": data.get("indicators", {}),
            "phase": phase,
            "has_position": position,
            "signal_summary": summary,
        }
    except Exception:
        return {
            "alive": False, "last_analyzed": None, "last_candle_age_s": None,
            "open_trades": [], "signal_summary": "offline",
        }


def get_bridge_status():
    """Parse last bridge status line."""
    bridge_log = BASE / "ftmo_bridge.log"
    if not bridge_log.exists():
        return {}
    try:
        result = subprocess.run(
            ["tail", "-20", str(bridge_log)],
            capture_output=True, text=True)
        import re
        for line in reversed(result.stdout.strip().split("\n")):
            if "Status:" in line:
                mt_ok = "mt=OK" in line
                m = re.search(r"garch=\[(.*?)\]", line)
                garch = m.group(1) if m else ""
                # Extract timestamp
                ts = line[:19] if len(line) > 19 else ""
                return {
                    "mt_ok": mt_ok,
                    "garch": garch,
                    "last_status_line": ts,
                }
    except Exception:
        pass
    return {}


def main():
    conn = psycopg2.connect(DB_URL)

    # Account snapshot
    with conn.cursor() as cur:
        cur.execute("""
            SELECT balance, equity, daily_pnl_pct, total_dd_pct, open_positions, mt5_connected
            FROM ftmo.account_snapshots ORDER BY snapshot_time DESC LIMIT 1
        """)
        row = cur.fetchone()

    account = {}
    if row:
        account = {
            "balance": float(row[0]),
            "equity": float(row[1]),
            "daily_pnl_pct": round(float(row[2]) * 100, 3),
            "total_dd_pct": round(float(row[3]) * 100, 3),
            "open_positions": row[4],
            "mt5_connected": row[5],
            "pnl_from_start": round(float(row[0]) - 80000, 2),
        }

    # Per-strategy live stats from DB
    with conn.cursor() as cur:
        cur.execute("""
            SELECT name, closed_trades, total_pnl, total_mt5_pnl, avg_profit_pct,
                   wins, losses, win_rate, last_trade_date
            FROM ftmo.v_strategy_summary WHERE is_active = true
        """)
        db_stats = {r[0]: {
            "closed_trades": r[1] or 0,
            "dryrun_pnl": round(float(r[2]), 2) if r[2] else 0,
            "mt5_pnl": round(float(r[3]), 2) if r[3] else None,
            "avg_profit_pct": round(float(r[4]), 2) if r[4] else None,
            "wins": r[5] or 0,
            "losses": r[6] or 0,
            "win_rate": round(float(r[7]) * 100, 1) if r[7] else None,
            "last_trade": r[8].isoformat() if r[8] else None,
        } for r in cur.fetchall()}

    conn.close()

    # Build strategies array
    strategies = []

    for bot in BOTS:
        live = db_stats.get(bot["name"], {})
        hb = get_bot_heartbeat(bot)

        strategies.append({
            "name": bot["name"],
            "pair": bot["display_pair"],
            "timeframe": bot["tf"],
            "description": bot["desc"],
            "type": "bridge",
            "heartbeat": hb,
            "live": {
                "closed_trades": live.get("closed_trades", 0),
                "mt5_pnl_chf": live.get("mt5_pnl"),
                "dryrun_pnl_usdt": live.get("dryrun_pnl", 0),
                "wins": live.get("wins", 0),
                "losses": live.get("losses", 0),
                "win_rate": live.get("win_rate"),
                "last_trade": live.get("last_trade"),
            },
            "backtest": bot["backtest"],
            "params": bot.get("params"),
        })

    for st in STANDALONE:
        live = db_stats.get(st["name"], {})
        hb = get_standalone_heartbeat(st)

        strategies.append({
            "name": st["name"],
            "pair": st["display_pair"],
            "timeframe": st["tf"],
            "description": st["desc"],
            "type": "standalone",
            "heartbeat": hb,
            "live": {
                "closed_trades": live.get("closed_trades", 0),
                "mt5_pnl_chf": live.get("mt5_pnl"),
                "dryrun_pnl_usdt": live.get("dryrun_pnl", 0),
                "wins": live.get("wins", 0),
                "losses": live.get("losses", 0),
                "win_rate": live.get("win_rate"),
                "last_trade": live.get("last_trade"),
            },
            "backtest": st["backtest"],
            "params": st.get("params"),
        })

    # Bridge status
    bridge = get_bridge_status()

    # Services status
    services = []
    for svc in ["ftmo-bridge", "ftmo-gold", "ftmo-oil", "freqtrade-ftmo-blitz",
                 "freqtrade-crypto-edge", "freqtrade-smc-sol",
                 "ftmo-collector", "ftmo-telegram",
                 "oil-news-collector", "oil-vessel-tracker"]:
        result = subprocess.run(
            ["systemctl", "is-active", svc],
            capture_output=True, text=True)
        services.append({
            "name": svc,
            "active": result.stdout.strip() == "active",
        })

    # Agent supervisor overrides
    agent = {}
    agent_file = BASE / "ftmo_risk_overrides.json"
    if agent_file.exists():
        try:
            agent_data = json.loads(agent_file.read_text())
            # Check if expired
            exp = agent_data.get("expires_at", "")
            if exp:
                exp_dt = datetime.fromisoformat(exp)
                if datetime.now(timezone.utc) <= exp_dt:
                    agent = agent_data
        except Exception:
            pass

    dashboard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "account": account,
        "bridge": bridge,
        "strategies": strategies,
        "services": services,
        "agent": agent,
        "ftmo_rules": {
            "initial_balance": 80000,
            "currency": "CHF",
            "max_daily_loss_pct": 5.0,
            "max_total_dd_pct": 10.0,
            "step1_target_pct": 10.0,
            "step2_target_pct": 5.0,
        },
    }

    OUTPUT.write_text(json.dumps(dashboard, indent=2, default=str))
    print(f"Dashboard JSON written: {OUTPUT}")


if __name__ == "__main__":
    main()
