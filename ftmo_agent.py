#!/usr/bin/env python3
"""
FTMO Strategy Supervisor Agent — Hybrid (Rules + LLM)

Runs every 10 minutes. Reads all system state, applies hard rule-based checks,
optionally calls HuggingFace LLM for soft decisions, writes risk overrides.

Architecture:
    [All strategies] → ftmo_agent.py → ftmo_risk_overrides.json → ftmo_risk_manager.py

The agent can REDUCE risk, never increase it above base profiles.
If the agent crashes or is stale, the system reverts to base risk (fail-open).
"""

import json
import fcntl
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psycopg2
import requests

# ── Config ──
BASE = Path("/home/giuseppe/freqtrade-bot")
OVERRIDE_FILE = BASE / "ftmo_risk_overrides.json"
DASHBOARD_FILE = Path("/var/www/showcase/ftmo_dashboard.json")
GOLD_HEALTH = BASE / "ftmo_gold_health.json"
OIL_HEALTH = BASE / "ftmo_oil_health.json"
RISK_STATE = BASE / "ftmo_shared_risk.json"
SENTIMENT_FILE = BASE / "oil_sentiment.json"
LOG_FILE = BASE / "ftmo_agent.log"
DB_URL = "postgresql://spi_user:spi_secure_2024@localhost:5432/spi"

OVERRIDE_TTL_MINUTES = 15  # overrides expire after this
AGENT_INTERVAL = 600       # expected run interval (10 min)
LLM_TIMEOUT = 15           # seconds

# HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN", "")
if not HF_TOKEN:
    # Try loading from .env
    env_file = BASE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                HF_TOKEN = line.split("=", 1)[1].strip()
                break

HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# FTMO limits (mirrors risk_manager)
FTMO_INITIAL_BALANCE = 80_000.0

# Base risk profiles (must match risk_manager SOURCE_PROFILES)
BASE_PROFILES = {
    "FTMO_Blitz":      {"risk_per_trade": 0.01, "max_positions": 1},
    "FTMO_CryptoEdge": {"risk_per_trade": 0.01, "max_positions": 2},
    "FTMO_SmartMoney": {"risk_per_trade": 0.01, "max_positions": 2},
    "FTMO_SMC_SOL":    {"risk_per_trade": 0.01, "max_positions": 2},
    "Gold":            {"risk_per_trade": 0.005, "max_positions": 1},
    "Oil":             {"risk_per_trade": 0.005, "max_positions": 1},
}

# Telegram (reuse from bridge env)
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""
_bridge_env = BASE / ".env.ftmo_bridge"
if _bridge_env.exists():
    for line in _bridge_env.read_text().splitlines():
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            TELEGRAM_TOKEN = line.split("=", 1)[1].strip()
_chat_id_file = BASE / "telegram_chat_id.json"
if _chat_id_file.exists():
    try:
        TELEGRAM_CHAT_ID = str(json.loads(_chat_id_file.read_text())["chat_id"])
    except Exception:
        pass

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("ftmo_agent")


# ═══════════════════════════════════════════
#  DATA COLLECTION
# ═══════════════════════════════════════════

def read_json_safe(path: Path) -> dict | None:
    """Read JSON with shared lock, return None on failure."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        return data
    except Exception as e:
        logger.warning(f"Failed to read {path.name}: {e}")
        return None


def collect_world_state() -> dict:
    """Assemble all available data into a unified world state."""
    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dashboard": None,
        "risk_state": None,
        "gold_health": None,
        "oil_health": None,
        "sentiment": None,
        "db_performance": {},
        "db_recent_trades": [],
    }

    # Dashboard (primary data source)
    state["dashboard"] = read_json_safe(DASHBOARD_FILE)
    state["risk_state"] = read_json_safe(RISK_STATE)
    state["gold_health"] = read_json_safe(GOLD_HEALTH)
    state["oil_health"] = read_json_safe(OIL_HEALTH)
    state["sentiment"] = read_json_safe(SENTIMENT_FILE)

    # DB: strategy performance + recent trades
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            # Per-strategy summary
            cur.execute("""
                SELECT name, closed_trades, total_pnl, total_mt5_pnl,
                       wins, losses, win_rate, last_trade_date
                FROM ftmo.v_strategy_summary WHERE is_active = true
            """)
            for r in cur.fetchall():
                state["db_performance"][r[0]] = {
                    "closed_trades": r[1] or 0,
                    "total_pnl": float(r[2]) if r[2] else 0,
                    "mt5_pnl": float(r[3]) if r[3] else None,
                    "wins": r[4] or 0,
                    "losses": r[5] or 0,
                    "win_rate": float(r[6]) if r[6] else None,
                    "last_trade": r[7].isoformat() if r[7] else None,
                }

            # Recent trades (last 48h) for streak detection
            cur.execute("""
                SELECT strategy_name, (profit_pct > 0) AS won, close_date, profit_pct
                FROM ftmo.trades
                WHERE close_date > NOW() - INTERVAL '48 hours'
                  AND close_date IS NOT NULL
                ORDER BY close_date DESC
            """)
            state["db_recent_trades"] = [
                {"strategy": r[0], "won": bool(r[1]), "closed": r[2].isoformat(), "pct": float(r[3]) if r[3] else 0}
                for r in cur.fetchall()
            ]
        conn.close()
    except Exception as e:
        logger.warning(f"DB query failed: {e}")

    return state


# ═══════════════════════════════════════════
#  HARD RULES (deterministic, no LLM)
# ═══════════════════════════════════════════

def apply_hard_rules(state: dict) -> dict:
    """
    Apply deterministic risk rules. Returns override structure.
    These ALWAYS take precedence over LLM suggestions.
    """
    overrides = {
        "global_halt": False,
        "hard_rules_triggered": [],
        "strategy_overrides": {},
    }

    # Initialize all strategies with defaults
    for name in BASE_PROFILES:
        overrides["strategy_overrides"][name] = {
            "enabled": True,
            "risk_multiplier": 1.0,
            "max_positions": BASE_PROFILES[name]["max_positions"],
            "reason": "No adjustment",
        }

    dashboard = state.get("dashboard")
    if not dashboard:
        logger.warning("No dashboard data — applying conservative defaults")
        overrides["hard_rules_triggered"].append("no_dashboard_data")
        for name in overrides["strategy_overrides"]:
            overrides["strategy_overrides"][name]["risk_multiplier"] = 0.5
            overrides["strategy_overrides"][name]["reason"] = "Conservative: no dashboard data"
        return overrides

    account = dashboard.get("account", {})
    daily_pnl_pct = account.get("daily_pnl_pct", 0)  # already in % (e.g., -2.5)
    balance = account.get("balance", FTMO_INITIAL_BALANCE)
    total_dd_pct = ((balance - FTMO_INITIAL_BALANCE) / FTMO_INITIAL_BALANCE) * 100

    # ── Rule 1: Graduated daily drawdown protection ──
    if daily_pnl_pct <= -3.5:
        overrides["global_halt"] = True
        overrides["hard_rules_triggered"].append("daily_pnl_-3.5pct_HALT")
        for name in overrides["strategy_overrides"]:
            overrides["strategy_overrides"][name]["enabled"] = False
            overrides["strategy_overrides"][name]["risk_multiplier"] = 0.0
            overrides["strategy_overrides"][name]["reason"] = "HALT: daily loss -3.5%"
        logger.critical(f"GLOBAL HALT: daily PnL {daily_pnl_pct:.2f}%")
        return overrides  # No need to check further

    if daily_pnl_pct <= -3.0:
        overrides["hard_rules_triggered"].append("daily_pnl_-3pct")
        for name in overrides["strategy_overrides"]:
            overrides["strategy_overrides"][name]["risk_multiplier"] = 0.25
            overrides["strategy_overrides"][name]["max_positions"] = 1
            overrides["strategy_overrides"][name]["reason"] = "Hard rule: daily loss > 3%"
        logger.warning(f"Daily PnL {daily_pnl_pct:.2f}% — risk at 25%, max 1 pos each")

    elif daily_pnl_pct <= -2.0:
        overrides["hard_rules_triggered"].append("daily_pnl_-2pct")
        for name in overrides["strategy_overrides"]:
            overrides["strategy_overrides"][name]["risk_multiplier"] = 0.5
            overrides["strategy_overrides"][name]["reason"] = "Hard rule: daily loss > 2%"
        logger.warning(f"Daily PnL {daily_pnl_pct:.2f}% — risk reduced to 50%")

    # ── Rule 2: Total drawdown protection ──
    if total_dd_pct <= -7.5:
        overrides["global_halt"] = True
        overrides["hard_rules_triggered"].append("total_dd_-7.5pct_HALT")
        for name in overrides["strategy_overrides"]:
            overrides["strategy_overrides"][name]["enabled"] = False
            overrides["strategy_overrides"][name]["risk_multiplier"] = 0.0
            overrides["strategy_overrides"][name]["reason"] = "HALT: total DD > 7.5%"
        logger.critical(f"GLOBAL HALT: total DD {total_dd_pct:.2f}%")
        return overrides

    if total_dd_pct <= -6.0:
        overrides["hard_rules_triggered"].append("total_dd_-6pct")
        for name in overrides["strategy_overrides"]:
            s = overrides["strategy_overrides"][name]
            s["risk_multiplier"] = min(s["risk_multiplier"], 0.5)
            s["reason"] = "Hard rule: total DD > 6%"
        logger.warning(f"Total DD {total_dd_pct:.2f}% — risk capped at 50%")

    # ── Rule 3: Losing streak protection ──
    recent = state.get("db_recent_trades", [])
    for strat_name in BASE_PROFILES:
        strat_trades = [t for t in recent if t["strategy"] == strat_name]
        # Count consecutive losses from most recent
        streak = 0
        for t in strat_trades:
            if not t["won"]:
                streak += 1
            else:
                break

        if streak >= 5:
            s = overrides["strategy_overrides"].get(strat_name)
            if s:
                s["enabled"] = False
                s["risk_multiplier"] = 0.0
                s["reason"] = f"Hard rule: {streak} consecutive losses — disabled until reset"
                overrides["hard_rules_triggered"].append(f"{strat_name}_losing_streak_{streak}")
                logger.warning(f"{strat_name}: {streak} consecutive losses — DISABLED")
        elif streak >= 3:
            s = overrides["strategy_overrides"].get(strat_name)
            if s:
                s["risk_multiplier"] = min(s["risk_multiplier"], 0.5)
                s["reason"] = f"Hard rule: {streak} consecutive losses — risk halved"
                overrides["hard_rules_triggered"].append(f"{strat_name}_losing_streak_{streak}")
                logger.info(f"{strat_name}: {streak} consecutive losses — risk halved")

    # ── Rule 4: Dashboard freshness check ──
    gen_at = dashboard.get("generated_at", "")
    if gen_at:
        try:
            gen_dt = datetime.fromisoformat(gen_at)
            age_min = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 60
            if age_min > 20:
                overrides["hard_rules_triggered"].append("stale_dashboard")
                for name in overrides["strategy_overrides"]:
                    s = overrides["strategy_overrides"][name]
                    s["risk_multiplier"] = min(s["risk_multiplier"], 0.5)
                logger.warning(f"Dashboard is {age_min:.0f}min old — conservative mode")
        except Exception:
            pass

    return overrides


# ═══════════════════════════════════════════
#  LLM REASONING (HuggingFace Inference API)
# ═══════════════════════════════════════════

def build_llm_messages(state: dict, hard_overrides: dict) -> list[dict]:
    """Build chat messages for the LLM."""
    dashboard = state.get("dashboard", {}) or {}
    account = dashboard.get("account", {})
    strategies = dashboard.get("strategies", [])

    # Strategy summaries for the LLM
    strat_summaries = []
    for s in strategies:
        hb = s.get("heartbeat", {})
        live = s.get("live", {})
        strat_summaries.append({
            "name": s["name"],
            "pair": s.get("pair"),
            "alive": hb.get("alive", False),
            "signal_summary": hb.get("signal_summary", "unknown"),
            "open_trades": len(hb.get("open_trades", [])),
            "closed_trades": live.get("closed_trades", 0),
            "mt5_pnl_chf": live.get("mt5_pnl_chf"),
            "win_rate": live.get("win_rate"),
            "wins": live.get("wins", 0),
            "losses": live.get("losses", 0),
        })

    recent = state.get("db_recent_trades", [])[:15]

    system_msg = """You are an FTMO trading risk supervisor managing a CHF 80,000 FTMO challenge account with 5 strategies.

RULES:
- Never recommend risk_multiplier above 1.0
- Be conservative: when uncertain, reduce risk
- Output ONLY valid JSON, no other text
- Your recommendations are suggestions — hard rules always override you"""

    user_msg = f"""ACCOUNT STATE:
- Balance: CHF {account.get('balance', 80000):,.2f}
- P&L from start: CHF {account.get('pnl_from_start', 0):,.2f}
- Daily P&L: {account.get('daily_pnl_pct', 0):.2f}%
- FTMO rules: max 5% daily loss, max 10% total DD, target +10% (Step 1)

STRATEGY STATUS:
{json.dumps(strat_summaries, indent=2)}

RECENT TRADES (last 48h, newest first):
{json.dumps(recent, indent=2)}

HARD RULES ALREADY APPLIED:
{json.dumps(hard_overrides.get('hard_rules_triggered', []))}

Analyze the portfolio and recommend risk adjustments. Consider:
1. Market regime for each strategy (trending/ranging/volatile)
2. Recent performance and momentum
3. Correlation risk between strategies
4. Overall portfolio heat

Output ONLY this JSON format:
{{
  "reasoning": "1-2 sentence analysis",
  "strategy_adjustments": {{
    "STRATEGY_NAME": {{"risk_multiplier": 0.5, "reason": "brief"}}
  }},
  "portfolio_heat": "cool|warm|hot",
  "regime": "trending|ranging|volatile|crisis"
}}"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def call_llm(messages: list[dict]) -> dict | None:
    """Call HuggingFace Inference API via SDK, return parsed JSON or None."""
    if not HF_TOKEN:
        logger.warning("No HF token — skipping LLM call")
        return None

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)
        resp = client.chat_completion(
            model=HF_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        logger.info(f"LLM raw response: {text[:300]}")

        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            parsed = json.loads(json_str)
            logger.info(f"LLM reasoning: {parsed.get('reasoning', 'no reasoning')}")
            return parsed
        else:
            logger.warning(f"No JSON found in LLM response: {text[:200]}")
            return None

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM call error: {e}")
        return None


def merge_llm_into_overrides(overrides: dict, llm_response: dict) -> dict:
    """
    Merge LLM suggestions into overrides. Hard rules always win (take the min).
    LLM can only reduce risk, never increase.
    """
    if not llm_response:
        return overrides

    overrides["llm_used"] = True
    overrides["llm_regime"] = llm_response.get("regime", "unknown")
    overrides["llm_portfolio_heat"] = llm_response.get("portfolio_heat", "unknown")
    overrides["llm_reasoning"] = llm_response.get("reasoning", "")

    adjustments = llm_response.get("strategy_adjustments", {})
    for strat_name, adj in adjustments.items():
        if strat_name not in overrides["strategy_overrides"]:
            continue  # ignore unknown strategies

        llm_mult = adj.get("risk_multiplier", 1.0)
        # Clamp: LLM can only reduce, never exceed 1.0
        llm_mult = max(0.0, min(1.0, llm_mult))

        existing = overrides["strategy_overrides"][strat_name]

        # Hard rule wins: take the minimum (more conservative)
        if not existing["enabled"]:
            continue  # hard rule disabled it, LLM can't re-enable

        final_mult = min(existing["risk_multiplier"], llm_mult)
        if final_mult < existing["risk_multiplier"]:
            existing["risk_multiplier"] = final_mult
            llm_reason = adj.get("reason", "LLM adjustment")
            existing["reason"] = f"LLM: {llm_reason}"
            logger.info(f"LLM adjusted {strat_name}: multiplier={final_mult:.2f} — {llm_reason}")

    return overrides


# ═══════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════

def write_overrides(overrides: dict):
    """Write risk overrides with exclusive lock."""
    now = datetime.now(timezone.utc)
    overrides["version"] = 1
    overrides["generated_at"] = now.isoformat()
    overrides["generated_by"] = "ftmo_agent"
    overrides["expires_at"] = (now + timedelta(minutes=OVERRIDE_TTL_MINUTES)).isoformat()

    try:
        with open(OVERRIDE_FILE, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(overrides, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)
        logger.info(f"Overrides written → {OVERRIDE_FILE.name}")
    except Exception as e:
        logger.error(f"Failed to write overrides: {e}")


def send_telegram(msg: str):
    """Send Telegram alert for critical actions."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=5
        )
    except Exception:
        pass


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("FTMO Agent — Strategy Supervisor starting")
    start = time.time()

    # 1. Collect all data
    state = collect_world_state()
    logger.info(f"Data collected: dashboard={'yes' if state['dashboard'] else 'NO'}, "
                f"risk_state={'yes' if state['risk_state'] else 'NO'}, "
                f"recent_trades={len(state['db_recent_trades'])}")

    # 2. Apply hard rules
    overrides = apply_hard_rules(state)
    overrides["llm_used"] = False
    overrides["llm_regime"] = None
    overrides["llm_portfolio_heat"] = None

    # 3. Call LLM (only if no global halt and data is fresh)
    if not overrides["global_halt"] and state.get("dashboard"):
        messages = build_llm_messages(state, overrides)
        logger.info(f"Calling LLM ({HF_MODEL})...")
        llm_response = call_llm(messages)
        overrides = merge_llm_into_overrides(overrides, llm_response)
    else:
        logger.info("Skipping LLM (global halt or no data)")

    # 4. Write overrides
    write_overrides(overrides)

    # 5. Alert on critical actions
    if overrides["global_halt"]:
        rules = ", ".join(overrides["hard_rules_triggered"])
        send_telegram(f"🚨 *FTMO Agent: GLOBAL HALT*\nRules: {rules}")

    disabled = [n for n, s in overrides["strategy_overrides"].items() if not s["enabled"]]
    if disabled:
        send_telegram(f"⚠️ *FTMO Agent: Strategies disabled*\n{', '.join(disabled)}")

    elapsed = time.time() - start
    logger.info(f"Agent cycle done in {elapsed:.1f}s — "
                f"halt={overrides['global_halt']}, "
                f"rules={overrides['hard_rules_triggered']}, "
                f"llm={overrides['llm_used']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
