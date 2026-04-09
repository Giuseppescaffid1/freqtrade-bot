"""
FTMO Oil Strategy — News-Driven USOIL Trading

Two modes:
  1. LIVE: Reads oil_sentiment.json from oil_news_collector.py for real-time news signals
  2. BACKTEST: Uses technical volatility detection as news proxy (ATR spike = news event)

Entry logic:
  - News sentiment + trend confirmation (EMA crossover)
  - Strong sentiment overrides trend filter (big news moves markets regardless)
  - Volume/volatility confirmation (news creates volume)

Exit logic:
  - Sentiment reversal
  - RSI extreme (overbought/oversold)
  - ATR-based trailing stop
  - Max hold time (news becomes stale after 4-8 hours)

Risk: Integrated with ftmo_risk_manager.py (shared FTMO account)

Run: python ftmo_oil.py
"""

import os
import sys
import json
import asyncio
import logging
import numpy as np
import httpx
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ftmo_risk_manager as risk_mgr

# ── Configuration ──
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN", "")
METAAPI_ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID", "")
METAAPI_REGION = "london"

SYMBOL = "USOIL.cash"
TIMEFRAME = "15m"
POLL_INTERVAL = 60  # check every 60s

# Risk — from shared manager
OIL_SOURCE = "Oil"
OIL_PROFILE = risk_mgr.get_profile(OIL_SOURCE) if "Oil" in risk_mgr.SOURCE_PROFILES else {
    "risk_per_trade": 0.01, "max_positions": 1, "default_sl_pct": 0.03
}
RISK_PER_TRADE = OIL_PROFILE["risk_per_trade"]
MAX_OIL_POSITIONS = OIL_PROFILE["max_positions"]
FTMO_INITIAL_BALANCE = risk_mgr.FTMO_INITIAL_BALANCE

# Indicator parameters
EMA_FAST = 20
EMA_SLOW = 50
EMA_MACRO = 200
ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14

# News sentiment thresholds
SENTIMENT_FILE = Path("/home/giuseppe/freqtrade-bot/oil_sentiment.json")
VESSEL_INTEL_FILE = Path("/home/giuseppe/freqtrade-bot/vessel_intelligence.json")
STRONG_SENTIMENT_THRESHOLD = 3.0  # strong signal overrides trend
WEAK_SENTIMENT_THRESHOLD = 1.5    # needs trend confirmation
SENTIMENT_MAX_AGE_SECONDS = 300   # sentiment data must be < 5 min old
VESSEL_INTEL_MAX_AGE_SECONDS = 600  # vessel data can be up to 10 min old

# ATR-based SL/TP
ATR_SL_MULT = 1.5       # stoploss = 1.5x ATR (tight)
ATR_TP_MULT = 4.5       # take profit = 4.5x ATR (R:R = 1:3)
MAX_HOLD_CANDLES = 24    # max 6 hours (24 x 15min) — news trades are fast
MIN_HOLD_CANDLES = 3     # min 45 min — give trades time to work before EMA exit
COOLDOWN_CANDLES = 4     # wait 1 hour after closing before re-entering

# Session filter (UTC)
SKIP_HOURS_UTC = [22, 23, 0, 1]  # low liquidity

LOG_FILE = "ftmo_oil.log"
STATE_FILE = "ftmo_oil_state.json"
HEALTH_FILE = Path("/home/giuseppe/freqtrade-bot/ftmo_oil_health.json")

# ── Logging (with rotation — prevents 300MB log files) ──
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=3),  # 10MB x 3
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ftmo_oil")
# Silence noisy MetaApi loggers — they drown out actual strategy logs
for noisy in ("engineio", "socketio", "urllib3", "httpcore", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════
# INDICATORS (reuse from ftmo_gold.py)
# ═══════════════════════════════════════════════════════════════

def calc_ema(values, period):
    result = np.full_like(values, np.nan, dtype=float)
    if len(values) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def calc_atr(highs, lows, closes, period=14):
    n = len(closes)
    if n < 2:
        return np.full(n, 0.0)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr_vals = np.full(n, np.nan)
    if n >= period:
        atr_vals[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period
    return np.nan_to_num(atr_vals, nan=0.0)


def calc_rsi(closes, period=14):
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full(len(closes), np.nan)
    avg_loss = np.full(len(closes), np.nan)
    if len(gains) < period:
        return np.full(len(closes), 50.0)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100.0 - (100.0 / (1.0 + rs))


def calc_adx(highs, lows, closes, period=14):
    n = len(closes)
    if n < period * 2:
        return np.full(n, 0.0)
    tr = np.maximum(highs[1:] - lows[1:],
                     np.maximum(np.abs(highs[1:] - closes[:-1]),
                                np.abs(lows[1:] - closes[:-1])))
    plus_dm = np.where((highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                        np.maximum(highs[1:] - highs[:-1], 0), 0)
    minus_dm = np.where((lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                         np.maximum(lows[:-1] - lows[1:], 0), 0)
    atr_v = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    atr_v[period] = np.mean(tr[:period])
    sm_plus = np.mean(plus_dm[:period])
    sm_minus = np.mean(minus_dm[:period])
    for i in range(period, len(tr)):
        atr_v[i + 1] = (atr_v[i] * (period - 1) + tr[i]) / period
        sm_plus = (sm_plus * (period - 1) + plus_dm[i]) / period
        sm_minus = (sm_minus * (period - 1) + minus_dm[i]) / period
        if atr_v[i + 1] > 0:
            plus_di[i + 1] = 100 * sm_plus / atr_v[i + 1]
            minus_di[i + 1] = 100 * sm_minus / atr_v[i + 1]
    dx = np.where((plus_di + minus_di) > 0,
                   100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_vals = np.full(n, np.nan)
    valid_dx = dx[~np.isnan(dx)]
    if len(valid_dx) >= period:
        adx_vals[np.where(~np.isnan(dx))[0][period - 1]] = np.mean(valid_dx[:period])
        idx = np.where(~np.isnan(dx))[0]
        for i in range(period, len(idx)):
            adx_vals[idx[i]] = (adx_vals[idx[i - 1]] * (period - 1) + dx[idx[i]]) / period
    return np.nan_to_num(adx_vals, nan=0.0)


# ═══════════════════════════════════════════════════════════════
# OIL STRATEGY
# ═══════════════════════════════════════════════════════════════

class OilStrategy:
    """News-driven USOIL strategy for FTMO."""

    def __init__(self):
        self.metaapi = None
        self.mt_account = None
        self.mt_connection = None
        self.state_file = Path(STATE_FILE)
        self.position = None
        self.hold_candles = 0  # how many candles we've held current position
        self.last_candle_time = None
        self.cooldown_remaining = 0  # candles to wait before re-entering
        self.last_indicators = {}    # for health file reporting
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.position = data.get("position")
                self.hold_candles = data.get("hold_candles", 0)
                self.last_candle_time = data.get("last_candle_time")
                self.cooldown_remaining = data.get("cooldown_remaining", 0)
                logger.info(f"Loaded state: position={self.position}, hold={self.hold_candles}, cooldown={self.cooldown_remaining}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")

    def _save_state(self):
        try:
            data = {
                "position": self.position,
                "hold_candles": self.hold_candles,
                "last_candle_time": self.last_candle_time,
                "cooldown_remaining": self.cooldown_remaining,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def read_sentiment(self) -> dict:
        """Read current oil sentiment from collector."""
        try:
            if not SENTIMENT_FILE.exists():
                return {"signal": "NEUTRAL", "score": 0, "stale": True}
            data = json.loads(SENTIMENT_FILE.read_text())
            updated = datetime.fromisoformat(data.get("updated_at", "2000-01-01T00:00:00+00:00"))
            age = (datetime.now(timezone.utc) - updated).total_seconds()
            if age > SENTIMENT_MAX_AGE_SECONDS:
                return {"signal": "NEUTRAL", "score": 0, "stale": True}
            return {
                "signal": data.get("signal", "NEUTRAL"),
                "score": data.get("composite_score", 0),
                "bullish": data.get("bullish_count", 0),
                "bearish": data.get("bearish_count", 0),
                "stale": False,
            }
        except Exception as e:
            logger.error(f"Error reading sentiment: {e}")
            return {"signal": "NEUTRAL", "score": 0, "stale": True}

    def read_vessel_intelligence(self) -> dict:
        """Read vessel tracking intelligence for supply signals."""
        try:
            if not VESSEL_INTEL_FILE.exists():
                return {"signal": "NEUTRAL", "score": 0, "confidence": 0, "stale": True}
            data = json.loads(VESSEL_INTEL_FILE.read_text())
            updated = datetime.fromisoformat(data.get("updated_at", "2000-01-01T00:00:00+00:00"))
            age = (datetime.now(timezone.utc) - updated).total_seconds()
            if age > VESSEL_INTEL_MAX_AGE_SECONDS:
                return {"signal": "NEUTRAL", "score": 0, "confidence": 0, "stale": True}

            # Check actual IMF data freshness — the file is refreshed every 10min
            # but the underlying IMF data may be days old
            details = data.get("details", {})
            hormuz_imf = details.get("hormuz_imf", {})
            latest_day = hormuz_imf.get("latest_day", {})
            latest_date_str = latest_day.get("date", "")
            imf_data_stale = True
            if latest_date_str:
                try:
                    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    imf_age_days = (datetime.now(timezone.utc) - latest_date).days
                    imf_data_stale = imf_age_days > 3  # IMF data older than 3 days is stale
                except Exception:
                    pass

            raw_score = data.get("score", 0)
            confidence = data.get("confidence", 0)

            # Degrade vessel score if underlying data is stale
            if imf_data_stale:
                raw_score = raw_score * 0.3  # heavily discount stale IMF data
                confidence = confidence * 0.3
                logger.info(f"Vessel IMF data stale (latest: {latest_date_str}), score degraded {data.get('score', 0):.1f}→{raw_score:.1f}")

            return {
                "signal": data.get("signal", "NEUTRAL"),
                "score": raw_score,
                "confidence": confidence,
                "source_count": data.get("source_count", 0),
                "stale": False,
                "imf_stale": imf_data_stale,
            }
        except Exception as e:
            logger.error(f"Error reading vessel intelligence: {e}")
            return {"signal": "NEUTRAL", "score": 0, "confidence": 0, "stale": True}

    async def connect(self):
        """Connect to FTMO MT5 via MetaApi. Reuses existing MetaApi instance."""
        if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID:
            logger.error("MetaApi credentials not set!")
            return False
        try:
            from metaapi_cloud_sdk import MetaApi

            # Reuse MetaApi instance — creating new ones causes connection conflicts
            # with ftmo_gold and ftmo_bridge which share the same account
            if not self.metaapi:
                self.metaapi = MetaApi(METAAPI_TOKEN)

            self.mt_account = await self.metaapi.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)

            if self.mt_account.state != "DEPLOYED":
                await self.mt_account.deploy()
                logger.info("Account deploying...")

            await self.mt_account.wait_connected(timeout_in_seconds=180)

            # Close previous RPC connection if exists
            if self.mt_connection:
                try:
                    await self.mt_connection.close()
                except Exception:
                    pass

            self.mt_connection = self.mt_account.get_rpc_connection()
            await self.mt_connection.connect()
            await self.mt_connection.wait_synchronized(timeout_in_seconds=180)

            info = await self.mt_connection.get_account_information()
            logger.info(f"Connected: balance={info['balance']} {info['currency']}, equity={info['equity']}")
            risk_mgr.update_balance_tracking(info["equity"], info["balance"])
            self._last_successful_sync = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def _check_connection_health(self) -> bool:
        """Quick health check — try to get account info. Returns True if healthy."""
        try:
            info = await asyncio.wait_for(
                self.mt_connection.get_account_information(), timeout=15
            )
            return info is not None and "balance" in info
        except Exception:
            return False

    async def fetch_candles(self, limit=500):
        """
        Fetch recent USOIL candles from MetaApi REST (not WebSocket).
        Retries on 504/503 gateway errors.
        """
        url = (f"https://mt-market-data-client-api-v1.{METAAPI_REGION}.agiliumtrade.ai/"
               f"users/current/accounts/{METAAPI_ACCOUNT_ID}/"
               f"historical-market-data/symbols/{SYMBOL}/timeframes/{TIMEFRAME}/candles")
        # startTime is the END anchor — API returns candles going backward from it
        start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        for attempt in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        url, headers={"auth-token": METAAPI_TOKEN},
                        params={"startTime": start, "limit": limit}, timeout=30,
                    )
                    if resp.status_code == 200:
                        candles = resp.json()
                        if candles:
                            last_time = candles[-1].get("time", "")
                            last_close = candles[-1].get("close", 0)
                            logger.info(f"Candles: {len(candles)}, last={last_time}, close={last_close}")
                        return candles
                    elif resp.status_code in (502, 503, 504):
                        logger.warning(f"Candle fetch {resp.status_code}, retry {attempt+1}/3...")
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Candle fetch failed: {resp.status_code}")
                        return []
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                logger.warning(f"Candle fetch network error: {e}, retry {attempt+1}/3...")
                await asyncio.sleep(5 * (attempt + 1))
        logger.error("Candle fetch failed after 3 retries")
        return []

    def process_candles(self, candles, sentiment: dict, vessel_intel: dict = None):
        """
        Process candles with news sentiment + vessel intelligence to generate trade signals.

        Signal fusion:
          - News sentiment (primary): real-time headlines, keyword scoring
          - Vessel intelligence (secondary): chokepoint tanker traffic, supply disruption
          - Combined: news + vessel amplify each other's signal

        Returns action dict or None.
        """
        if len(candles) < EMA_MACRO + 10:
            return None

        vessel_intel = vessel_intel or {}

        closes = np.array([c["close"] for c in candles], dtype=float)
        highs = np.array([c["high"] for c in candles], dtype=float)
        lows = np.array([c["low"] for c in candles], dtype=float)
        opens = np.array([c["open"] for c in candles], dtype=float)

        ema_fast = calc_ema(closes, EMA_FAST)
        ema_slow = calc_ema(closes, EMA_SLOW)
        ema_macro = calc_ema(closes, EMA_MACRO)
        atr_vals = calc_atr(highs, lows, closes, ATR_PERIOD)
        rsi_vals = calc_rsi(closes, RSI_PERIOD)
        adx_vals = calc_adx(highs, lows, closes, ADX_PERIOD)

        c = candles[-1]
        candle_time = c.get("time", "")

        # ── SAFEGUARD: reject stale candle data ──
        # If last candle is older than 30 min, data is stale — don't trade on ghosts
        try:
            candle_dt = datetime.fromisoformat(candle_time.replace("Z", "+00:00"))
            candle_age_min = (datetime.now(timezone.utc) - candle_dt).total_seconds() / 60
            if candle_age_min > 30:
                logger.warning(f"Stale candles: last candle {candle_age_min:.0f}min old ({candle_time}), skipping")
                return None
        except Exception:
            pass

        if candle_time == self.last_candle_time:
            return None
        self.last_candle_time = candle_time

        cur_close = closes[-1]
        cur_atr = atr_vals[-1]
        cur_rsi = rsi_vals[-1]
        cur_adx = adx_vals[-1]
        cur_ema_fast = ema_fast[-1]
        cur_ema_slow = ema_slow[-1]
        cur_ema_macro = ema_macro[-1]

        # ATR spike detection (proxy for news volatility)
        atr_sma = np.mean(atr_vals[-50:]) if len(atr_vals) >= 50 else cur_atr
        atr_spike = cur_atr > atr_sma * 1.5

        uptrend = cur_ema_fast > cur_ema_slow
        downtrend = cur_ema_fast < cur_ema_slow
        macro_bullish = cur_close > cur_ema_macro
        macro_bearish = cur_close < cur_ema_macro

        news_score = sentiment.get("score", 0)
        news_signal = sentiment.get("signal", "NEUTRAL")
        news_stale = sentiment.get("stale", True)

        # Vessel intelligence
        vessel_score = vessel_intel.get("score", 0)
        vessel_signal = vessel_intel.get("signal", "NEUTRAL")
        vessel_confidence = vessel_intel.get("confidence", 0)
        vessel_stale = vessel_intel.get("stale", True)

        # ── COMBINED SIGNAL FUSION ──
        # News is primary (weight 0.6), vessel is secondary (weight 0.4)
        # Vessel amplifies news: if both agree, stronger signal
        combined_score = news_score
        if not vessel_stale and vessel_confidence > 0.1:
            combined_score = (news_score * 0.6) + (vessel_score * 0.4)

        # Determine combined signal level
        if combined_score >= 3.0:
            combined_signal = "STRONG_BULLISH"
        elif combined_score >= 1.5:
            combined_signal = "BULLISH"
        elif combined_score <= -3.0:
            combined_signal = "STRONG_BEARISH"
        elif combined_score <= -1.5:
            combined_signal = "BEARISH"
        else:
            combined_signal = "NEUTRAL"

        # Use combined signal if vessel data available, else fall back to news only
        effective_signal = combined_signal if not vessel_stale else news_signal
        effective_score = combined_score if not vessel_stale else news_score
        effective_stale = news_stale and vessel_stale  # both must be stale to block

        vessel_tag = ""
        if not vessel_stale:
            vessel_tag = f", vessel={vessel_signal}({vessel_score:+.1f}|{vessel_confidence:.0%})"

        logger.info(
            f"close={cur_close:.2f}, EMA{EMA_FAST}={cur_ema_fast:.2f}, "
            f"EMA{EMA_SLOW}={cur_ema_slow:.2f}, RSI={cur_rsi:.1f}, ADX={cur_adx:.1f}, "
            f"ATR={cur_atr:.2f}{'*' if atr_spike else ''}, "
            f"news={news_signal}({news_score:+.1f}){vessel_tag}, "
            f"combined={effective_signal}({effective_score:+.1f})"
        )

        # Store for health file
        self.last_indicators = {
            "price": round(cur_close, 2),
            "ema_20": round(cur_ema_fast, 2),
            "ema_50": round(cur_ema_slow, 2),
            "ema_200": round(cur_ema_macro, 2),
            "rsi": round(cur_rsi, 1),
            "adx": round(cur_adx, 1),
            "atr": round(cur_atr, 2),
            "atr_spike": bool(atr_spike),
            "uptrend": bool(uptrend),
            "news_signal": news_signal,
            "news_score": round(news_score, 1),
            "news_stale": news_stale,
            "vessel_signal": vessel_signal if not vessel_stale else None,
            "vessel_score": round(vessel_score, 1) if not vessel_stale else None,
            "combined_signal": effective_signal,
            "combined_score": round(effective_score, 1),
            "cooldown": self.cooldown_remaining,
        }

        # ── EXIT LOGIC (if in position) ──
        if self.position:
            self.hold_candles += 1
            pos_dir = self.position.get("direction")

            # Max hold time — news becomes stale
            if self.hold_candles >= MAX_HOLD_CANDLES:
                return {"action": "close", "reason": "max_hold_time"}

            # Sentiment reversal exit (always check, even during min hold)
            if pos_dir == "long" and effective_signal in ("STRONG_BEARISH",) and not effective_stale:
                return {"action": "close", "reason": "sentiment_reversal"}
            if pos_dir == "short" and effective_signal in ("STRONG_BULLISH",) and not effective_stale:
                return {"action": "close", "reason": "sentiment_reversal"}

            # Min hold time — give trades room to breathe before technical exits
            if self.hold_candles < MIN_HOLD_CANDLES:
                return None  # hold — only SL/TP (on MT5) or strong reversal can close early

            # RSI extreme exit
            if pos_dir == "long" and cur_rsi > 78:
                return {"action": "close", "reason": "rsi_overbought"}
            if pos_dir == "short" and cur_rsi < 22:
                return {"action": "close", "reason": "rsi_oversold"}

            # Moderate sentiment reversal (after min hold)
            if pos_dir == "long" and effective_signal == "BEARISH" and not effective_stale:
                return {"action": "close", "reason": "sentiment_reversal"}
            if pos_dir == "short" and effective_signal == "BULLISH" and not effective_stale:
                return {"action": "close", "reason": "sentiment_reversal"}

            # EMA cross exit (after min hold)
            if pos_dir == "long" and cur_ema_fast < cur_ema_slow:
                return {"action": "close", "reason": "ema_cross_bearish"}
            if pos_dir == "short" and cur_ema_fast > cur_ema_slow:
                return {"action": "close", "reason": "ema_cross_bullish"}

            return None  # hold

        # ── COOLDOWN CHECK ──
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self._save_state()
            logger.info(f"Cooldown: {self.cooldown_remaining} candles remaining")
            return None

        # ── ENTRY LOGIC (no position) ──
        if effective_stale:
            # No fresh data from either source — don't trade
            return None

        # ALL entries require trend confirmation (EMA alignment)
        # Strong signal relaxes ADX/RSI filters but NEVER overrides trend direction
        # This prevents the churn loop where vessel signal opens against the trend

        # STRONG BULLISH: relaxed filters but still need uptrend
        if effective_signal == "STRONG_BULLISH" and uptrend and cur_adx > 15:
            sl = cur_close - (cur_atr * ATR_SL_MULT)
            tp = cur_close + (cur_atr * ATR_TP_MULT)
            reason = "strong_bullish"
            if not vessel_stale and vessel_score > 1.0:
                reason += "_vessel_confirmed"
            return {"action": "open", "direction": "long", "entry": cur_close,
                    "stoploss": sl, "takeprofit": tp, "atr": cur_atr,
                    "reason": reason}

        if effective_signal == "STRONG_BEARISH" and downtrend and cur_adx > 15:
            sl = cur_close + (cur_atr * ATR_SL_MULT)
            tp = cur_close - (cur_atr * ATR_TP_MULT)
            reason = "strong_bearish"
            if not vessel_stale and vessel_score < -1.0:
                reason += "_vessel_confirmed"
            return {"action": "open", "direction": "short", "entry": cur_close,
                    "stoploss": sl, "takeprofit": tp, "atr": cur_atr,
                    "reason": reason}

        # MODERATE signal: needs trend + stricter filters
        if effective_signal == "BULLISH" and uptrend and cur_rsi < 65 and cur_adx > 20:
            sl = cur_close - (cur_atr * ATR_SL_MULT)
            tp = cur_close + (cur_atr * ATR_TP_MULT)
            reason = "bullish_trend_confirm"
            if not vessel_stale and vessel_score > 0.5:
                reason += "_vessel_boost"
            return {"action": "open", "direction": "long", "entry": cur_close,
                    "stoploss": sl, "takeprofit": tp, "atr": cur_atr,
                    "reason": reason}

        if effective_signal == "BEARISH" and downtrend and cur_rsi > 35 and cur_adx > 20:
            sl = cur_close + (cur_atr * ATR_SL_MULT)
            tp = cur_close - (cur_atr * ATR_TP_MULT)
            reason = "bearish_trend_confirm"
            if not vessel_stale and vessel_score < -0.5:
                reason += "_vessel_boost"
            return {"action": "open", "direction": "short", "entry": cur_close,
                    "stoploss": sl, "takeprofit": tp, "atr": cur_atr,
                    "reason": reason}

        # VESSEL-ONLY signal: strong vessel data can trigger even without news
        # (e.g., Hormuz blockade detected before news picks it up)
        # RSI filter: don't buy overbought (>70) or sell oversold (<30)
        if not vessel_stale and vessel_confidence >= 0.3 and abs(vessel_score) >= 2.5:
            if vessel_signal in ("STRONG_BULLISH", "BULLISH") and uptrend and cur_adx > 20 and cur_rsi < 70:
                sl = cur_close - (cur_atr * ATR_SL_MULT)
                tp = cur_close + (cur_atr * ATR_TP_MULT)
                return {"action": "open", "direction": "long", "entry": cur_close,
                        "stoploss": sl, "takeprofit": tp, "atr": cur_atr,
                        "reason": "vessel_supply_disruption"}
            if vessel_signal in ("STRONG_BEARISH", "BEARISH") and downtrend and cur_adx > 20 and cur_rsi > 30:
                sl = cur_close + (cur_atr * ATR_SL_MULT)
                tp = cur_close - (cur_atr * ATR_TP_MULT)
                return {"action": "open", "direction": "short", "entry": cur_close,
                        "stoploss": sl, "takeprofit": tp, "atr": cur_atr,
                        "reason": "vessel_oversupply"}

        return None

    # ═══════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def calculate_volume(self, entry_price, stoploss, account_balance):
        """
        USOIL lot sizing for FTMO MT5.

        FTMO USOIL.cash specs (from MetaApi):
          - digits: 3 (prices like 98.560)
          - contractSize: 100
          - tickSize: 0.001
          - volumeStep: 0.01

        1 lot = 100 barrels → $1 move = $100 per lot
        Account is CHF — exchange rate ~0.80 CHF/USD
        """
        # Apply agent supervisor multiplier
        adjusted_risk, agent_mult, agent_reason = risk_mgr.get_agent_adjusted_risk(OIL_SOURCE)
        if agent_mult < 1.0:
            logger.info(f"Agent risk adj [Oil]: {RISK_PER_TRADE:.3%} × {agent_mult:.2f} = {adjusted_risk:.3%} ({agent_reason})")
        risk_amount = account_balance * adjusted_risk  # CHF
        sl_distance = abs(entry_price - stoploss)
        if sl_distance == 0:
            return 0.01
        # contractSize=100: 1 lot, $1 move = $100 per lot
        # Convert USD to CHF: ~0.80
        chf_per_usd = 0.80
        risk_per_lot = sl_distance * 100 * chf_per_usd
        volume = risk_amount / risk_per_lot
        volume = round(volume, 2)
        # FTMO limits: keep conservative
        return max(0.01, min(volume, 50.0))

    async def get_oil_positions(self):
        try:
            positions = await asyncio.wait_for(
                self.mt_connection.get_positions(), timeout=15
            )
            # Match USOIL.cash or USOIL
            return [p for p in positions if "USOIL" in p.get("symbol", "")]
        except asyncio.TimeoutError:
            logger.warning("get_positions timed out (15s)")
            return None  # None = unknown (don't assume closed)
        except Exception as e:
            logger.warning(f"Error getting positions: {str(e)[:150]}")
            return None  # None = unknown

    async def open_position(self, signal):
        info = await self.mt_connection.get_account_information()
        balance = info["balance"]
        volume = self.calculate_volume(signal["entry"], signal["stoploss"], balance)
        # USOIL.cash uses 3 decimal places on FTMO MT5 (digits=3, tickSize=0.001)
        sl = round(signal["stoploss"], 3)
        tp = round(signal["takeprofit"], 3)

        # Ensure minimum stop distance
        entry = signal["entry"]
        min_stop_distance = 0.05
        if abs(entry - sl) < min_stop_distance:
            if signal["direction"] == "long":
                sl = round(entry - min_stop_distance, 2)
            else:
                sl = round(entry + min_stop_distance, 2)

        logger.info(
            f"OPENING {signal['direction'].upper()} {volume} {SYMBOL} @ ~{signal['entry']:.2f}, "
            f"SL={sl}, TP={tp}, reason={signal.get('reason', 'signal')}"
        )

        try:
            # Open without stops first, then modify to add SL/TP
            # (some FTMO symbols reject inline stops on market orders)
            if signal["direction"] == "long":
                result = await self.mt_connection.create_market_buy_order(SYMBOL, volume)
            else:
                result = await self.mt_connection.create_market_sell_order(SYMBOL, volume)

            pos_id = result.get("positionId", result.get("orderId", "unknown"))
            logger.info(f"Position opened: {pos_id}, adding SL={sl}/TP={tp}...")

            # Add SL/TP via position modify — CRITICAL: never run naked
            await asyncio.sleep(2)  # wait for position to settle
            sl_set = False
            for retry in range(3):
                try:
                    await self.mt_connection.modify_position(str(pos_id), stop_loss=sl, take_profit=tp)
                    logger.info(f"SL/TP set on {pos_id}: SL={sl}, TP={tp}")
                    sl_set = True
                    break
                except Exception as mod_err:
                    logger.warning(f"SL/TP attempt {retry+1}/3 failed: {mod_err}")
                    await asyncio.sleep(2)

            if not sl_set:
                # SAFETY: close position immediately — never trade without stops
                logger.error(f"CRITICAL: Could not set SL/TP after 3 attempts — closing position {pos_id}")
                try:
                    await self.mt_connection.close_position(str(pos_id))
                    risk_mgr.unregister_position(OIL_SOURCE, str(pos_id))
                    self.position = None
                    self._save_state()
                    return False
                except Exception as close_err:
                    logger.error(f"Failed to close naked position: {close_err}")
                    # Still track it so we can close next cycle
                    pass

            self.position = {
                "id": str(pos_id),
                "direction": signal["direction"],
                "entry": signal["entry"],
                "stoploss": sl,
                "takeprofit": tp,
                "volume": volume,
                "reason": signal.get("reason", "signal"),
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }
            self.hold_candles = 0
            self._save_state()

            risk_mgr.register_position(
                OIL_SOURCE, str(pos_id), SYMBOL,
                signal["direction"], volume, RISK_PER_TRADE
            )
            return True
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return False

    async def close_position(self, reason="signal"):
        if not self.position:
            return
        pos_id = self.position["id"]
        logger.info(f"CLOSING {SYMBOL} position {pos_id}, reason: {reason}, held={self.hold_candles} candles")
        try:
            positions = await self.get_oil_positions()
            for p in positions:
                await self.mt_connection.close_position(p["id"])
                logger.info(f"Closed MT5 position {p['id']}")
            risk_mgr.unregister_position(OIL_SOURCE, pos_id)
            self.position = None
            self.hold_candles = 0
            self.cooldown_remaining = COOLDOWN_CANDLES
            logger.info(f"Cooldown set: {COOLDOWN_CANDLES} candles before next entry")
            self._save_state()
        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    # ═══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════

    async def sync(self):
        hour = datetime.now(timezone.utc).hour
        if hour in SKIP_HOURS_UTC:
            return
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return

        # Sync position state with MT5
        # get_oil_positions returns None on error (unknown), [] on confirmed empty
        mt5_positions = await self.get_oil_positions()
        if mt5_positions is None:
            # Connection issue — don't change position state, just skip this cycle
            logger.warning("Could not verify positions, skipping sync cycle")
            return
        if not mt5_positions and self.position:
            logger.info(f"Position closed by SL/TP: {self.position['direction']}")
            risk_mgr.unregister_position(OIL_SOURCE, self.position["id"])
            self.position = None
            self.hold_candles = 0
            self._save_state()
        elif mt5_positions and not self.position:
            p = mt5_positions[0]
            self.position = {
                "id": p["id"],
                "direction": "long" if p["type"] == "POSITION_TYPE_BUY" else "short",
                "entry": p["openPrice"],
                "volume": p["volume"],
            }
            self._save_state()

        # Read sentiment + vessel intelligence
        sentiment = self.read_sentiment()
        vessel_intel = self.read_vessel_intelligence()

        # Fetch candles and process
        candles = await self.fetch_candles(limit=500)
        if not candles or len(candles) < EMA_MACRO + 10:
            logger.warning(f"Insufficient candle data: {len(candles) if candles else 0}")
            return

        action = self.process_candles(candles, sentiment, vessel_intel)
        if not action:
            return

        if action.get("action") == "close":
            await self.close_position(reason=action.get("reason", "signal"))
            return

        if action.get("action") == "open" and not self.position:
            # Check FTMO limits via shared risk manager
            try:
                info = await self.mt_connection.get_account_information()
                within_limits, reason = risk_mgr.check_ftmo_limits(info["equity"], info["balance"])
                if not within_limits:
                    logger.warning(f"FTMO limits breached: {reason}")
                    return
                can_open, reason = risk_mgr.can_open_position(OIL_SOURCE, RISK_PER_TRADE)
                if not can_open:
                    logger.warning(f"Risk manager blocked: {reason}")
                    return
            except Exception as e:
                logger.error(f"Risk check failed: {e}")
                return

            if len(mt5_positions) >= MAX_OIL_POSITIONS:
                return
            await self.open_position(action)

    def _write_health(self, status: str, detail: str = ""):
        """Write health status for cron monitoring."""
        try:
            ind = self.last_indicators
            # Build signal explanation
            reasons = []
            if ind:
                # Trend
                if ind.get("uptrend"):
                    reasons.append(f"Trend: bullish (EMA20>50)")
                else:
                    reasons.append(f"Trend: bearish (EMA20<50)")

                # ADX
                adx = ind.get("adx", 0)
                reasons.append(f"ADX: {adx:.0f}" + (" ✓" if adx > 20 else " (weak)"))

                # RSI
                rsi = ind.get("rsi", 0)
                reasons.append(f"RSI: {rsi:.0f}")

                # News sentiment
                ns = ind.get("news_signal", "?")
                score = ind.get("combined_score", 0)
                if ind.get("news_stale"):
                    reasons.append(f"News: STALE")
                else:
                    reasons.append(f"News: {ns} ({score:+.1f})")

                # Vessel
                vs = ind.get("vessel_signal")
                if vs:
                    reasons.append(f"Vessel: {vs}")

                # Cooldown
                cd = ind.get("cooldown", 0)
                if cd > 0:
                    reasons.append(f"Cooldown: {cd} candles")

                # ATR spike
                if ind.get("atr_spike"):
                    reasons.append("ATR: spike!")

            data = {
                "status": status,
                "detail": detail,
                "position": self.position is not None,
                "last_candle": self.last_candle_time,
                "hold_candles": self.hold_candles,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "indicators": ind,
                "signal_explanation": " | ".join(reasons) if reasons else ("market closed — waiting for fresh candles" if status == "ok" else "starting up"),
            }
            HEALTH_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    async def run(self):
        logger.info("=" * 60)
        logger.info("FTMO Oil Strategy — News-Driven USOIL")
        logger.info(f"Symbol: {SYMBOL}, Timeframe: {TIMEFRAME}")
        logger.info(f"Risk: {RISK_PER_TRADE:.1%}, SL: {ATR_SL_MULT}x ATR, TP: {ATR_TP_MULT}x ATR")
        logger.info(f"News thresholds: strong={STRONG_SENTIMENT_THRESHOLD}, weak={WEAK_SENTIMENT_THRESHOLD}")
        logger.info(f"Max hold: {MAX_HOLD_CANDLES} candles ({MAX_HOLD_CANDLES * 15 / 60:.0f}h)")
        logger.info("=" * 60)

        self._last_successful_sync = None
        self._write_health("starting")

        # Initial connection with retries
        for attempt in range(5):
            connected = await self.connect()
            if connected:
                break
            wait = 30 * (attempt + 1)
            logger.warning(f"Connection attempt {attempt+1}/5 failed, retrying in {wait}s...")
            self._write_health("connecting", f"attempt {attempt+1}/5")
            await asyncio.sleep(wait)
        else:
            logger.error("Failed to connect after 5 attempts. Exiting.")
            self._write_health("dead", "failed all connection attempts")
            return

        logger.info(f"Starting oil strategy loop (poll every {POLL_INTERVAL}s)...")
        consecutive_errors = 0
        reconnect_backoff = 30

        while True:
            try:
                # Periodic health check — detect stale WebSocket before it causes errors
                if self._last_successful_sync:
                    since_last = (datetime.now(timezone.utc) - self._last_successful_sync).total_seconds()
                    if since_last > 600:  # 10 minutes without a successful sync
                        logger.warning(f"No successful sync for {since_last:.0f}s, checking connection...")
                        healthy = await self._check_connection_health()
                        if not healthy:
                            logger.warning("Connection unhealthy, forcing reconnect...")
                            await self.connect()

                await self.sync()
                self._last_successful_sync = datetime.now(timezone.utc)
                consecutive_errors = 0
                reconnect_backoff = 30
                self._write_health("ok")

            except Exception as e:
                consecutive_errors += 1
                err_msg = str(e)
                # Filter out noisy non-critical errors
                if "timed out" in err_msg.lower() or "failed to connect" in err_msg.lower():
                    log_fn = logger.warning
                else:
                    log_fn = logger.error
                log_fn(f"Sync error ({consecutive_errors}): {err_msg[:200]}")
                self._write_health("error", f"sync_error_{consecutive_errors}: {err_msg[:100]}")

                if consecutive_errors >= 3:
                    logger.warning(f"Reconnecting MetaApi (backoff={reconnect_backoff}s)...")
                    await asyncio.sleep(reconnect_backoff)
                    try:
                        success = await self.connect()
                        if success:
                            consecutive_errors = 0
                            reconnect_backoff = 30
                            logger.info("Reconnected successfully")
                        else:
                            reconnect_backoff = min(reconnect_backoff * 2, 600)  # max 10min
                            logger.error(f"Reconnect failed, next backoff={reconnect_backoff}s")
                    except Exception as re:
                        reconnect_backoff = min(reconnect_backoff * 2, 600)
                        logger.error(f"Reconnect exception: {re}")

            await asyncio.sleep(POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════════
# BACKTESTER — simulates news with ATR spikes
# ═══════════════════════════════════════════════════════════════

def backtest_oil(candles_file: str = "usoil_historical.json"):
    """
    Backtest oil strategy using ATR spikes as news proxy.
    Since we don't have historical news data, we use:
    - ATR spike (>1.5x average) = "news event"
    - Direction: EMA trend at time of spike
    """
    with open(candles_file) as f:
        candles = json.load(f)

    logger.info(f"Backtesting {len(candles)} candles...")

    closes = np.array([c["close"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    opens = np.array([c["open"] for c in candles], dtype=float)

    ema_fast = calc_ema(closes, EMA_FAST)
    ema_slow = calc_ema(closes, EMA_SLOW)
    ema_macro = calc_ema(closes, EMA_MACRO)
    atr_vals = calc_atr(highs, lows, closes, ATR_PERIOD)
    rsi_vals = calc_rsi(closes, RSI_PERIOD)
    adx_vals = calc_adx(highs, lows, closes, ADX_PERIOD)

    # Calculate rolling ATR average for spike detection
    atr_sma = np.full(len(atr_vals), np.nan)
    for i in range(50, len(atr_vals)):
        atr_sma[i] = np.mean(atr_vals[i - 50:i])

    # Backtest state
    balance = FTMO_INITIAL_BALANCE
    peak_balance = balance
    position = None
    hold_count = 0
    cooldown = 0
    trades = []
    daily_pnl = {}

    for i in range(EMA_MACRO + 10, len(candles)):
        cur_close = closes[i]
        cur_atr = atr_vals[i]
        cur_rsi = rsi_vals[i]
        cur_adx = adx_vals[i]
        cur_ema_fast = ema_fast[i]
        cur_ema_slow = ema_slow[i]
        cur_ema_macro = ema_macro[i]
        cur_atr_sma = atr_sma[i] if not np.isnan(atr_sma[i]) else cur_atr

        candle_date = candles[i].get("time", "")[:10]
        atr_spike = cur_atr > cur_atr_sma * 1.5
        uptrend = cur_ema_fast > cur_ema_slow
        downtrend = cur_ema_fast < cur_ema_slow

        # Skip weekends/low hours
        time_str = candles[i].get("time", "")
        if time_str:
            try:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                if dt.weekday() >= 5 or dt.hour in SKIP_HOURS_UTC:
                    continue
            except Exception:
                pass

        # ── EXIT ──
        if position:
            hold_count += 1
            pos_dir = position["direction"]

            exit_reason = None
            if hold_count >= MAX_HOLD_CANDLES:
                exit_reason = "max_hold"
            # SL/TP always checked regardless of min hold
            elif pos_dir == "long" and lows[i] <= position["stoploss"]:
                exit_reason = "stoploss"
                cur_close = position["stoploss"]
            elif pos_dir == "short" and highs[i] >= position["stoploss"]:
                exit_reason = "stoploss"
                cur_close = position["stoploss"]
            elif pos_dir == "long" and highs[i] >= position["takeprofit"]:
                exit_reason = "takeprofit"
                cur_close = position["takeprofit"]
            elif pos_dir == "short" and lows[i] <= position["takeprofit"]:
                exit_reason = "takeprofit"
                cur_close = position["takeprofit"]
            # Technical exits only after min hold time
            elif hold_count >= MIN_HOLD_CANDLES:
                if pos_dir == "long" and cur_rsi > 78:
                    exit_reason = "rsi_overbought"
                elif pos_dir == "short" and cur_rsi < 22:
                    exit_reason = "rsi_oversold"
                elif pos_dir == "long" and cur_ema_fast < cur_ema_slow:
                    exit_reason = "ema_cross"
                elif pos_dir == "short" and cur_ema_fast > cur_ema_slow:
                    exit_reason = "ema_cross"

            if exit_reason:
                entry = position["entry"]
                if pos_dir == "long":
                    pnl_pct = (cur_close - entry) / entry
                else:
                    pnl_pct = (entry - cur_close) / entry

                risk_amt = balance * RISK_PER_TRADE
                sl_dist = abs(entry - position["stoploss"])
                if sl_dist > 0:
                    pnl_usd = risk_amt * (pnl_pct * entry / sl_dist)
                else:
                    pnl_usd = 0

                balance += pnl_usd
                peak_balance = max(peak_balance, balance)

                trades.append({
                    "entry_time": position.get("time", ""),
                    "exit_time": candle_date,
                    "direction": pos_dir,
                    "entry": entry,
                    "exit": cur_close,
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "pnl_usd": round(pnl_usd, 2),
                    "exit_reason": exit_reason,
                    "hold_candles": hold_count,
                    "balance": round(balance, 2),
                })

                daily_pnl.setdefault(candle_date, 0)
                daily_pnl[candle_date] += pnl_usd

                position = None
                hold_count = 0
                cooldown = COOLDOWN_CANDLES
                continue

        # ── COOLDOWN ──
        if cooldown > 0:
            cooldown -= 1
            continue

        # ── ENTRY ──
        # EMA crossover + macro trend + ADX strength + RSI zone
        # This captures the same directional moves that news creates.
        # Live mode adds news sentiment for better timing.
        macro_bull = cur_close > cur_ema_macro
        macro_bear = cur_close < cur_ema_macro

        # Detect fresh EMA crossover (happened within last 8 candles = 2h)
        cross_up = (ema_fast[i] > ema_slow[i]) and (ema_fast[i-8] <= ema_slow[i-8])
        cross_down = (ema_fast[i] < ema_slow[i]) and (ema_fast[i-8] >= ema_slow[i-8])

        # Momentum: current candle confirms direction
        bullish_candle = closes[i] > opens[i]
        bearish_candle = closes[i] < opens[i]

        if not position and cur_adx > 28:
            # LONG: EMA cross up + macro bullish + RSI zone + bullish candle
            if cross_up and macro_bull and 30 < cur_rsi < 65 and bullish_candle:
                sl = cur_close - (cur_atr * ATR_SL_MULT)
                tp = cur_close + (cur_atr * ATR_TP_MULT)
                position = {
                    "direction": "long", "entry": cur_close,
                    "stoploss": sl, "takeprofit": tp, "time": candle_date,
                }
                hold_count = 0
            # SHORT: EMA cross down + macro bearish + RSI zone + bearish candle
            elif cross_down and macro_bear and 35 < cur_rsi < 70 and bearish_candle:
                sl = cur_close + (cur_atr * ATR_SL_MULT)
                tp = cur_close - (cur_atr * ATR_TP_MULT)
                position = {
                    "direction": "short", "entry": cur_close,
                    "stoploss": sl, "takeprofit": tp, "time": candle_date,
                }
                hold_count = 0

    # ── Results ──
    if not trades:
        print("No trades generated!")
        return

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    max_dd = 0
    peak = FTMO_INITIAL_BALANCE
    for t in trades:
        peak = max(peak, t["balance"])
        dd = (peak - t["balance"]) / peak
        max_dd = max(max_dd, dd)

    worst_day = min(daily_pnl.values()) if daily_pnl else 0
    best_day = max(daily_pnl.values()) if daily_pnl else 0

    print("\n" + "=" * 60)
    print("FTMO OIL STRATEGY — BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {candles[0]['time'][:10]} to {candles[-1]['time'][:10]}")
    print(f"Initial balance: CHF {FTMO_INITIAL_BALANCE:,.2f}")
    print(f"Final balance:   CHF {balance:,.2f}")
    print(f"Total P&L:       CHF {total_pnl:,.2f} ({total_pnl/FTMO_INITIAL_BALANCE*100:.2f}%)")
    print(f"Trades:          {len(trades)} ({len(wins)}W / {len(losses)}L)")
    print(f"Win rate:        {len(wins)/len(trades)*100:.1f}%")
    print(f"Avg win:         CHF {np.mean([t['pnl_usd'] for t in wins]):,.2f}" if wins else "Avg win: N/A")
    print(f"Avg loss:        CHF {np.mean([t['pnl_usd'] for t in losses]):,.2f}" if losses else "Avg loss: N/A")
    print(f"Max drawdown:    {max_dd*100:.2f}%")
    print(f"Best day:        CHF {best_day:,.2f}")
    print(f"Worst day:       CHF {worst_day:,.2f} ({worst_day/FTMO_INITIAL_BALANCE*100:.2f}%)")
    print(f"Avg hold:        {np.mean([t['hold_candles'] for t in trades]):.0f} candles "
          f"({np.mean([t['hold_candles'] for t in trades])*15/60:.1f}h)")
    print()
    print("FTMO Compliance:")
    print(f"  Max daily loss: {abs(worst_day/FTMO_INITIAL_BALANCE*100):.2f}% (limit: 5%)")
    print(f"  Max drawdown:   {max_dd*100:.2f}% (limit: 10%)")
    print(f"  Profit target:  {total_pnl/FTMO_INITIAL_BALANCE*100:.2f}% (target: 10%)")
    print("=" * 60)

    # Save trade log
    with open("ftmo_oil_backtest.json", "w") as f:
        json.dump({"trades": trades, "summary": {
            "total_pnl": total_pnl, "win_rate": len(wins)/len(trades),
            "max_dd": max_dd, "worst_day": worst_day, "trades": len(trades),
        }}, f, indent=2)
    print(f"Trade log saved to ftmo_oil_backtest.json")


async def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        backtest_oil()
    else:
        strategy = OilStrategy()
        await strategy.run()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   FTMO Oil Strategy v1.0                      ║
    ║   News-Driven USOIL Trading                   ║
    ║   MetaApi → FTMO MT5                          ║
    ╚═══════════════════════════════════════════════╝

    Usage:
      python ftmo_oil.py            # Live trading
      python ftmo_oil.py backtest   # Backtest with historical data
    """)
    asyncio.run(main())
