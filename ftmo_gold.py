"""
FTMO Gold Strategy v2 — 4-Phase Pullback Breakout on XAUUSD

Inspired by backtrader-pullback-window-xauusd (55% WR, 5.81% max DD, +44.75% in 5yr).
Adapted for live trading on FTMO MT5 via MetaApi.

4-Phase State Machine:
  1. SCANNING  — Wait for EMA crossover + macro filter + ADX confirmation
  2. ARMED     — Wait for 1-3 counter-trend candles (pullback confirmation)
  3. WINDOW    — Set breakout channel around pullback range, wait for breakout
  4. ENTRY     — Price breaks channel → open position with ATR-based SL/TP

Key features:
  - Global invalidation: opposite signal cancels entire setup
  - Failure boundary: if wrong side breaks, reset to ARMED (not full reset)
  - ATR-scaled channels adapt to volatility
  - Asymmetric R:R — SL=2.5x ATR, TP=10x ATR for longs (1:4 ratio)
  - FTMO risk management (shared account with crypto bridge)

Run: python ftmo_gold.py
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
from enum import Enum

import ftmo_risk_manager as risk_mgr

# ── Configuration ──
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN", "")
METAAPI_ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID", "")
METAAPI_REGION = "london"

SYMBOL = "XAUUSD"
TIMEFRAME = "15m"
POLL_INTERVAL = 60  # check every 60s

# Risk parameters — sourced from shared risk manager
GOLD_PROFILE = risk_mgr.get_profile("Gold")
RISK_PER_TRADE = GOLD_PROFILE["risk_per_trade"]  # 1.5%
FTMO_INITIAL_BALANCE = risk_mgr.FTMO_INITIAL_BALANCE
MAX_GOLD_POSITIONS = GOLD_PROFILE["max_positions"]

# ── Indicator parameters ──
EMA_CONFIRM = 3                # near-real-time confirmation EMA
EMA_FAST = 14                  # fast trend
EMA_SLOW = 24                  # slow trend
EMA_MACRO = 100                # macro filter (100-period, not 200 — gold trends faster)
ATR_PERIOD = 10                # ATR period (as per reference strategy)
ADX_PERIOD = 14
ADX_MIN = 20                   # minimum trend strength

# ── Pullback-Window parameters ──
PULLBACK_COUNT_LONG = 2        # need 2 red candles for long pullback
PULLBACK_COUNT_SHORT = 2       # need 2 green candles for short pullback
WINDOW_BARS = 6                # breakout window lasts 6 bars (1.5h on 15m)
WINDOW_PRICE_MULT = 0.5        # channel = pullback_range × 0.5 expansion
WINDOW_MAX_BARS = 10           # absolute max window duration

# ── ATR-based SL/TP (asymmetric R:R) ──
ATR_SL_MULT = 2.5              # stoploss = 2.5x ATR
ATR_TP_LONG = 10.0             # take profit longs = 10x ATR (R:R = 1:4)
ATR_TP_SHORT = 6.5             # take profit shorts = 6.5x ATR (R:R = 1:2.6)

# Trading sessions (UTC)
SKIP_HOURS_UTC = [22, 23, 0, 1]  # skip low-liquidity hours

LOG_FILE = "ftmo_gold.log"
STATE_FILE = "ftmo_gold_state.json"

# ── Logging ──
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=3),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ftmo_gold")

for noisy in ("engineio", "socketio", "urllib3", "httpcore", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════

class Phase(Enum):
    SCANNING = "SCANNING"
    ARMED_LONG = "ARMED_LONG"
    ARMED_SHORT = "ARMED_SHORT"
    WINDOW_LONG = "WINDOW_LONG"
    WINDOW_SHORT = "WINDOW_SHORT"


# ═══════════════════════════════════════════════════════════════
# INDICATOR FUNCTIONS
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


def calc_atr(highs, lows, closes, period=10):
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


# ═══════════════════════════════════════════════════════════════
# GOLD STRATEGY
# ═══════════════════════════════════════════════════════════════

class GoldStrategy:
    """4-Phase Pullback Breakout strategy for XAUUSD on FTMO."""

    def __init__(self):
        self.metaapi = None
        self.mt_account = None
        self.mt_connection = None
        self.state_file = Path(STATE_FILE)
        self.position = None

        # State machine
        self.phase = Phase.SCANNING
        self.armed_direction = None      # "long" or "short"
        self.pullback_count = 0          # counter-trend candle count
        self.pullback_high = 0.0         # highest high during pullback
        self.pullback_low = float("inf") # lowest low during pullback
        self.window_upper = 0.0          # breakout channel upper boundary
        self.window_lower = 0.0          # breakout channel lower boundary
        self.window_bars_left = 0        # bars remaining in window
        self.last_candle_time = None     # prevent processing same candle twice
        self.last_indicators = {}        # for health file reporting

        # FTMO tracking (shared with bridge via risk_mgr)
        self.peak_balance = FTMO_INITIAL_BALANCE

        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.position = data.get("position")
                self.peak_balance = data.get("peak_balance", FTMO_INITIAL_BALANCE)
                self.phase = Phase(data.get("phase", "SCANNING"))
                self.armed_direction = data.get("armed_direction")
                self.pullback_count = data.get("pullback_count", 0)
                self.pullback_high = data.get("pullback_high", 0.0)
                self.pullback_low = data.get("pullback_low", float("inf"))
                self.window_upper = data.get("window_upper", 0.0)
                self.window_lower = data.get("window_lower", 0.0)
                self.window_bars_left = data.get("window_bars_left", 0)
                self.last_candle_time = data.get("last_candle_time")
                logger.info(f"Loaded state: phase={self.phase.value}, position={self.position}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")

    def _save_state(self):
        try:
            data = {
                "position": self.position,
                "peak_balance": self.peak_balance,
                "phase": self.phase.value,
                "armed_direction": self.armed_direction,
                "pullback_count": self.pullback_count,
                "pullback_high": self.pullback_high,
                "pullback_low": self.pullback_low,
                "window_upper": self.window_upper,
                "window_lower": self.window_lower,
                "window_bars_left": self.window_bars_left,
                "last_candle_time": self.last_candle_time,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _reset_to_scanning(self, reason=""):
        """Reset state machine to SCANNING phase."""
        if reason:
            logger.info(f"RESET to SCANNING: {reason}")
        self.phase = Phase.SCANNING
        self.armed_direction = None
        self.pullback_count = 0
        self.pullback_high = 0.0
        self.pullback_low = float("inf")
        self.window_upper = 0.0
        self.window_lower = 0.0
        self.window_bars_left = 0

    async def connect(self):
        """Connect to FTMO MT5 via MetaApi. Reuses MetaApi instance."""
        if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID:
            logger.error("MetaApi credentials not set!")
            return False
        try:
            from metaapi_cloud_sdk import MetaApi
            if not self.metaapi:
                self.metaapi = MetaApi(METAAPI_TOKEN)
            self.mt_account = await self.metaapi.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)
            if self.mt_account.state != "DEPLOYED":
                await self.mt_account.deploy()
            await self.mt_account.wait_connected(timeout_in_seconds=180)
            # Close previous connection if exists
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
            self.peak_balance = max(self.peak_balance, info["balance"])
            risk_mgr.update_balance_tracking(info["equity"], info["balance"])
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def _check_connection_health(self) -> bool:
        """Check if MetaApi connection is alive."""
        try:
            info = await asyncio.wait_for(
                self.mt_connection.get_account_information(), timeout=15
            )
            return info is not None and "balance" in info
        except Exception:
            return False

    async def fetch_candles(self, limit=500):
        """Fetch historical candles for XAUUSD."""
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
                            logger.info(f"Candles: {len(candles)}, last={candles[-1].get('time','')}, close={candles[-1].get('close',0)}")
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

    # ═══════════════════════════════════════════════════════════════
    # 4-PHASE STATE MACHINE
    # ═══════════════════════════════════════════════════════════════

    def process_candles(self, candles):
        """
        Run the 4-phase state machine on candle data.
        Returns an action dict or None.
        """
        if len(candles) < EMA_MACRO + 10:
            return None

        closes = np.array([c["close"] for c in candles], dtype=float)
        highs = np.array([c["high"] for c in candles], dtype=float)
        lows = np.array([c["low"] for c in candles], dtype=float)
        opens = np.array([c["open"] for c in candles], dtype=float)

        # Calculate indicators
        ema_confirm = calc_ema(closes, EMA_CONFIRM)
        ema_fast = calc_ema(closes, EMA_FAST)
        ema_slow = calc_ema(closes, EMA_SLOW)
        ema_macro = calc_ema(closes, EMA_MACRO)
        atr_vals = calc_atr(highs, lows, closes, ATR_PERIOD)
        adx_vals = calc_adx(highs, lows, closes, ADX_PERIOD)
        rsi_vals = calc_rsi(closes)

        # Latest candle
        c = candles[-1]
        candle_time = c.get("time", "")

        # Skip if we already processed this candle
        if candle_time == self.last_candle_time:
            return None
        self.last_candle_time = candle_time

        i = -1
        cur_close = closes[i]
        cur_high = highs[i]
        cur_low = lows[i]
        cur_open = opens[i]
        cur_atr = atr_vals[i]
        cur_adx = adx_vals[i]
        cur_rsi = rsi_vals[i]
        cur_ema_confirm = ema_confirm[i]
        cur_ema_fast = ema_fast[i]
        cur_ema_slow = ema_slow[i]
        cur_ema_macro = ema_macro[i]
        prev_ema_confirm = ema_confirm[-2]
        prev_ema_fast = ema_fast[-2]

        is_bullish_candle = cur_close > cur_open
        is_bearish_candle = cur_close < cur_open

        # Detect EMA crossover signals
        cross_up = (prev_ema_confirm <= prev_ema_fast) and (cur_ema_confirm > cur_ema_fast)
        cross_down = (prev_ema_confirm >= prev_ema_fast) and (cur_ema_confirm < cur_ema_fast)
        macro_bullish = cur_close > cur_ema_macro
        macro_bearish = cur_close < cur_ema_macro
        adx_ok = cur_adx > ADX_MIN

        logger.info(
            f"[{self.phase.value}] close={cur_close:.2f}, EMA{EMA_CONFIRM}={cur_ema_confirm:.2f}, "
            f"EMA{EMA_FAST}={cur_ema_fast:.2f}, EMA{EMA_SLOW}={cur_ema_slow:.2f}, "
            f"EMA{EMA_MACRO}={cur_ema_macro:.2f}, RSI={cur_rsi:.1f}, ADX={cur_adx:.1f}, "
            f"ATR={cur_atr:.2f}, candle={'↑' if is_bullish_candle else '↓'}"
        )

        # Store for health file
        self.last_indicators = {
            "price": round(cur_close, 2),
            "ema_3": round(cur_ema_confirm, 2),
            "ema_14": round(cur_ema_fast, 2),
            "ema_24": round(cur_ema_slow, 2),
            "ema_100": round(cur_ema_macro, 2),
            "rsi": round(cur_rsi, 1),
            "adx": round(cur_adx, 1),
            "atr": round(cur_atr, 2),
            "cross_up": bool(cross_up),
            "cross_down": bool(cross_down),
            "macro_bullish": bool(macro_bullish),
            "adx_ok": bool(adx_ok),
        }

        # ── Handle exit signals for open positions ──
        if self.position:
            pos_dir = self.position.get("direction")
            # Exit long: EMA confirm crosses below slow, or RSI > 80
            if pos_dir == "long":
                if cur_ema_confirm < cur_ema_slow or cur_rsi > 80:
                    reason = "ema_cross_down" if cur_ema_confirm < cur_ema_slow else "rsi_overbought"
                    return {"action": "close", "reason": reason}
            # Exit short: EMA confirm crosses above slow, or RSI < 20
            if pos_dir == "short":
                if cur_ema_confirm > cur_ema_slow or cur_rsi < 20:
                    reason = "ema_cross_up" if cur_ema_confirm > cur_ema_slow else "rsi_oversold"
                    return {"action": "close", "reason": reason}
            return None  # hold position, no new entries while in trade

        # ════════════════════════════════════════════
        # PHASE 1: SCANNING — look for EMA crossover
        # ════════════════════════════════════════════
        if self.phase == Phase.SCANNING:
            # Long setup: confirm EMA crosses above fast + macro bullish + ADX
            if cross_up and macro_bullish and adx_ok:
                self.phase = Phase.ARMED_LONG
                self.armed_direction = "long"
                self.pullback_count = 0
                self.pullback_high = cur_high
                self.pullback_low = cur_low
                logger.info(f"→ ARMED_LONG: EMA crossover detected, waiting for pullback")
                self._save_state()
                return None

            # Short setup: confirm EMA crosses below fast + macro bearish + ADX
            if cross_down and macro_bearish and adx_ok:
                self.phase = Phase.ARMED_SHORT
                self.armed_direction = "short"
                self.pullback_count = 0
                self.pullback_high = cur_high
                self.pullback_low = cur_low
                logger.info(f"→ ARMED_SHORT: EMA crossover detected, waiting for pullback")
                self._save_state()
                return None

            return None

        # ════════════════════════════════════════════
        # PHASE 2: ARMED — count pullback candles
        # ════════════════════════════════════════════
        if self.phase in (Phase.ARMED_LONG, Phase.ARMED_SHORT):
            # Global invalidation: opposite signal cancels everything
            if self.phase == Phase.ARMED_LONG and cross_down:
                self._reset_to_scanning("invalidated by bearish crossover")
                self._save_state()
                return None
            if self.phase == Phase.ARMED_SHORT and cross_up:
                self._reset_to_scanning("invalidated by bullish crossover")
                self._save_state()
                return None

            # Track pullback range
            self.pullback_high = max(self.pullback_high, cur_high)
            self.pullback_low = min(self.pullback_low, cur_low)

            # Count counter-trend candles
            required = PULLBACK_COUNT_LONG if self.phase == Phase.ARMED_LONG else PULLBACK_COUNT_SHORT

            if self.phase == Phase.ARMED_LONG and is_bearish_candle:
                self.pullback_count += 1
                logger.info(f"  Pullback candle {self.pullback_count}/{required} (bearish)")
            elif self.phase == Phase.ARMED_SHORT and is_bullish_candle:
                self.pullback_count += 1
                logger.info(f"  Pullback candle {self.pullback_count}/{required} (bullish)")

            # Check if pullback is confirmed
            if self.pullback_count >= required:
                # Build breakout window channel
                pullback_range = self.pullback_high - self.pullback_low
                expansion = pullback_range * WINDOW_PRICE_MULT

                if self.phase == Phase.ARMED_LONG:
                    self.window_upper = self.pullback_high + expansion
                    self.window_lower = self.pullback_low - expansion
                    self.phase = Phase.WINDOW_LONG
                else:
                    self.window_upper = self.pullback_high + expansion
                    self.window_lower = self.pullback_low - expansion
                    self.phase = Phase.WINDOW_SHORT

                self.window_bars_left = WINDOW_BARS
                logger.info(
                    f"→ {self.phase.value}: channel [{self.window_lower:.2f} — {self.window_upper:.2f}], "
                    f"range={pullback_range:.2f}, expansion={expansion:.2f}, "
                    f"window={self.window_bars_left} bars"
                )
                self._save_state()

            return None

        # ════════════════════════════════════════════
        # PHASE 3: WINDOW — wait for breakout
        # ════════════════════════════════════════════
        if self.phase in (Phase.WINDOW_LONG, Phase.WINDOW_SHORT):
            self.window_bars_left -= 1

            # Global invalidation
            if self.phase == Phase.WINDOW_LONG and cross_down:
                self._reset_to_scanning("window invalidated by bearish crossover")
                self._save_state()
                return None
            if self.phase == Phase.WINDOW_SHORT and cross_up:
                self._reset_to_scanning("window invalidated by bullish crossover")
                self._save_state()
                return None

            # Check for breakout
            if self.phase == Phase.WINDOW_LONG:
                if cur_high > self.window_upper:
                    # BREAKOUT UP → ENTRY LONG
                    entry_price = self.window_upper  # approximate fill at channel break
                    sl = entry_price - (cur_atr * ATR_SL_MULT)
                    tp = entry_price + (cur_atr * ATR_TP_LONG)
                    logger.info(
                        f"★ BREAKOUT LONG! high={cur_high:.2f} > upper={self.window_upper:.2f}"
                    )
                    self._reset_to_scanning()
                    self._save_state()
                    return {
                        "action": "open", "direction": "long",
                        "entry": entry_price, "stoploss": sl, "takeprofit": tp,
                        "atr": cur_atr,
                    }

                # Failure boundary: price breaks lower channel
                if cur_low < self.window_lower:
                    # Failed breakout — reset to ARMED (give another chance)
                    self.phase = Phase.ARMED_LONG
                    self.pullback_count = 0
                    self.pullback_high = cur_high
                    self.pullback_low = cur_low
                    logger.info(f"  Window failed (broke lower). Back to ARMED_LONG.")
                    self._save_state()
                    return None

            if self.phase == Phase.WINDOW_SHORT:
                if cur_low < self.window_lower:
                    # BREAKOUT DOWN → ENTRY SHORT
                    entry_price = self.window_lower
                    sl = entry_price + (cur_atr * ATR_SL_MULT)
                    tp = entry_price - (cur_atr * ATR_TP_SHORT)
                    logger.info(
                        f"★ BREAKOUT SHORT! low={cur_low:.2f} < lower={self.window_lower:.2f}"
                    )
                    self._reset_to_scanning()
                    self._save_state()
                    return {
                        "action": "open", "direction": "short",
                        "entry": entry_price, "stoploss": sl, "takeprofit": tp,
                        "atr": cur_atr,
                    }

                # Failure boundary: price breaks upper channel
                if cur_high > self.window_upper:
                    self.phase = Phase.ARMED_SHORT
                    self.pullback_count = 0
                    self.pullback_high = cur_high
                    self.pullback_low = cur_low
                    logger.info(f"  Window failed (broke upper). Back to ARMED_SHORT.")
                    self._save_state()
                    return None

            # Window expired
            if self.window_bars_left <= 0:
                self._reset_to_scanning("window expired without breakout")
                self._save_state()
                return None

            logger.info(
                f"  Window: [{self.window_lower:.2f} — {self.window_upper:.2f}], "
                f"bars_left={self.window_bars_left}"
            )
            self._save_state()
            return None

        return None

    # ═══════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def calculate_volume(self, entry_price, stoploss, account_balance):
        """XAUUSD lot sizing. 1 lot = 100 oz."""
        # Apply agent supervisor multiplier
        adjusted_risk, agent_mult, agent_reason = risk_mgr.get_agent_adjusted_risk("Gold")
        if agent_mult < 1.0:
            logger.info(f"Agent risk adj [Gold]: {RISK_PER_TRADE:.3%} × {agent_mult:.2f} = {adjusted_risk:.3%} ({agent_reason})")
        risk_amount = account_balance * adjusted_risk
        sl_distance = abs(entry_price - stoploss)
        contract_size = 100
        if sl_distance == 0:
            return 0.01
        volume = risk_amount / (contract_size * sl_distance)
        volume = round(volume, 2)
        return max(0.01, min(volume, 5.0))

    async def check_ftmo_limits(self):
        """Check FTMO limits via shared risk manager (coordinates with bridge)."""
        try:
            info = await self.mt_connection.get_account_information()
            equity = info.get("equity", FTMO_INITIAL_BALANCE)
            balance = info.get("balance", FTMO_INITIAL_BALANCE)

            # Use shared risk manager (same state as bridge)
            within_limits, reason = risk_mgr.check_ftmo_limits(equity, balance)
            if not within_limits:
                logger.warning(f"FTMO LIMIT: {reason}")
                return False

            # Also check combined risk cap before opening
            can_open, reason = risk_mgr.can_open_position("Gold", RISK_PER_TRADE)
            if not can_open:
                logger.warning(f"Risk manager blocked Gold: {reason}")
                return False

            return True
        except Exception as e:
            logger.error(f"FTMO limit check error: {e}")
            return True

    async def get_gold_positions(self):
        try:
            positions = await self.mt_connection.get_positions()
            return [p for p in positions if p.get("symbol") == SYMBOL]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def open_position(self, signal):
        """Open a gold position on FTMO."""
        info = await self.mt_connection.get_account_information()
        balance = info["balance"]
        volume = self.calculate_volume(signal["entry"], signal["stoploss"], balance)
        sl = round(signal["stoploss"], 2)
        tp = round(signal["takeprofit"], 2)

        logger.info(
            f"OPENING {signal['direction'].upper()} {volume} {SYMBOL} @ ~{signal['entry']:.2f}, "
            f"SL={sl}, TP={tp}, ATR={signal['atr']:.2f}, "
            f"R:R=1:{abs(signal['takeprofit']-signal['entry'])/abs(signal['entry']-signal['stoploss']):.1f}"
        )

        try:
            if signal["direction"] == "long":
                result = await self.mt_connection.create_market_buy_order(
                    SYMBOL, volume, stop_loss=sl, take_profit=tp
                )
            else:
                result = await self.mt_connection.create_market_sell_order(
                    SYMBOL, volume, stop_loss=sl, take_profit=tp
                )

            pos_id = result.get("positionId", result.get("orderId", "unknown"))
            logger.info(f"Position opened: {pos_id}, result: {result.get('stringCode')}")

            self.position = {
                "id": str(pos_id),
                "direction": signal["direction"],
                "entry": signal["entry"],
                "stoploss": sl,
                "takeprofit": tp,
                "volume": volume,
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_state()

            # Register with shared risk manager
            risk_mgr.register_position(
                "Gold", str(pos_id), SYMBOL,
                signal["direction"], volume, RISK_PER_TRADE
            )
            return True
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return False

    async def close_position(self, reason="signal"):
        if not self.position:
            return
        pos_id = self.position['id']
        logger.info(f"CLOSING {SYMBOL} position {pos_id}, reason: {reason}")
        try:
            positions = await self.get_gold_positions()
            for p in positions:
                await self.mt_connection.close_position(p["id"])
                logger.info(f"Closed MT5 position {p['id']}")
            # Unregister from shared risk manager
            risk_mgr.unregister_position("Gold", pos_id)
            self.position = None
            self._save_state()
        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    # ═══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════

    async def sync(self):
        """Main sync cycle."""
        hour = datetime.now(timezone.utc).hour
        if hour in SKIP_HOURS_UTC:
            return

        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return

        # Sync position state with MT5
        mt5_positions = await self.get_gold_positions()
        if not mt5_positions and self.position:
            logger.info(f"Position closed by SL/TP: {self.position['direction']}")
            risk_mgr.unregister_position("Gold", self.position["id"])
            self.position = None
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
            logger.info(f"Synced existing position: {self.position}")

        # Fetch candles and run state machine
        candles = await self.fetch_candles(limit=500)
        if not candles or len(candles) < EMA_MACRO + 10:
            logger.warning(f"Insufficient candle data: {len(candles) if candles else 0}")
            return

        action = self.process_candles(candles)
        if not action:
            return

        if action.get("action") == "close":
            await self.close_position(reason=action.get("reason", "signal"))
            return

        if action.get("action") == "open" and not self.position:
            within_limits = await self.check_ftmo_limits()
            if not within_limits:
                logger.warning("FTMO limits breached — skipping gold trade")
                return
            if len(mt5_positions) >= MAX_GOLD_POSITIONS:
                return
            await self.open_position(action)

    async def run(self):
        """Main loop."""
        logger.info("=" * 60)
        logger.info("FTMO Gold Strategy v2 — 4-Phase Pullback Breakout")
        logger.info(f"Symbol: {SYMBOL}, Timeframe: {TIMEFRAME}")
        logger.info(f"Risk: {RISK_PER_TRADE:.0%}, SL: {ATR_SL_MULT}x ATR, TP: {ATR_TP_LONG}x/{ATR_TP_SHORT}x ATR")
        logger.info(f"EMA: confirm={EMA_CONFIRM}, fast={EMA_FAST}, slow={EMA_SLOW}, macro={EMA_MACRO}")
        logger.info(f"Pullback: {PULLBACK_COUNT_LONG} candles, Window: {WINDOW_BARS} bars, Mult: {WINDOW_PRICE_MULT}")
        logger.info("=" * 60)

        connected = await self.connect()
        if not connected:
            logger.error("Failed to connect. Exiting.")
            return

        logger.info(f"Starting gold strategy loop (poll every {POLL_INTERVAL}s)...")
        consecutive_errors = 0
        last_healthy_sync = datetime.now(timezone.utc)

        while True:
            try:
                # Check connection health every 10 minutes
                if (datetime.now(timezone.utc) - last_healthy_sync).total_seconds() > 600:
                    healthy = await self._check_connection_health()
                    if not healthy:
                        logger.warning("Connection stale for >10min, reconnecting...")
                        await self.connect()
                    last_healthy_sync = datetime.now(timezone.utc)

                await self.sync()
                consecutive_errors = 0
                last_healthy_sync = datetime.now(timezone.utc)
                self._write_health("ok")
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Sync error ({consecutive_errors}): {e}")
                self._write_health("error", str(e))
                if consecutive_errors >= 3:
                    logger.warning("Too many errors, reconnecting MetaApi...")
                    try:
                        await self.connect()
                        consecutive_errors = 0
                        logger.info("Reconnected successfully")
                    except Exception as re:
                        logger.error(f"Reconnect failed: {re}")
            await asyncio.sleep(POLL_INTERVAL)

    def _write_health(self, status: str, detail: str = ""):
        """Write health status for cron monitoring."""
        try:
            ind = self.last_indicators
            # Build signal explanation
            reasons = []
            if ind:
                # EMA cross status
                if ind.get("cross_up"):
                    reasons.append("EMA 3/14: bullish cross ✓")
                elif ind.get("cross_down"):
                    reasons.append("EMA 3/14: bearish cross ✓")
                else:
                    ema3 = ind.get("ema_3", 0)
                    ema14 = ind.get("ema_14", 0)
                    if ema3 > ema14:
                        reasons.append(f"EMA 3>14: bullish ({ema3:.0f}>{ema14:.0f})")
                    else:
                        reasons.append(f"EMA 3<14: bearish ({ema3:.0f}<{ema14:.0f})")

                # Macro trend
                if ind.get("macro_bullish"):
                    reasons.append(f"Price > EMA100: bullish ✓")
                else:
                    reasons.append(f"Price < EMA100: bearish")

                # ADX
                adx = ind.get("adx", 0)
                if ind.get("adx_ok"):
                    reasons.append(f"ADX: {adx:.0f} (>{ADX_MIN} ✓)")
                else:
                    reasons.append(f"ADX: {adx:.0f} (need >{ADX_MIN})")

                # RSI
                rsi = ind.get("rsi", 0)
                reasons.append(f"RSI: {rsi:.0f}")

                # Phase-specific context
                if self.phase in (Phase.ARMED_LONG, Phase.ARMED_SHORT):
                    direction = "long" if self.phase == Phase.ARMED_LONG else "short"
                    required = PULLBACK_COUNT_LONG if direction == "long" else PULLBACK_COUNT_SHORT
                    reasons.append(f"Pullback: {self.pullback_count}/{required} candles")
                elif self.phase in (Phase.WINDOW_LONG, Phase.WINDOW_SHORT):
                    reasons.append(f"Window: {self.window_bars_left} bars left [{self.window_lower:.0f}-{self.window_upper:.0f}]")

            data = {
                "status": status,
                "detail": detail,
                "phase": self.phase.value,
                "position": self.position is not None,
                "last_candle": self.last_candle_time,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "indicators": ind,
                "signal_explanation": " | ".join(reasons) if reasons else ("market closed — waiting for fresh candles" if status == "ok" else "starting up"),
            }
            Path("ftmo_gold_health.json").write_text(json.dumps(data, indent=2))
        except Exception:
            pass


async def main():
    strategy = GoldStrategy()
    await strategy.run()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   FTMO Gold Strategy v2.0                     ║
    ║   4-Phase Pullback Breakout — XAUUSD          ║
    ║   MetaApi → FTMO MT5                          ║
    ╚═══════════════════════════════════════════════╝
    """)
    asyncio.run(main())
