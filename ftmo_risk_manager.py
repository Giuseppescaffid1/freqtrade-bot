"""
FTMO Shared Risk Manager — Coordinates risk across all FTMO trading systems

All systems (ftmo_bridge, ftmo_gold) share one FTMO account (CHF 80K).
This module ensures combined risk never breaches FTMO limits:
    - 5% max daily loss (we use 4% as safety margin)
    - 10% max total drawdown (we use 9% as safety margin)
    - Combined position risk capped at 3.5% of account

Architecture:
    ftmo_bridge.py (crypto) ──┐
                               ├── ftmo_risk_manager.py ──► ftmo_shared_risk.json
    ftmo_gold.py   (XAUUSD) ──┘

File locking prevents race conditions between concurrent processes.
"""

import json
import fcntl
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger("ftmo_risk_manager")

# GARCH volatility filter — dynamic risk sizing based on vol regime
try:
    from garch_vol_filter import GarchVolFilter
    garch_filter = GarchVolFilter()
    logger.info("GARCH volatility filter loaded")
except ImportError:
    garch_filter = None
    logger.info("GARCH filter not available (arch package not installed)")

# ── FTMO Account Config ──
FTMO_INITIAL_BALANCE = 80_000.0    # CHF 80K challenge
MAX_DAILY_LOSS_PCT = 0.04          # 4% hard stop (FTMO = 5%, 1% buffer)
MAX_TOTAL_DD_PCT = 0.09            # 9% hard stop (FTMO = 10%, 1% buffer)
MAX_COMBINED_RISK_PCT = 0.03       # 3% max combined open risk (2% buffer to daily limit)
# Research: "Never let total open risk exceed 3% at any time"

# Per-source risk profiles
# Informed by FTMO research: 0.5% risk = 67% pass rate, 1% = 35%, 3% = 12%
# Keep risk conservative to maximize pass probability.
# risk_per_trade: fraction of account risked per position
# max_positions: concurrent position cap for this source
SOURCE_PROFILES = {
    "FTMO_SmartMoney": {"risk_per_trade": 0.01, "max_positions": 2, "default_sl_pct": 0.10},
    "FTMO_CryptoEdge": {"risk_per_trade": 0.01, "max_positions": 2, "default_sl_pct": 0.015},
    "QuantSOL":        {"risk_per_trade": 0.005, "max_positions": 1, "default_sl_pct": 0.10},
    "FTMO_Blitz":      {"risk_per_trade": 0.01, "max_positions": 1, "default_sl_pct": 0.10},
    "Gold":            {"risk_per_trade": 0.005, "max_positions": 1, "default_sl_pct": 0.05},
    "Oil":             {"risk_per_trade": 0.005, "max_positions": 1, "default_sl_pct": 0.03},
}

# Correlated instruments: FTMO flags opposing positions on correlated assets.
# BTC and ETH are ~85% correlated. Treat as one bucket for risk limits.
CORRELATED_BUCKETS = {
    "crypto": {"SOLUSD", "BTCUSD", "ETHUSD", "LTCUSD"},
    "commodities": {"XAUUSD", "USOIL.cash"},
}
MAX_CORRELATED_POSITIONS = 3  # max positions within same correlation bucket

STATE_FILE = Path("/home/giuseppe/freqtrade-bot/ftmo_shared_risk.json")
OVERRIDE_FILE = Path("/home/giuseppe/freqtrade-bot/ftmo_risk_overrides.json")


def _read_state() -> dict:
    """Read shared state with file locking."""
    default = {
        "positions": {},
        "daily_start_balance": FTMO_INITIAL_BALANCE,
        "peak_balance": FTMO_INITIAL_BALANCE,
        "last_day_reset": "",
        "updated_at": "",
    }
    if not STATE_FILE.exists():
        return default
    try:
        with open(STATE_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        return data
    except Exception as e:
        logger.error(f"Error reading shared risk state: {e}")
        return default


def _write_state(data: dict):
    """Write shared state with exclusive file locking."""
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(STATE_FILE, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Error writing shared risk state: {e}")


def _read_agent_overrides() -> dict | None:
    """Read agent risk overrides, return None if missing/expired/invalid."""
    if not OVERRIDE_FILE.exists():
        return None
    try:
        with open(OVERRIDE_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        expires = data.get("expires_at", "")
        if expires:
            exp_dt = datetime.fromisoformat(expires)
            if datetime.now(timezone.utc) > exp_dt:
                logger.debug("Agent overrides expired — ignoring")
                return None
        return data
    except Exception as e:
        logger.debug(f"Failed to read agent overrides: {e}")
        return None


def get_profile(source: str) -> dict:
    """Get risk profile for a source."""
    return SOURCE_PROFILES.get(source, {"risk_per_trade": 0.01, "max_positions": 1, "default_sl_pct": 0.10})


def get_agent_adjusted_risk(source: str) -> tuple[float, float, str]:
    """
    Get agent-adjusted risk for a source.
    Returns (adjusted_risk_pct, multiplier, reason).
    If no overrides or expired, returns base risk unchanged.
    """
    profile = get_profile(source)
    base_risk = profile["risk_per_trade"]

    overrides = _read_agent_overrides()
    if not overrides:
        return base_risk, 1.0, "no_agent_overrides"

    strat_override = overrides.get("strategy_overrides", {}).get(source)
    if not strat_override:
        return base_risk, 1.0, "no_override_for_source"

    multiplier = strat_override.get("risk_multiplier", 1.0)
    multiplier = max(0.0, min(1.0, multiplier))  # safety clamp
    adjusted_risk = base_risk * multiplier
    reason = strat_override.get("reason", "agent_override")

    if multiplier < 1.0:
        logger.info(
            f"Agent risk adjustment [{source}]: "
            f"{base_risk:.3%} × {multiplier:.2f} = {adjusted_risk:.3%} ({reason})"
        )

    return adjusted_risk, multiplier, reason


def get_garch_adjusted_risk(source: str, symbol: str, closes=None) -> tuple[float, float, str]:
    """
    Get GARCH-adjusted risk for a trade.

    Returns (adjusted_risk_pct, multiplier, regime).
    If GARCH filter is unavailable or data insufficient, returns base risk.
    """
    profile = get_profile(source)
    base_risk = profile["risk_per_trade"]

    if garch_filter is None or closes is None:
        return base_risk, 1.0, "NO_FILTER"

    multiplier = garch_filter.get_risk_multiplier(symbol, closes)
    adjusted_risk = base_risk * multiplier
    regime = garch_filter.get_regime(symbol).get("regime", "UNKNOWN")

    if multiplier < 1.0:
        logger.info(
            f"GARCH risk adjustment [{source}/{symbol}]: "
            f"{base_risk:.3%} × {multiplier:.2f} = {adjusted_risk:.3%} ({regime})"
        )

    return adjusted_risk, multiplier, regime


def get_open_positions() -> dict:
    """Get all currently tracked positions across all sources."""
    state = _read_state()
    return state.get("positions", {})


def get_combined_risk() -> float:
    """Calculate total open risk across all sources."""
    positions = get_open_positions()
    total_risk = 0.0
    for pos_id, pos in positions.items():
        total_risk += pos.get("risk_pct", 0.0)
    return total_risk


def count_positions(source: str = None) -> int:
    """Count open positions, optionally filtered by source."""
    positions = get_open_positions()
    if source:
        return sum(1 for p in positions.values() if p.get("source") == source)
    return len(positions)


def can_open_position(source: str, risk_pct: float = None) -> tuple[bool, str]:
    """
    Check if a new position can be opened within FTMO limits.

    Returns (allowed: bool, reason: str).
    """
    # Check agent overrides first
    agent = _read_agent_overrides()
    if agent:
        if agent.get("global_halt"):
            return False, "Agent: global halt active"
        strat_ov = agent.get("strategy_overrides", {}).get(source)
        if strat_ov and not strat_ov.get("enabled", True):
            return False, f"Agent: {source} disabled — {strat_ov.get('reason', 'agent decision')}"

    profile = get_profile(source)
    risk = risk_pct or profile["risk_per_trade"]

    # Check per-source position limit (use agent override if stricter)
    max_pos = profile["max_positions"]
    if agent:
        strat_ov = agent.get("strategy_overrides", {}).get(source)
        if strat_ov:
            agent_max = strat_ov.get("max_positions", max_pos)
            max_pos = min(max_pos, agent_max)

    source_count = count_positions(source)
    if source_count >= max_pos:
        return False, f"{source} at max positions ({max_pos})"

    # Check combined risk cap
    current_risk = get_combined_risk()
    if current_risk + risk > MAX_COMBINED_RISK_PCT:
        return False, (f"Combined risk would be {(current_risk + risk):.2%} "
                       f"(limit: {MAX_COMBINED_RISK_PCT:.2%})")

    # Check total position count (hard limit: 5 across everything)
    total = count_positions()
    if total >= 5:
        return False, f"Total position cap reached ({total}/5)"

    # Check correlated bucket limit
    # FTMO flags opposing positions on correlated instruments (BTC/ETH ~85% correlated)
    positions = get_open_positions()
    for bucket_name, bucket_symbols in CORRELATED_BUCKETS.items():
        bucket_count = sum(1 for p in positions.values() if p.get("symbol") in bucket_symbols)
        if bucket_count >= MAX_CORRELATED_POSITIONS:
            return False, f"Correlated bucket '{bucket_name}' at max ({bucket_count}/{MAX_CORRELATED_POSITIONS})"

    return True, "OK"


def register_position(source: str, position_id: str, symbol: str,
                      direction: str, volume: float, risk_pct: float = None):
    """Register a new position in shared state."""
    profile = get_profile(source)
    risk = risk_pct or profile["risk_per_trade"]

    state = _read_state()
    key = f"{source}:{position_id}"
    state["positions"][key] = {
        "source": source,
        "symbol": symbol,
        "direction": direction,
        "volume": volume,
        "risk_pct": risk,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(state)
    logger.info(f"Registered position {key}: {direction} {volume} {symbol} (risk={risk:.2%})")


def unregister_position(source: str, position_id: str):
    """Remove a closed position from shared state."""
    state = _read_state()
    key = f"{source}:{position_id}"
    if key in state.get("positions", {}):
        pos = state["positions"].pop(key)
        _write_state(state)
        logger.info(f"Unregistered position {key}: {pos.get('symbol')}")
    else:
        logger.debug(f"Position {key} not found in shared state (already removed?)")


def update_balance_tracking(equity: float, balance: float):
    """
    Update daily P&L and peak balance tracking from actual MetaApi data.
    Called by whichever system fetches account info.
    """
    state = _read_state()

    # Reset daily tracking at midnight CEST
    now = datetime.now(timezone.utc)
    cest_now = now + timedelta(hours=2)
    today = str(cest_now.date())

    if state.get("last_day_reset") != today:
        state["daily_start_balance"] = balance
        state["last_day_reset"] = today
        logger.info(f"FTMO new day. Start balance: {balance:,.2f}")

    # Update peak
    state["peak_balance"] = max(state.get("peak_balance", FTMO_INITIAL_BALANCE), balance)

    _write_state(state)


def check_ftmo_limits(equity: float, balance: float) -> tuple[bool, str]:
    """
    Check if we're within FTMO daily loss and total drawdown limits.
    Uses ACTUAL MetaApi equity (includes all open P&L across all systems).

    Returns (within_limits: bool, reason: str).
    """
    # Update tracking first
    update_balance_tracking(equity, balance)
    state = _read_state()

    # Daily loss check (equity vs start-of-day balance)
    daily_start = state.get("daily_start_balance", FTMO_INITIAL_BALANCE)
    if daily_start > 0:
        daily_pnl = (equity - daily_start) / daily_start
        if daily_pnl <= -MAX_DAILY_LOSS_PCT:
            return False, f"DAILY LOSS BREAKER: {daily_pnl:.2%} (limit: {-MAX_DAILY_LOSS_PCT:.2%})"

    # Total drawdown check (equity vs initial balance)
    total_dd = (equity - FTMO_INITIAL_BALANCE) / FTMO_INITIAL_BALANCE
    if total_dd <= -MAX_TOTAL_DD_PCT:
        return False, f"TOTAL DD BREAKER: {total_dd:.2%} (limit: {-MAX_TOTAL_DD_PCT:.2%})"

    return True, f"OK (daily: {daily_pnl:.2%}, total: {total_dd:.2%})"


def get_status() -> dict:
    """Get full status for logging/monitoring."""
    state = _read_state()
    positions = state.get("positions", {})

    by_source = {}
    for pos in positions.values():
        src = pos.get("source", "unknown")
        by_source.setdefault(src, []).append(pos)

    result = {
        "total_positions": len(positions),
        "combined_risk": get_combined_risk(),
        "positions_by_source": {k: len(v) for k, v in by_source.items()},
        "peak_balance": state.get("peak_balance", FTMO_INITIAL_BALANCE),
        "daily_start_balance": state.get("daily_start_balance", FTMO_INITIAL_BALANCE),
        "last_day_reset": state.get("last_day_reset", ""),
    }

    # Add GARCH vol regime info if available
    if garch_filter:
        result["garch_regimes"] = garch_filter.get_all_regimes()

    return result
