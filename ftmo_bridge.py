"""
FTMO MetaApi Bridge v2 — Connects ALL Freqtrade bots to FTMO MT4/MT5

Architecture:
    [Freqtrade Bots (3x)] → polls via REST API → detects new trades
    → [This Bridge] → sends orders via MetaApi SDK → [FTMO MT4/MT5 Account]
    → [Shared Risk Manager] → coordinates with Gold strategy

Bots monitored:
    1. FTMO_SmartMoney (port 8085) — SOL/LTC/ETH multi-pair
    2. QuantSOL (port 8083) — SOL scalper
    3. FTMO_Blitz (port 8086) — ETH aggressive

Setup:
    1. Sign up at https://metaapi.cloud (free tier: 1 account)
    2. Set env vars: METAAPI_TOKEN, METAAPI_ACCOUNT_ID
    3. Run: python ftmo_bridge.py

FTMO-specific:
    - Shared risk manager coordinates with Gold strategy
    - Per-bot risk profiles (different sizing per strategy)
    - Combined risk cap (3.5% max across all positions)
    - Session filter: no trades 00:00-03:00 UTC
    - Weekend skip (no trading Sat/Sun)
    - Auto-reconnect after consecutive errors
    - Stale position cleanup (detects MT SL/TP closures)
"""
import os
import sys
import json
import time
import asyncio
import logging
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ftmo_risk_manager as risk_mgr
from garch_vol_filter import GarchVolFilter

# GARCH vol filter instance (shared across bridge lifetime)
garch_filter = GarchVolFilter()

# ── Configuration ──
FREQTRADE_BOTS = [
    # SmartMoney disabled — SOL/LTC losing badly (-88%). Replaced by CryptoEdge.
    # {
    #     "name": "FTMO_SmartMoney",
    #     "url": "http://127.0.0.1:8085",
    #     "user": "ftmosmc",
    #     "password": "ftmosmc2026",
    # },
    {
        "name": "FTMO_CryptoEdge",
        "url": os.getenv("FREQTRADE_URL", "http://127.0.0.1:8087"),
        "user": os.getenv("FREQTRADE_USER", "ftmoedge"),
        "password": os.getenv("FREQTRADE_PASS", "ftmoedge2026"),
    },
    # QuantSOL disabled from FTMO bridge — 22% max DD too risky for challenge.
    # Still runs on Bybit via freqtrade-sol service (port 8083).
    {
        "name": "FTMO_Blitz",
        "url": os.getenv("FREQTRADE_URL_3", "http://127.0.0.1:8086"),
        "user": os.getenv("FREQTRADE_USER_3", "ftmoblitz"),
        "password": os.getenv("FREQTRADE_PASS_3", "ftmoblitz2026"),
    },
    {
        "name": "FTMO_SMC_SOL",
        "url": "http://127.0.0.1:8088",
        "user": "ftmosmcsol",
        "password": "ftmosmcsol2026",
    },
]

METAAPI_TOKEN = os.getenv("METAAPI_TOKEN", "")
METAAPI_ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID", "")

# Symbol mapping: Freqtrade pair -> FTMO MT4/MT5 symbol
SYMBOL_MAP = {
    "SOL/USDT:USDT": os.getenv("FTMO_SOL_SYMBOL", "SOLUSD"),
    "LTC/USDT:USDT": os.getenv("FTMO_LTC_SYMBOL", "LTCUSD"),
    "BTC/USDT:USDT": os.getenv("FTMO_BTC_SYMBOL", "BTCUSD"),
    "ETH/USDT:USDT": os.getenv("FTMO_ETH_SYMBOL", "ETHUSD"),
}

POLL_INTERVAL = 10  # seconds between checks
LOG_FILE = "ftmo_bridge.log"

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
logger = logging.getLogger("ftmo_bridge")

# Silence noisy loggers
for noisy in ("engineio", "socketio", "urllib3", "httpcore", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


class FreqtradeClient:
    """Simple REST client for Freqtrade API."""

    def __init__(self, url: str, username: str, password: str):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (username, password)

    def _get(self, endpoint: str) -> dict:
        try:
            resp = self.session.get(f"{self.url}{endpoint}", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Freqtrade API error ({self.url}{endpoint}): {e}")
            return {}

    def get_open_trades(self) -> list:
        return self._get("/api/v1/status") or []

    def get_closed_trades(self, limit: int = 10) -> list:
        data = self._get(f"/api/v1/trades?limit={limit}")
        if isinstance(data, dict):
            return data.get("trades", [])
        return data or []

    def get_profit(self) -> dict:
        return self._get("/api/v1/profit") or {}

    def get_pair_candles(self, pair: str, timeframe: str = "15m", limit: int = 500) -> list[float]:
        """Fetch recent close prices for a pair from Freqtrade's data."""
        data = self._get(f"/api/v1/pair_candles?pair={pair}&timeframe={timeframe}&limit={limit}")
        if isinstance(data, dict) and "data" in data:
            # Freqtrade returns: data = [[timestamp, open, high, low, close, ...], ...]
            closes = [candle[4] for candle in data["data"] if len(candle) > 4]
            return closes
        return []

    def is_alive(self) -> bool:
        try:
            resp = self.session.get(f"{self.url}/api/v1/ping", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


class FTMOBridge:
    """
    Bridges Freqtrade signals to FTMO via MetaApi.
    Monitors multiple Freqtrade bots and mirrors trades to FTMO MT4/MT5.
    Uses shared risk manager to coordinate with Gold strategy.
    """

    def __init__(self):
        self.ft_clients = {}
        for bot in FREQTRADE_BOTS:
            client = FreqtradeClient(bot["url"], bot["user"], bot["password"])
            self.ft_clients[bot["name"]] = client
            logger.info(f"Registered bot: {bot['name']} @ {bot['url']}")

        self.metaapi = None
        self.mt_account = None
        self.mt_connection = None

        # Track mirrored positions: ft_key -> mt_position_id
        self.mirrored_trades = {}
        self.state_file = Path("ftmo_bridge_state.json")
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.mirrored_trades = data.get("mirrored_trades", {})
                logger.info(f"Loaded state: {len(self.mirrored_trades)} mirrored trades")
            except Exception as e:
                logger.error(f"Error loading state: {e}")

    def _save_state(self):
        try:
            data = {
                "mirrored_trades": self.mirrored_trades,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    async def connect_metaapi(self):
        """Connect to FTMO MT4/MT5 account via MetaApi."""
        if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID:
            logger.warning(
                "MetaApi credentials not set. Running in DRY MODE.\n"
                "Set METAAPI_TOKEN and METAAPI_ACCOUNT_ID to enable live trading."
            )
            return False

        try:
            from metaapi_cloud_sdk import MetaApi

            self.metaapi = MetaApi(METAAPI_TOKEN)
            self.mt_account = await self.metaapi.metatrader_account_api.get_account(
                METAAPI_ACCOUNT_ID
            )

            if self.mt_account.state != "DEPLOYED":
                logger.info("Deploying MetaApi account...")
                await self.mt_account.deploy()

            logger.info("Waiting for MetaApi connection...")
            await asyncio.wait_for(
                self.mt_account.wait_connected(timeout_in_seconds=90), timeout=120
            )

            self.mt_connection = self.mt_account.get_rpc_connection()
            await asyncio.wait_for(self.mt_connection.connect(), timeout=30)
            await asyncio.wait_for(
                self.mt_connection.wait_synchronized(), timeout=90
            )

            account_info = await self.mt_connection.get_account_information()
            logger.info(
                f"Connected to FTMO: balance={account_info['balance']}, "
                f"equity={account_info['equity']}, server={account_info.get('server', 'N/A')}"
            )

            # Update shared risk manager with real balance
            risk_mgr.update_balance_tracking(account_info["equity"], account_info["balance"])
            return True

        except Exception as e:
            logger.error(f"MetaApi connection failed: {e}")
            return False

    async def get_mt_account_info(self) -> dict | None:
        if not self.mt_connection:
            return None
        try:
            return await self.mt_connection.get_account_information()
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def get_mt_positions(self) -> list:
        """Get all open positions on MT account."""
        if not self.mt_connection:
            return []
        try:
            return await self.mt_connection.get_positions()
        except Exception as e:
            logger.error(f"Error getting MT positions: {e}")
            return []

    def map_symbol(self, ft_pair: str) -> str | None:
        symbol = SYMBOL_MAP.get(ft_pair)
        if not symbol:
            logger.warning(f"No symbol mapping for {ft_pair}")
        return symbol

    # FTMO MT5 symbol specs: 1 lot = contractSize units of base currency
    # Max volume = 5 lots for all crypto, volumeStep = 0.01
    SYMBOL_SPECS = {
        "SOLUSD": {"contract_size": 100, "max_vol": 5.0, "step": 0.01},   # 1 lot = 100 SOL
        "ETHUSD": {"contract_size": 10,  "max_vol": 5.0, "step": 0.01},   # 1 lot = 10 ETH
        "BTCUSD": {"contract_size": 1,   "max_vol": 5.0, "step": 0.01},   # 1 lot = 1 BTC
        "LTCUSD": {"contract_size": 100, "max_vol": 5.0, "step": 0.01},   # 1 lot = 100 LTC
    }

    def calculate_volume(self, symbol: str, entry_price: float, account_balance: float,
                         stoploss_pct: float | None = None,
                         bot_name: str | None = None,
                         garch_multiplier: float = 1.0) -> float:
        """
        Calculate position size in FTMO lots using per-bot risk profile.
        Applies GARCH volatility filter to dynamically scale risk.

        Formula: lots = risk_amount / (sl_distance * contract_size)
        Where: risk_amount = balance * risk_pct * garch_multiplier
               sl_distance = entry * sl_pct (price move per unit)
               contract_size = units per lot (e.g. 100 for SOL)
        """
        profile = risk_mgr.get_profile(bot_name or "")
        base_risk_pct = profile["risk_per_trade"]
        # Apply agent supervisor multiplier (stacks with GARCH)
        _, agent_mult, agent_reason = risk_mgr.get_agent_adjusted_risk(bot_name or "")
        risk_pct = base_risk_pct * garch_multiplier * agent_mult
        if agent_mult < 1.0:
            logger.info(f"Agent risk adj [{bot_name}]: ×{agent_mult:.2f} ({agent_reason})")
        sl_pct = stoploss_pct or profile["default_sl_pct"]

        spec = self.SYMBOL_SPECS.get(symbol, {"contract_size": 1, "max_vol": 5.0, "step": 0.01})
        contract_size = spec["contract_size"]
        max_vol = spec["max_vol"]
        step = spec["step"]

        risk_amount = account_balance * risk_pct
        sl_distance = entry_price * sl_pct

        # lots = risk_amount / (sl_distance_per_unit * units_per_lot)
        if sl_distance > 0:
            volume = risk_amount / (sl_distance * contract_size)
        else:
            volume = 0.01

        # Round to step size
        volume = round(volume / step) * step
        volume = round(volume, 2)  # clean floating point

        volume = max(0.01, min(volume, max_vol))

        garch_tag = f", GARCH={garch_multiplier:.2f}x" if garch_multiplier < 1.0 else ""
        logger.info(f"Volume calc [{bot_name}]: risk={base_risk_pct:.2%}→{risk_pct:.2%}{garch_tag}, "
                     f"sl={sl_pct:.3f}, contract={contract_size}, "
                     f"raw_lots={risk_amount/(sl_distance*contract_size) if sl_distance > 0 else 0:.4f}, "
                     f"final={volume} lots (max={max_vol})")
        return volume

    def _calculate_safety_tp(self, entry_price: float, stoploss: float, is_short: bool) -> float | None:
        """
        Calculate SAFETY take profit — wide 5:1 R:R as catastrophe protection only.

        The real exit is managed by each Freqtrade strategy (ROI tables, trailing stops,
        custom exits, circuit breakers). This TP is just a safety net on MT5 in case
        the bridge loses connection while a position is massively in profit.

        Professional approach: let the strategy decide when to exit, don't impose
        a fixed TP that overrides the strategy's carefully backtested exit logic.
        """
        if entry_price <= 0 or stoploss <= 0:
            return None

        sl_distance = abs(entry_price - stoploss)
        rr_ratio = 5.0  # 5:1 — wide safety net, not a target

        if is_short:
            tp = entry_price - (sl_distance * rr_ratio)
        else:
            tp = entry_price + (sl_distance * rr_ratio)

        # Sanity check
        if not is_short and tp <= entry_price:
            return None
        if is_short and tp >= entry_price:
            return None

        return round(tp, 5)

    async def open_position(self, ft_trade: dict, bot_name: str = "") -> str | None:
        """Open a position on FTMO MT4/MT5 mirroring a Freqtrade trade."""
        pair = ft_trade.get("pair", "")
        symbol = self.map_symbol(pair)
        if not symbol:
            return None

        side = ft_trade.get("trade_direction", ft_trade.get("direction", ""))
        is_short = ft_trade.get("is_short", False) or side == "short"
        action = "ORDER_TYPE_SELL" if is_short else "ORDER_TYPE_BUY"

        entry_price = ft_trade.get("open_rate", 0)
        stoploss = ft_trade.get("stop_loss_abs", 0)

        # Calculate SL distance for sizing
        sl_pct = None
        if entry_price > 0 and stoploss > 0:
            sl_pct = abs(entry_price - stoploss) / entry_price

        # Safety TP (wide 5:1 R:R — real exits managed by Freqtrade strategy)
        take_profit = self._calculate_safety_tp(entry_price, stoploss, is_short)

        # GARCH volatility filter — fetch recent candles and get risk multiplier
        garch_multiplier = 1.0
        try:
            client = self.ft_clients.get(bot_name)
            if client:
                # QuantSOL uses 5m, others use 15m
                tf = "5m" if bot_name == "QuantSOL" else "15m"
                closes = client.get_pair_candles(pair, timeframe=tf, limit=500)
                if closes and len(closes) >= 100:
                    garch_multiplier = garch_filter.get_risk_multiplier(symbol, closes)
                    regime_info = garch_filter.get_regime(symbol)
                    logger.info(
                        f"GARCH [{symbol}]: {regime_info.get('regime', '?')} "
                        f"→ {garch_multiplier:.2f}x risk"
                    )
                    if garch_multiplier == 0.0:
                        logger.warning(
                            f"GARCH BLOCKED [{bot_name}/{symbol}]: EXTREME vol regime — skipping trade"
                        )
                        return None
        except Exception as e:
            logger.warning(f"GARCH filter error (proceeding with full risk): {e}")
            garch_multiplier = 1.0

        # Get account balance for sizing
        account_info = await self.get_mt_account_info()
        balance = account_info["balance"] if account_info else risk_mgr.FTMO_INITIAL_BALANCE
        volume = self.calculate_volume(symbol, entry_price, balance, sl_pct, bot_name, garch_multiplier)

        profile = risk_mgr.get_profile(bot_name)
        logger.info(
            f"OPENING [{bot_name}]: {action} {volume} {symbol} @ ~{entry_price:.2f}, "
            f"SL={stoploss:.2f}, TP={take_profit}, risk={profile['risk_per_trade']:.2%}"
        )

        if not self.mt_connection:
            logger.info("[DRY MODE] Would open position")
            dry_id = f"dry_{ft_trade.get('trade_id', 0)}"
            # Register in shared risk manager even in dry mode
            risk_mgr.register_position(
                bot_name, dry_id, symbol,
                "short" if is_short else "long", volume,
                profile["risk_per_trade"]
            )
            return dry_id

        try:
            # Open without stops first, then modify to add SL/TP
            # (some FTMO symbols reject inline stops on market orders)
            if is_short:
                result = await self.mt_connection.create_market_sell_order(
                    symbol=symbol, volume=volume)
            else:
                result = await self.mt_connection.create_market_buy_order(
                    symbol=symbol, volume=volume)

            position_id = result.get("positionId", result.get("orderId", "unknown"))
            logger.info(f"Position opened: {position_id}, adding SL={stoploss}/TP={take_profit}...")

            # Add SL/TP via position modify — CRITICAL: never run naked on FTMO
            await asyncio.sleep(2)  # wait for position to settle
            sl_tp_set = False
            modify_kwargs = {}
            if stoploss > 0:
                modify_kwargs["stop_loss"] = stoploss
            if take_profit:
                modify_kwargs["take_profit"] = take_profit

            if modify_kwargs:
                for retry in range(3):
                    try:
                        await self.mt_connection.modify_position(
                            str(position_id), **modify_kwargs)
                        logger.info(f"SL/TP set on {position_id}: SL={stoploss}, TP={take_profit}")
                        sl_tp_set = True
                        break
                    except Exception as mod_err:
                        logger.warning(f"SL/TP attempt {retry+1}/3 failed: {mod_err}")
                        await asyncio.sleep(2)

                if not sl_tp_set:
                    # SAFETY: close position immediately — never trade without stops on FTMO
                    logger.error(
                        f"CRITICAL: Could not set SL/TP after 3 attempts — "
                        f"closing position {position_id}")
                    try:
                        await self.mt_connection.close_position(str(position_id))
                    except Exception as close_err:
                        logger.error(f"Failed to close naked position: {close_err}")
                    return None
            else:
                logger.warning(f"No SL/TP to set on {position_id} (missing price data)")

            # Register in shared risk manager
            risk_mgr.register_position(
                bot_name, str(position_id), symbol,
                "short" if is_short else "long", volume,
                profile["risk_per_trade"]
            )
            return str(position_id)

        except Exception as e:
            logger.error(f"Failed to open FTMO position: {e}")
            return None

    async def close_position(self, mt_position_id: str, ft_key: str, bot_name: str = ""):
        """Close a position on FTMO MT4/MT5."""
        logger.info(f"CLOSING [{bot_name}]: position={mt_position_id}, ft_key={ft_key}")

        # Unregister from shared risk manager
        risk_mgr.unregister_position(bot_name, mt_position_id)

        if not self.mt_connection or mt_position_id.startswith("dry_"):
            logger.info("[DRY MODE] Would close position")
            return

        try:
            await self.mt_connection.close_position(mt_position_id)
            logger.info(f"FTMO position closed: {mt_position_id}")
        except Exception as e:
            logger.error(f"Failed to close FTMO position {mt_position_id}: {e}")

    async def sync_sl_to_mt5(self, ft_trades_by_key: dict):
        """
        Sync Freqtrade's live stop_loss_abs to MT5 positions.

        Professional trading rule: the strategy is the brain, MT5 is execution.
        When Freqtrade trails its SL upward, MT5 must follow. This protects
        profits on the FTMO account exactly as the strategy intends.

        Only moves SL in the favorable direction (tighter) — never widens it.
        """
        if not self.mt_connection:
            return

        mt_positions = await self.get_mt_positions()
        mt_pos_map = {str(p.get("id", "")): p for p in mt_positions}

        for ft_key, mt_id in self.mirrored_trades.items():
            if mt_id.startswith("dry_") or mt_id not in mt_pos_map:
                continue

            ft_trade = ft_trades_by_key.get(ft_key)
            if not ft_trade:
                continue

            mt_pos = mt_pos_map[mt_id]
            ft_sl = ft_trade.get("stop_loss_abs", 0)
            mt_sl = mt_pos.get("stopLoss", 0)

            if not ft_sl or ft_sl <= 0 or not mt_sl:
                continue

            is_short = "SELL" in mt_pos.get("type", "")

            # Only tighten SL, never widen
            # Long: new SL must be HIGHER than current (tighter protection)
            # Short: new SL must be LOWER than current (tighter protection)
            should_update = False
            if is_short and ft_sl < mt_sl:
                should_update = True
            elif not is_short and ft_sl > mt_sl:
                should_update = True

            if should_update:
                try:
                    await self.mt_connection.modify_position(
                        mt_id, stop_loss=round(ft_sl, 5))
                    logger.info(
                        f"SL SYNCED [{ft_key}]: MT5 SL {mt_sl:.5f} → {ft_sl:.5f} "
                        f"({'SHORT' if is_short else 'LONG'})")
                except Exception as e:
                    logger.warning(f"SL sync failed for {ft_key}: {e}")

    async def cleanup_stale_positions(self):
        """
        Detect positions that were closed by MT SL/TP but bridge still tracks.
        This prevents phantom positions from blocking new trades.
        """
        if not self.mt_connection:
            return

        try:
            mt_positions = await self.get_mt_positions()
            mt_position_ids = {str(p.get("id", "")) for p in mt_positions}

            stale = []
            for ft_key, mt_id in self.mirrored_trades.items():
                if mt_id.startswith("dry_"):
                    continue
                if mt_id not in mt_position_ids:
                    stale.append((ft_key, mt_id))

            for ft_key, mt_id in stale:
                bot_name = ft_key.split(":")[0] if ":" in ft_key else ""
                logger.info(f"Cleaning stale position: {ft_key} -> {mt_id} (closed by SL/TP)")
                self.mirrored_trades.pop(ft_key, None)
                risk_mgr.unregister_position(bot_name, mt_id)
                self._save_state()

        except Exception as e:
            logger.error(f"Error cleaning stale positions: {e}")

    async def check_mt_connection(self) -> bool:
        """Check if MetaApi connection is alive, reconnect if dead.
        All MetaApi calls wrapped in asyncio.wait_for to prevent hanging."""
        if not self.mt_connection:
            # Connection was set to None after a previous failure — try fresh connect
            logger.info("MetaApi connection is None, attempting fresh connect...")
            try:
                connected = await self.connect_metaapi()
                if connected:
                    logger.info("MetaApi fresh reconnect succeeded")
                    return True
                else:
                    logger.warning("MetaApi fresh reconnect failed — staying in DRY MODE")
                    return False
            except Exception as e:
                logger.error(f"MetaApi fresh reconnect error: {e}")
                return False
        try:
            info = await asyncio.wait_for(
                self.mt_connection.get_account_information(), timeout=15
            )
            return info is not None and "balance" in info
        except Exception:
            logger.warning("MetaApi connection dead, attempting reconnect...")
            try:
                if self.mt_connection:
                    try:
                        await asyncio.wait_for(self.mt_connection.close(), timeout=10)
                    except Exception:
                        pass
                self.mt_connection = self.mt_account.get_rpc_connection()
                await asyncio.wait_for(self.mt_connection.connect(), timeout=30)
                await asyncio.wait_for(
                    self.mt_connection.wait_synchronized(), timeout=60
                )
                logger.info("MetaApi reconnected successfully")
                return True
            except asyncio.TimeoutError:
                logger.error("MetaApi reconnect TIMED OUT (60s) — trying fresh connect...")
                self.mt_connection = None
                # Aggressive retry: don't wait for next cycle, try fresh connect now
                try:
                    connected = await self.connect_metaapi()
                    if connected:
                        logger.info("MetaApi fresh reconnect succeeded")
                        return True
                    return False
                except Exception:
                    return False
            except Exception as e:
                logger.error(f"MetaApi reconnect failed: {e}")
                return False

    # Crypto symbols trade 24/7 on FTMO MT5 (including weekends)
    CRYPTO_SYMBOLS = {"SOLUSD", "ETHUSD", "BTCUSD", "LTCUSD"}

    async def sync_trades(self):
        """
        Main sync loop — poll ALL bots, open new, close finished.
        Uses shared risk manager for FTMO limit coordination.
        """
        now = datetime.now(timezone.utc)
        is_weekend = now.weekday() >= 5

        # Session filter: skip dead zone for NON-CRYPTO (but always keep MT alive)
        skip_non_crypto = not is_weekend and 0 <= now.hour < 3

        # ALWAYS check MetaApi connection — even during dead zone.
        # Letting it stay dead during off-hours means it's still dead when markets open.
        mt_alive = await self.check_mt_connection()

        # Check FTMO limits via actual account (if connected)
        account_info = await self.get_mt_account_info() if mt_alive else None
        within_limits = True
        if account_info:
            equity = account_info.get("equity", risk_mgr.FTMO_INITIAL_BALANCE)
            balance = account_info.get("balance", risk_mgr.FTMO_INITIAL_BALANCE)
            within_limits, reason = risk_mgr.check_ftmo_limits(equity, balance)
            if not within_limits:
                logger.warning(f"FTMO LIMITS BREACHED: {reason}")

        # Clean up stale positions (closed by SL/TP on MT side)
        if mt_alive:
            await self.cleanup_stale_positions()

        # Collect all open trades across all bots
        all_open_ids = set()
        ft_trades_by_key = {}  # for SL sync

        for bot_name, client in self.ft_clients.items():
            ft_open = client.get_open_trades()
            if ft_open is None:
                continue

            for trade in ft_open:
                raw_id = str(trade.get("trade_id", ""))
                if not raw_id:
                    continue

                ft_key = f"{bot_name}:{raw_id}"
                all_open_ids.add(ft_key)
                ft_trades_by_key[ft_key] = trade

                if ft_key not in self.mirrored_trades:
                    pair = trade.get("pair", "")
                    symbol = self.map_symbol(pair)
                    # Session filter: skip non-crypto during dead zone / weekends
                    if symbol and symbol not in self.CRYPTO_SYMBOLS:
                        if is_weekend or skip_non_crypto:
                            continue

                    # New trade — check if we can open
                    if not within_limits:
                        logger.warning(f"FTMO limit breached — NOT mirroring {ft_key}")
                        continue

                    # Shared risk manager check (coordinates with Gold)
                    profile = risk_mgr.get_profile(bot_name)
                    can_open, reason = risk_mgr.can_open_position(
                        bot_name, profile["risk_per_trade"]
                    )
                    if not can_open:
                        logger.info(f"Risk manager blocked {ft_key}: {reason}")
                        continue

                    logger.info(
                        f"New trade from {bot_name}: {trade.get('pair')} "
                        f"({'short' if trade.get('is_short') else 'long'})"
                    )
                    mt_id = await self.open_position(trade, bot_name)
                    if mt_id:
                        self.mirrored_trades[ft_key] = mt_id
                        self._save_state()

        # Sync trailing stops: Freqtrade SL → MT5 SL (every cycle)
        if mt_alive and self.mirrored_trades:
            await self.sync_sl_to_mt5(ft_trades_by_key)

        # Close finished positions
        closed_keys = set(self.mirrored_trades.keys()) - all_open_ids
        for ft_key in closed_keys:
            mt_id = self.mirrored_trades.pop(ft_key, None)
            if mt_id:
                bot_name = ft_key.split(":")[0] if ":" in ft_key else ""
                await self.close_position(mt_id, ft_key, bot_name)
                self._save_state()

    async def run(self):
        """Main loop with auto-reconnect."""
        logger.info("=" * 60)
        logger.info("FTMO MetaApi Bridge v2.0 starting...")
        logger.info(f"Monitoring {len(self.ft_clients)} Freqtrade bots:")
        for name, client in self.ft_clients.items():
            profile = risk_mgr.get_profile(name)
            logger.info(f"  - {name}: {client.url} "
                        f"(risk={profile['risk_per_trade']:.2%}, max={profile['max_positions']})")
        logger.info(f"MetaApi: {'CONFIGURED' if METAAPI_TOKEN else 'DRY MODE'}")
        logger.info(f"Symbols: {SYMBOL_MAP}")
        logger.info(f"Combined risk cap: {risk_mgr.MAX_COMBINED_RISK_PCT:.1%}")
        logger.info(f"FTMO daily loss limit: {risk_mgr.MAX_DAILY_LOSS_PCT:.1%}")
        logger.info(f"FTMO total DD limit: {risk_mgr.MAX_TOTAL_DD_PCT:.1%}")
        logger.info("=" * 60)

        # Check Freqtrade connections
        any_alive = False
        for name, client in self.ft_clients.items():
            if client.is_alive():
                logger.info(f"  {name}: CONNECTED")
                any_alive = True
            else:
                logger.warning(f"  {name}: OFFLINE (will retry)")
        if not any_alive:
            logger.error("No Freqtrade bots reachable. Is at least one running?")
            return

        # Connect MetaApi
        connected = await self.connect_metaapi()
        if not connected:
            logger.info("Running in DRY MODE — will log but not send real orders")

        # Main polling loop with auto-reconnect
        logger.info(f"Starting sync loop (polling every {POLL_INTERVAL}s)...")
        consecutive_errors = 0
        last_status_log = 0

        while True:
            try:
                await self.sync_trades()
                consecutive_errors = 0

                # Periodic status log (every 5 min)
                now_ts = time.time()
                if now_ts - last_status_log >= 300:
                    last_status_log = now_ts
                    status = risk_mgr.get_status()
                    mt_ok = self.mt_connection is not None
                    # Check which bots are alive
                    bot_status = {n: c.is_alive() for n, c in self.ft_clients.items()}
                    garch_regimes = garch_filter.get_all_regimes()
                    garch_str = ", ".join(
                        f"{s}={r.get('regime','?')}({r.get('multiplier',1):.1f}x)"
                        for s, r in garch_regimes.items()
                    ) if garch_regimes else "no data yet"
                    logger.info(
                        f"Status: mt={'OK' if mt_ok else 'DEAD'}, "
                        f"positions={status['total_positions']}, "
                        f"risk={status['combined_risk']:.2%}, "
                        f"mirrored={len(self.mirrored_trades)}, "
                        f"bots={bot_status}, "
                        f"garch=[{garch_str}]"
                    )

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Sync error ({consecutive_errors}): {e}")

                # Auto-reconnect after 3 consecutive errors
                if consecutive_errors >= 3:
                    logger.warning("Too many errors, reconnecting MetaApi...")
                    try:
                        connected = await self.connect_metaapi()
                        if connected:
                            consecutive_errors = 0
                            logger.info("Reconnected successfully")
                        else:
                            logger.error("Reconnect failed, continuing in DRY MODE")
                    except Exception as re:
                        logger.error(f"Reconnect failed: {re}")

            await asyncio.sleep(POLL_INTERVAL)


async def main():
    bridge = FTMOBridge()
    await bridge.run()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════╗
    ║       FTMO MetaApi Bridge v2.0                ║
    ║  3 Freqtrade Bots → MetaApi → FTMO MT4/MT5   ║
    ║  Shared risk management with Gold strategy    ║
    ╚═══════════════════════════════════════════════╝

    Bots: FTMO_SmartMoney (8085), QuantSOL (8083), FTMO_Blitz (8086)
    Gold: Coordinated via ftmo_risk_manager.py

    Required env vars:
      METAAPI_TOKEN        - Your MetaApi API token
      METAAPI_ACCOUNT_ID   - Your FTMO account ID in MetaApi

    Risk profiles (per ftmo_risk_manager):
      FTMO_SmartMoney: 1.5% risk, max 2 positions
      QuantSOL:        0.75% risk, max 1 position
      FTMO_Blitz:      1.5% risk, max 1 position
      Gold:            1.5% risk, max 1 position (separate process)
      Combined cap:    3.5% max open risk
    """)
    asyncio.run(main())
