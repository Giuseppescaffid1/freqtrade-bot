"""
QuantSOL — High-Frequency Intraday Solana Scalper

Targets ~1 trade per day on SOL/USDT perpetual futures (Bybit).
Uses RSI mean-reversion + EMA trend filter on 5-min bars.
Tight risk management: small gains, fast exits.

Designed for SOL/USDT perpetual futures on Bybit.
"""

import talib.abstract as ta
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class QuantSOL(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True

    # ROI targets (800-epoch hyperopt optimized)
    minimal_roi = {
        "0": 0.235,
        "23": 0.088,
        "80": 0.022,
        "190": 0,
    }

    stoploss = -0.197
    trailing_stop = True
    trailing_stop_positive = 0.076
    trailing_stop_positive_offset = 0.167
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # ── FreqUI Chart Indicators ──
    plot_config = {
        "main_plot": {
            "ema_11": {"color": "#a855f7", "type": "line"},
            "ema_30": {"color": "#3b82f6", "type": "line"},
            "ema_50": {"color": "#6b7280", "type": "line"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "#f59e0b", "type": "line"},
            },
            "ADX": {
                "adx": {"color": "#10b981", "type": "line"},
            },
            "MACD Histogram": {
                "macd_hist": {"color": "#8b5cf6", "type": "bar", "fill_to": 0},
            },
        },
    }

    # ── Trend Filter ──
    ema_fast = IntParameter(5, 15, default=11, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(15, 30, default=30, space="buy", optimize=True, load=True)

    # ── RSI Entry ──
    rsi_buy_upper = IntParameter(35, 55, default=47, space="buy", optimize=True, load=True)
    rsi_sell_lower = IntParameter(45, 65, default=51, space="sell", optimize=True, load=True)

    # ── ADX minimum ──
    adx_min = IntParameter(10, 25, default=21, space="buy", optimize=True, load=True)

    # ── Cooldown: min candles between entries ──
    cooldown_candles = IntParameter(10, 60, default=15, space="buy", optimize=True, load=True)

    # ── Exit ──
    exit_rsi_long = IntParameter(65, 85, default=78, space="sell", optimize=True, load=True)
    exit_rsi_short = IntParameter(15, 35, default=25, space="sell", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # EMAs
        for period in range(5, 31):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # EMA 50 for trend context
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # ATR for volatility context
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # MACD for momentum confirmation
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        # Volume SMA
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        # Trend direction
        uptrend = dataframe[ema_fast] > dataframe[ema_slow]
        downtrend = dataframe[ema_fast] < dataframe[ema_slow]

        # Minimum trend strength
        adx_ok = dataframe["adx"] > self.adx_min.value

        # RSI pullback entries (mean-reversion within trend)
        # Long: RSI dips below threshold in uptrend → buy the dip
        rsi_buy = dataframe["rsi"] < self.rsi_buy_upper.value
        # Short: RSI spikes above threshold in downtrend → sell the rally
        rsi_sell = dataframe["rsi"] > self.rsi_sell_lower.value

        # MACD momentum confirmation
        macd_up = dataframe["macd_hist"] > dataframe["macd_hist"].shift(1)
        macd_down = dataframe["macd_hist"] < dataframe["macd_hist"].shift(1)

        # Cooldown: prevent overtrading by requiring gap between signals
        # Use a rolling window — only enter if no signal in last N candles
        cooldown = self.cooldown_candles.value

        # ── LONG ──
        long_signal = uptrend & rsi_buy & adx_ok & macd_up
        # Apply cooldown: suppress signal if there was one recently
        long_cooled = long_signal & ~long_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)

        dataframe.loc[long_cooled, "enter_long"] = 1

        # ── SHORT ──
        short_signal = downtrend & rsi_sell & adx_ok & macd_down
        short_cooled = short_signal & ~short_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)

        dataframe.loc[short_cooled, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long when RSI hits overbought
        dataframe.loc[
            dataframe["rsi"] > self.exit_rsi_long.value,
            "exit_long",
        ] = 1

        # Exit short when RSI hits oversold
        dataframe.loc[
            dataframe["rsi"] < self.exit_rsi_short.value,
            "exit_short",
        ] = 1

        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
