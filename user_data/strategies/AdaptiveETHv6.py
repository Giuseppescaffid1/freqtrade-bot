"""
Adaptive ETH Strategy v6 — Maximum Profit

Key improvements over v5:
- Exit signals use AND (not OR) — no more premature exits
- Optimizable exit RSI thresholds
- Toggle to disable exit signals entirely (let ROI/SL handle it)
- Optimizable max_open_trades for concurrent positions
"""

import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class AdaptiveETHv6(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

    # ROI — 2-year robust hyperopt (1000 epochs, 2024-2026)
    minimal_roi = {
        "0": 0.089,
        "44": 0.062,
        "211": 0.04,
        "405": 0,
    }

    stoploss = -0.246
    trailing_stop = True
    trailing_stop_positive = 0.08
    trailing_stop_positive_offset = 0.172
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # ── FreqUI Chart Indicators ──
    plot_config = {
        "main_plot": {
            "ema_10": {"color": "#10b981", "type": "line"},
            "ema_56": {"color": "#3b82f6", "type": "line"},
            "bb_upper": {"color": "#6b7280", "type": "line"},
            "bb_lower": {"color": "#6b7280", "type": "line"},
            "bb_mid": {"color": "#9ca3af", "type": "line"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "#f59e0b", "type": "line"},
            },
            "ADX": {
                "adx": {"color": "#10b981", "type": "line"},
            },
            "MACD": {
                "macd_hist": {"color": "#8b5cf6", "type": "bar", "fill_to": 0},
                "macd": {"color": "#3b82f6", "type": "line"},
                "macd_signal": {"color": "#ef4444", "type": "line"},
            },
        },
    }

    # ── Entry Parameters (2-year robust hyperopt) ──
    ema_fast = IntParameter(5, 25, default=10, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(26, 80, default=56, space="buy", optimize=True, load=True)
    rsi_buy_low = IntParameter(20, 45, default=43, space="buy", optimize=True, load=True)
    rsi_buy_high = IntParameter(55, 75, default=55, space="buy", optimize=True, load=True)
    adx_threshold = IntParameter(15, 30, default=29, space="buy", optimize=True, load=True)
    volume_factor = DecimalParameter(0.8, 2.0, default=0.9, decimals=1, space="buy", optimize=True, load=True)
    use_macd = IntParameter(0, 1, default=1, space="buy", optimize=True, load=True)
    use_bb = IntParameter(0, 1, default=1, space="buy", optimize=True, load=True)

    # ── Exit Parameters (2-year robust hyperopt) ──
    rsi_sell_low = IntParameter(25, 45, default=27, space="sell", optimize=True, load=True)
    rsi_sell_high = IntParameter(55, 80, default=57, space="sell", optimize=True, load=True)
    exit_rsi_long = IntParameter(70, 90, default=81, space="sell", optimize=True, load=True)
    exit_rsi_short = IntParameter(10, 30, default=16, space="sell", optimize=True, load=True)
    # Toggle: 0=disabled (ROI/SL only), 1=AND logic, 2=OR logic
    exit_mode = IntParameter(0, 2, default=2, space="sell", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        for period in range(5, 81):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]

        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        # ── LONG ──
        uptrend = dataframe[ema_fast] > dataframe[ema_slow]
        rsi_ok = (dataframe["rsi"] > self.rsi_buy_low.value) & (dataframe["rsi"] < self.rsi_buy_high.value)
        adx_ok = dataframe["adx"] > self.adx_threshold.value
        vol_ok = dataframe["volume"] > dataframe["vol_sma"] * self.volume_factor.value

        macd_ok = (
            (self.use_macd.value == 0) |
            (dataframe["macd_hist"] > dataframe["macd_hist"].shift(1))
        )
        bb_ok = (
            (self.use_bb.value == 0) |
            (dataframe["close"] < dataframe["bb_mid"])
        )

        dataframe.loc[
            uptrend & rsi_ok & adx_ok & vol_ok & macd_ok & bb_ok,
            "enter_long"
        ] = 1

        # ── SHORT ──
        downtrend = dataframe[ema_fast] < dataframe[ema_slow]
        rsi_short = (dataframe["rsi"] > self.rsi_sell_low.value) & (dataframe["rsi"] < self.rsi_sell_high.value)

        macd_short = (
            (self.use_macd.value == 0) |
            (dataframe["macd_hist"] < dataframe["macd_hist"].shift(1))
        )
        bb_short = (
            (self.use_bb.value == 0) |
            (dataframe["close"] > dataframe["bb_mid"])
        )

        dataframe.loc[
            downtrend & rsi_short & adx_ok & vol_ok & macd_short & bb_short,
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        if self.exit_mode.value == 0:
            # No exit signals — rely entirely on ROI + stoploss + trailing
            return dataframe

        trend_reversed_long = dataframe[ema_fast] < dataframe[ema_slow]
        rsi_overbought = dataframe["rsi"] > self.exit_rsi_long.value
        trend_reversed_short = dataframe[ema_fast] > dataframe[ema_slow]
        rsi_oversold = dataframe["rsi"] < self.exit_rsi_short.value

        if self.exit_mode.value == 1:
            # AND logic — both conditions must be true (much stricter)
            dataframe.loc[
                trend_reversed_long & rsi_overbought,
                "exit_long"
            ] = 1
            dataframe.loc[
                trend_reversed_short & rsi_oversold,
                "exit_short"
            ] = 1
        else:
            # OR logic — either condition triggers exit (v5 behavior)
            dataframe.loc[
                trend_reversed_long | rsi_overbought,
                "exit_long"
            ] = 1
            dataframe.loc[
                trend_reversed_short | rsi_oversold,
                "exit_short"
            ] = 1

        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
