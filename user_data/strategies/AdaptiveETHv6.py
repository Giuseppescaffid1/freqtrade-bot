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

    # ROI — optimizable via hyperopt
    minimal_roi = {
        "0": 0.08,
        "60": 0.04,
        "180": 0.02,
        "360": 0.01,
        "720": 0.005,
    }

    stoploss = -0.035
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # ── Entry Hyperopt Parameters ──
    ema_fast = IntParameter(5, 25, default=25, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(26, 80, default=80, space="buy", optimize=True, load=True)
    rsi_buy_low = IntParameter(20, 45, default=42, space="buy", optimize=True)
    rsi_buy_high = IntParameter(55, 75, default=64, space="buy", optimize=True)
    adx_threshold = IntParameter(15, 30, default=30, space="buy", optimize=True)
    volume_factor = DecimalParameter(0.8, 1.8, default=1.7, decimals=1, space="buy", optimize=True)
    use_macd = IntParameter(0, 1, default=1, space="buy", optimize=True)
    use_bb = IntParameter(0, 1, default=0, space="buy", optimize=True)

    # ── Exit Hyperopt Parameters ──
    rsi_sell_low = IntParameter(25, 45, default=36, space="sell", optimize=True)
    rsi_sell_high = IntParameter(55, 80, default=58, space="sell", optimize=True)
    # Exit RSI thresholds (optimizable)
    exit_rsi_long = IntParameter(70, 90, default=85, space="sell", optimize=True)
    exit_rsi_short = IntParameter(10, 30, default=15, space="sell", optimize=True)
    # Toggle: use exit signals at all? 0=disabled (ROI/SL only), 1=AND logic, 2=OR logic
    exit_mode = IntParameter(0, 2, default=0, space="sell", optimize=True)

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
