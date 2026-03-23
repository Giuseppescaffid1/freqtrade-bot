"""
Adaptive ETH Strategy v5 — High Frequency Trend Following

15m timeframe for more trade opportunities.
Fewer entry conditions = more trades, but each condition is high-value.
Designed for hyperopt optimization across buy/sell/roi/stoploss/trailing spaces.
"""

import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class AdaptiveETHv5(IStrategy):
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

    # ── Hyperopt Parameters ──
    # Trend
    ema_fast = IntParameter(5, 25, default=12, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(26, 80, default=50, space="buy", optimize=True, load=True)

    # Entry filters
    rsi_buy_low = IntParameter(20, 45, default=30, space="buy", optimize=True)
    rsi_buy_high = IntParameter(55, 75, default=65, space="buy", optimize=True)
    rsi_sell_low = IntParameter(25, 45, default=35, space="sell", optimize=True)
    rsi_sell_high = IntParameter(55, 80, default=70, space="sell", optimize=True)
    adx_threshold = IntParameter(15, 30, default=20, space="buy", optimize=True)
    volume_factor = DecimalParameter(0.8, 1.8, default=1.1, decimals=1, space="buy", optimize=True)

    # Use MACD confirmation toggle
    use_macd = IntParameter(0, 1, default=1, space="buy", optimize=True)
    # Use BB bounce
    use_bb = IntParameter(0, 1, default=1, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # EMAs — pre-compute range for hyperopt
        for period in range(5, 81):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_width"] = (bb["upperband"] - bb["lowerband"]) / bb["middleband"]

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        # ATR for volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        # ── LONG ──
        # Core: trend alignment
        uptrend = dataframe[ema_fast] > dataframe[ema_slow]

        # RSI in buy zone (wide range for more signals)
        rsi_ok = (dataframe["rsi"] > self.rsi_buy_low.value) & (dataframe["rsi"] < self.rsi_buy_high.value)

        # Trend strength
        adx_ok = dataframe["adx"] > self.adx_threshold.value

        # Volume filter
        vol_ok = dataframe["volume"] > dataframe["vol_sma"] * self.volume_factor.value

        # Optional MACD confirmation
        macd_ok = (
            (self.use_macd.value == 0) |
            (dataframe["macd_hist"] > dataframe["macd_hist"].shift(1))
        )

        # Optional BB bounce (price near lower band = good entry in uptrend)
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

        # Exit long: trend reversal or RSI overbought
        dataframe.loc[
            (dataframe[ema_fast] < dataframe[ema_slow]) |
            (dataframe["rsi"] > 75),
            "exit_long"
        ] = 1

        # Exit short: trend reversal or RSI oversold
        dataframe.loc[
            (dataframe[ema_fast] > dataframe[ema_slow]) |
            (dataframe["rsi"] < 25),
            "exit_short"
        ] = 1

        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
