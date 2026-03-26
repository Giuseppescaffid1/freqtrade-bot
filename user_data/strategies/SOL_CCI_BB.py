"""
SOL_CCI_BB — CCI + Bollinger Band Mean-Reversion

Uses Commodity Channel Index (CCI) for overbought/oversold detection
combined with Bollinger Band position. CCI is faster than RSI for
detecting momentum shifts — ideal for intraday scalping.
"""
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class SOL_CCI_BB(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True

    minimal_roi = {
        "0": 0.18,
        "15": 0.08,
        "45": 0.035,
        "100": 0.015,
        "200": 0,
    }

    stoploss = -0.16
    trailing_stop = True
    trailing_stop_positive = 0.06
    trailing_stop_positive_offset = 0.12
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # Trend
    ema_fast = IntParameter(8, 21, default=12, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(21, 55, default=40, space="buy", optimize=True, load=True)

    # CCI
    cci_buy = IntParameter(-200, -50, default=-100, space="buy", optimize=True, load=True)
    cci_sell = IntParameter(50, 200, default=100, space="sell", optimize=True, load=True)

    # BB position (0=lower, 1=upper)
    bb_buy_pct = DecimalParameter(0.0, 0.3, default=0.15, space="buy", optimize=True, load=True)
    bb_sell_pct = DecimalParameter(0.7, 1.0, default=0.85, space="sell", optimize=True, load=True)

    # ADX
    adx_min = IntParameter(12, 28, default=18, space="buy", optimize=True, load=True)

    # Cooldown
    cooldown_candles = IntParameter(10, 60, default=20, space="buy", optimize=True, load=True)

    # Exit
    exit_cci_long = IntParameter(80, 200, default=150, space="sell", optimize=True, load=True)
    exit_cci_short = IntParameter(-200, -80, default=-150, space="sell", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for period in range(8, 56):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # CCI
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)

        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_width"] = bb["upperband"] - bb["lowerband"]
        # BB position: 0 = at lower band, 1 = at upper band
        dataframe["bb_pct"] = (dataframe["close"] - dataframe["bb_lower"]) / dataframe["bb_width"]

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # MFI (Money Flow Index — volume-weighted RSI)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        uptrend = dataframe[ema_fast] > dataframe[ema_slow]
        downtrend = dataframe[ema_fast] < dataframe[ema_slow]
        adx_ok = dataframe["adx"] > self.adx_min.value

        # CCI oversold + price near lower BB = buy
        cci_buy = dataframe["cci"] < self.cci_buy.value
        bb_buy = dataframe["bb_pct"] < self.bb_buy_pct.value

        cci_sell = dataframe["cci"] > self.cci_sell.value
        bb_sell = dataframe["bb_pct"] > self.bb_sell_pct.value

        cooldown = self.cooldown_candles.value

        # LONG
        long_signal = uptrend & cci_buy & bb_buy & adx_ok
        long_cooled = long_signal & ~long_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[long_cooled, "enter_long"] = 1

        # SHORT
        short_signal = downtrend & cci_sell & bb_sell & adx_ok
        short_cooled = short_signal & ~short_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[short_cooled, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["cci"] > self.exit_cci_long.value, "exit_long"] = 1
        dataframe.loc[dataframe["cci"] < self.exit_cci_short.value, "exit_short"] = 1
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
