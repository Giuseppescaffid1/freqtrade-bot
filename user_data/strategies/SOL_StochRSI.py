"""
SOL_StochRSI — Stochastic RSI Mean-Reversion Scalper

Uses Stochastic RSI (more sensitive than regular RSI) for precise
oversold/overbought detection within EMA trend. Williams %R as confirmation.
Targets 1 trade/day with tight profit targets.
"""
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class SOL_StochRSI(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

    minimal_roi = {
        "0": 0.12,
        "20": 0.06,
        "60": 0.03,
        "120": 0.015,
        "240": 0,
    }

    stoploss = -0.15
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # Trend
    ema_fast = IntParameter(8, 21, default=13, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(21, 55, default=34, space="buy", optimize=True, load=True)

    # StochRSI thresholds
    stoch_buy = IntParameter(10, 30, default=20, space="buy", optimize=True, load=True)
    stoch_sell = IntParameter(70, 90, default=80, space="sell", optimize=True, load=True)

    # Williams %R
    willr_buy = IntParameter(-90, -60, default=-80, space="buy", optimize=True, load=True)
    willr_sell = IntParameter(-40, -10, default=-20, space="sell", optimize=True, load=True)

    # ADX
    adx_min = IntParameter(15, 30, default=20, space="buy", optimize=True, load=True)

    # Cooldown
    cooldown_candles = IntParameter(10, 80, default=40, space="buy", optimize=True, load=True)

    # Exit
    exit_stoch_long = IntParameter(70, 95, default=85, space="sell", optimize=True, load=True)
    exit_stoch_short = IntParameter(5, 30, default=15, space="sell", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs
        for period in range(8, 56):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Stochastic RSI
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        dataframe["stoch_k"] = stoch_rsi["fastk"]
        dataframe["stoch_d"] = stoch_rsi["fastd"]

        # Williams %R
        dataframe["willr"] = ta.WILLR(dataframe, timeperiod=14)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        uptrend = dataframe[ema_fast] > dataframe[ema_slow]
        downtrend = dataframe[ema_fast] < dataframe[ema_slow]
        adx_ok = dataframe["adx"] > self.adx_min.value

        # StochRSI oversold + Williams %R oversold = strong buy
        stoch_buy = dataframe["stoch_k"] < self.stoch_buy.value
        willr_buy = dataframe["willr"] < self.willr_buy.value

        stoch_sell = dataframe["stoch_k"] > self.stoch_sell.value
        willr_sell = dataframe["willr"] > self.willr_sell.value

        cooldown = self.cooldown_candles.value

        # LONG
        long_signal = uptrend & stoch_buy & willr_buy & adx_ok
        long_cooled = long_signal & ~long_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[long_cooled, "enter_long"] = 1

        # SHORT
        short_signal = downtrend & stoch_sell & willr_sell & adx_ok
        short_cooled = short_signal & ~short_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[short_cooled, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["stoch_k"] > self.exit_stoch_long.value, "exit_long"] = 1
        dataframe.loc[dataframe["stoch_k"] < self.exit_stoch_short.value, "exit_short"] = 1
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
