"""
Adaptive ETH Strategy v4 — Optimized via Hyperopt

Trend-following with wider stops and longer holding periods.
Designed for 1h timeframe to avoid noise and give trades room.
"""

import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame


class AdaptiveETH(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True

    # Wide ROI — let winners run
    minimal_roi = {
        "0": 0.10,
        "360": 0.05,
        "720": 0.03,
        "1440": 0.015,
        "2880": 0.005,
    }

    # Wider stoploss — give trades room
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # ── Hyperopt Parameters ──
    rsi_buy = IntParameter(25, 45, default=42, space="buy", optimize=True)
    rsi_sell = IntParameter(55, 80, default=80, space="sell", optimize=True)
    ema_fast = IntParameter(10, 30, default=14, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(40, 100, default=74, space="buy", optimize=True, load=True)
    adx_threshold = IntParameter(15, 35, default=33, space="buy", optimize=True)
    volume_factor = DecimalParameter(0.5, 2.0, default=1.7, decimals=1, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Pre-compute all EMAs that hyperopt might request
        for period in range(10, 101):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=20)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["slowk"] = stoch["slowk"]
        dataframe["slowd"] = stoch["slowd"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        # ── LONG ──
        # Trend: fast EMA above slow EMA (uptrend)
        uptrend = dataframe[ema_fast] > dataframe[ema_slow]

        # Pullback: RSI in buy zone (not oversold, just pulled back)
        rsi_ok = (dataframe["rsi"] > self.rsi_buy.value) & (dataframe["rsi"] < self.rsi_buy.value + 15)

        # RSI turning up
        rsi_turning = dataframe["rsi"] > dataframe["rsi"].shift(1)

        # MACD momentum positive or improving
        macd_ok = (dataframe["macd_hist"] > dataframe["macd_hist"].shift(1))

        # Volume decent
        vol_ok = dataframe["volume"] > dataframe["vol_sma"] * self.volume_factor.value

        # ADX: trend exists
        adx_ok = dataframe["adx"] > self.adx_threshold.value

        # Price above EMA200 (major trend)
        above_200 = dataframe["close"] > dataframe["ema_200"]

        # Stochastic not overbought
        stoch_ok = dataframe["slowk"] < 80

        dataframe.loc[
            uptrend & rsi_ok & rsi_turning & macd_ok & vol_ok & adx_ok & above_200 & stoch_ok,
            "enter_long"
        ] = 1

        # ── SHORT ──
        downtrend = dataframe[ema_fast] < dataframe[ema_slow]
        rsi_high = (dataframe["rsi"] < self.rsi_sell.value) & (dataframe["rsi"] > self.rsi_sell.value - 15)
        rsi_turning_down = dataframe["rsi"] < dataframe["rsi"].shift(1)
        macd_down = dataframe["macd_hist"] < dataframe["macd_hist"].shift(1)
        below_200 = dataframe["close"] < dataframe["ema_200"]
        stoch_down = dataframe["slowk"] > 20

        dataframe.loc[
            downtrend & rsi_high & rsi_turning_down & macd_down & vol_ok & adx_ok & below_200 & stoch_down,
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        # Exit long when trend reverses
        dataframe.loc[
            (dataframe[ema_fast] < dataframe[ema_slow]) & (dataframe["rsi"] > 70),
            "exit_long"
        ] = 1

        # Exit short when trend reverses
        dataframe.loc[
            (dataframe[ema_fast] > dataframe[ema_slow]) & (dataframe["rsi"] < 30),
            "exit_short"
        ] = 1

        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
