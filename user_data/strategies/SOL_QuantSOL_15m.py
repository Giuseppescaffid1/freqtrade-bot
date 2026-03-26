"""
SOL_QuantSOL_15m — Current QuantSOL logic ported to 15m timeframe.
Baseline comparison to see if 15m is better than 5m for the same logic.
"""
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class SOL_QuantSOL_15m(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

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

    ema_fast = IntParameter(5, 15, default=11, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(15, 30, default=30, space="buy", optimize=True, load=True)
    rsi_buy_upper = IntParameter(35, 55, default=47, space="buy", optimize=True, load=True)
    rsi_sell_lower = IntParameter(45, 65, default=51, space="sell", optimize=True, load=True)
    adx_min = IntParameter(10, 25, default=21, space="buy", optimize=True, load=True)
    cooldown_candles = IntParameter(10, 60, default=15, space="buy", optimize=True, load=True)
    exit_rsi_long = IntParameter(65, 85, default=78, space="sell", optimize=True, load=True)
    exit_rsi_short = IntParameter(15, 35, default=25, space="sell", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        for period in range(5, 31):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"
        uptrend = dataframe[ema_fast] > dataframe[ema_slow]
        downtrend = dataframe[ema_fast] < dataframe[ema_slow]
        adx_ok = dataframe["adx"] > self.adx_min.value
        rsi_buy = dataframe["rsi"] < self.rsi_buy_upper.value
        rsi_sell = dataframe["rsi"] > self.rsi_sell_lower.value
        macd_up = dataframe["macd_hist"] > dataframe["macd_hist"].shift(1)
        macd_down = dataframe["macd_hist"] < dataframe["macd_hist"].shift(1)
        cooldown = self.cooldown_candles.value

        long_signal = uptrend & rsi_buy & adx_ok & macd_up
        long_cooled = long_signal & ~long_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[long_cooled, "enter_long"] = 1

        short_signal = downtrend & rsi_sell & adx_ok & macd_down
        short_cooled = short_signal & ~short_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[short_cooled, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["rsi"] > self.exit_rsi_long.value, "exit_long"] = 1
        dataframe.loc[dataframe["rsi"] < self.exit_rsi_short.value, "exit_short"] = 1
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
