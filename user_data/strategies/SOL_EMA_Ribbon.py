"""
SOL_EMA_Ribbon — EMA Ribbon Trend Rider

Uses a ribbon of 6 EMAs (8/13/21/34/55/89 — Fibonacci sequence).
When all EMAs align (stacked in order), it's a strong trend signal.
Entry on pullback to inner ribbon during aligned trend.
This is the classic "trend is your friend" approach — high win rate,
moderate profit per trade.
"""
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class SOL_EMA_Ribbon(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

    minimal_roi = {
        "0": 0.15,
        "30": 0.07,
        "90": 0.035,
        "180": 0.015,
        "360": 0,
    }

    stoploss = -0.14
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # Ribbon alignment strength (how many EMAs must be stacked)
    min_aligned = IntParameter(4, 6, default=5, space="buy", optimize=True, load=True)

    # RSI pullback zone
    rsi_buy_low = IntParameter(30, 45, default=38, space="buy", optimize=True, load=True)
    rsi_buy_high = IntParameter(50, 60, default=55, space="buy", optimize=True, load=True)
    rsi_sell_low = IntParameter(40, 50, default=45, space="sell", optimize=True, load=True)
    rsi_sell_high = IntParameter(55, 70, default=62, space="sell", optimize=True, load=True)

    # ADX
    adx_min = IntParameter(18, 35, default=25, space="buy", optimize=True, load=True)

    # Cooldown
    cooldown_candles = IntParameter(10, 60, default=30, space="buy", optimize=True, load=True)

    # Exit
    exit_rsi_long = IntParameter(70, 90, default=80, space="sell", optimize=True, load=True)
    exit_rsi_short = IntParameter(10, 30, default=20, space="sell", optimize=True, load=True)

    # Fibonacci EMA periods
    RIBBON_PERIODS = [8, 13, 21, 34, 55, 89]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA Ribbon
        for period in self.RIBBON_PERIODS:
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Count bullish alignment (each shorter EMA > longer EMA)
        def count_bull_align(row):
            count = 0
            for i in range(len(self.RIBBON_PERIODS) - 1):
                if row[f"ema_{self.RIBBON_PERIODS[i]}"] > row[f"ema_{self.RIBBON_PERIODS[i+1]}"]:
                    count += 1
            return count

        def count_bear_align(row):
            count = 0
            for i in range(len(self.RIBBON_PERIODS) - 1):
                if row[f"ema_{self.RIBBON_PERIODS[i]}"] < row[f"ema_{self.RIBBON_PERIODS[i+1]}"]:
                    count += 1
            return count

        dataframe["bull_aligned"] = dataframe.apply(count_bull_align, axis=1)
        dataframe["bear_aligned"] = dataframe.apply(count_bear_align, axis=1)

        # Price relative to ribbon (pullback detection)
        # Price between fastest and 3rd EMA = pullback zone
        dataframe["above_ema8"] = (dataframe["close"] > dataframe["ema_8"]).astype(int)
        dataframe["below_ema21"] = (dataframe["close"] < dataframe["ema_21"]).astype(int)
        dataframe["pullback_long"] = (
            (dataframe["close"] < dataframe["ema_8"]) &
            (dataframe["close"] > dataframe["ema_34"])
        )
        dataframe["pullback_short"] = (
            (dataframe["close"] > dataframe["ema_8"]) &
            (dataframe["close"] < dataframe["ema_34"])
        )

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        min_align = self.min_aligned.value
        adx_ok = dataframe["adx"] > self.adx_min.value

        # Bullish ribbon + price pullback to inner ribbon + RSI not overbought
        bull_ribbon = dataframe["bull_aligned"] >= min_align
        rsi_buy = (dataframe["rsi"] > self.rsi_buy_low.value) & (dataframe["rsi"] < self.rsi_buy_high.value)

        bear_ribbon = dataframe["bear_aligned"] >= min_align
        rsi_sell = (dataframe["rsi"] > self.rsi_sell_low.value) & (dataframe["rsi"] < self.rsi_sell_high.value)

        # MACD momentum confirmation
        macd_up = dataframe["macd_hist"] > dataframe["macd_hist"].shift(1)
        macd_down = dataframe["macd_hist"] < dataframe["macd_hist"].shift(1)

        cooldown = self.cooldown_candles.value

        # LONG: aligned uptrend + pullback + RSI in buy zone + momentum
        long_signal = bull_ribbon & dataframe["pullback_long"] & rsi_buy & adx_ok & macd_up
        long_cooled = long_signal & ~long_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[long_cooled, "enter_long"] = 1

        # SHORT: aligned downtrend + pullback + RSI in sell zone + momentum
        short_signal = bear_ribbon & dataframe["pullback_short"] & rsi_sell & adx_ok & macd_down
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
