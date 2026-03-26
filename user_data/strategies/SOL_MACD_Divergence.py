"""
SOL_MACD_Divergence — MACD Crossover + RSI Divergence Hunter

Classic MACD signal-line crossover strategy with RSI divergence detection.
MACD crossovers are the bread-and-butter of institutional momentum trading.
Adding RSI divergence filters out false signals.
"""
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class SOL_MACD_Divergence(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

    minimal_roi = {
        "0": 0.16,
        "25": 0.08,
        "75": 0.04,
        "150": 0.015,
        "300": 0,
    }

    stoploss = -0.14
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.11
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # MACD params
    macd_fast = IntParameter(8, 16, default=12, space="buy", optimize=True, load=True)
    macd_slow = IntParameter(20, 30, default=26, space="buy", optimize=True, load=True)
    macd_signal = IntParameter(7, 12, default=9, space="buy", optimize=True, load=True)

    # RSI
    rsi_buy = IntParameter(25, 45, default=38, space="buy", optimize=True, load=True)
    rsi_sell = IntParameter(55, 75, default=62, space="sell", optimize=True, load=True)

    # Trend
    ema_trend = IntParameter(30, 100, default=50, space="buy", optimize=True, load=True)

    # ADX
    adx_min = IntParameter(15, 30, default=20, space="buy", optimize=True, load=True)

    # Cooldown
    cooldown_candles = IntParameter(10, 60, default=25, space="buy", optimize=True, load=True)

    # Exit
    exit_rsi_long = IntParameter(70, 90, default=80, space="sell", optimize=True, load=True)
    exit_rsi_short = IntParameter(10, 30, default=20, space="sell", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs
        for period in [20, 30, 40, 50, 60, 80, 100]:
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # MACD (multiple configs for hyperopt)
        for fast in [8, 10, 12, 14, 16]:
            for slow in [20, 24, 26, 28, 30]:
                for sig in [7, 9, 11]:
                    key = f"macd_{fast}_{slow}_{sig}"
                    macd = ta.MACD(dataframe, fastperiod=fast, slowperiod=slow, signalperiod=sig)
                    dataframe[f"{key}_line"] = macd["macd"]
                    dataframe[f"{key}_signal"] = macd["macdsignal"]
                    dataframe[f"{key}_hist"] = macd["macdhist"]

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema = f"ema_{self.ema_trend.value}"
        macd_key = f"macd_{self.macd_fast.value}_{self.macd_slow.value}_{self.macd_signal.value}"

        adx_ok = dataframe["adx"] > self.adx_min.value

        # MACD bullish crossover (line crosses above signal)
        macd_cross_up = (
            (dataframe[f"{macd_key}_line"] > dataframe[f"{macd_key}_signal"]) &
            (dataframe[f"{macd_key}_line"].shift(1) <= dataframe[f"{macd_key}_signal"].shift(1))
        )

        # MACD bearish crossover (line crosses below signal)
        macd_cross_down = (
            (dataframe[f"{macd_key}_line"] < dataframe[f"{macd_key}_signal"]) &
            (dataframe[f"{macd_key}_line"].shift(1) >= dataframe[f"{macd_key}_signal"].shift(1))
        )

        # RSI confirmation
        rsi_buy = dataframe["rsi"] < self.rsi_buy.value
        rsi_sell = dataframe["rsi"] > self.rsi_sell.value

        # Trend filter
        above_ema = dataframe["close"] > dataframe[ema]
        below_ema = dataframe["close"] < dataframe[ema]

        # Histogram momentum (rising for long, falling for short)
        hist_rising = dataframe[f"{macd_key}_hist"] > dataframe[f"{macd_key}_hist"].shift(1)
        hist_falling = dataframe[f"{macd_key}_hist"] < dataframe[f"{macd_key}_hist"].shift(1)

        cooldown = self.cooldown_candles.value

        # LONG: MACD cross up + RSI not overbought + above trend EMA
        long_signal = (macd_cross_up | hist_rising) & rsi_buy & above_ema & adx_ok
        long_cooled = long_signal & ~long_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[long_cooled, "enter_long"] = 1

        # SHORT: MACD cross down + RSI not oversold + below trend EMA
        short_signal = (macd_cross_down | hist_falling) & rsi_sell & below_ema & adx_ok
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
