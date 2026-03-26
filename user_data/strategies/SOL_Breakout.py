"""
SOL_Breakout — Donchian Channel Breakout + ATR Volatility Filter

Opposite approach to mean-reversion: buys breakouts of recent highs/lows.
Uses Donchian Channels (20-period high/low) — the classic turtle trading method.
ATR filter ensures we only trade when volatility is expanding (real breakout,
not noise). Targets 1 trade/day.
"""
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class SOL_Breakout(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True

    minimal_roi = {
        "0": 0.20,
        "60": 0.10,
        "180": 0.05,
        "360": 0.02,
        "720": 0,
    }

    stoploss = -0.18
    trailing_stop = True
    trailing_stop_positive = 0.06
    trailing_stop_positive_offset = 0.14
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # Donchian Channel period
    donchian_period = IntParameter(10, 30, default=20, space="buy", optimize=True, load=True)

    # ATR multiplier for volatility expansion
    atr_mult = DecimalParameter(1.0, 2.5, default=1.5, space="buy", optimize=True, load=True)

    # EMA trend
    ema_period = IntParameter(30, 100, default=50, space="buy", optimize=True, load=True)

    # ADX for trend strength
    adx_min = IntParameter(18, 35, default=25, space="buy", optimize=True, load=True)

    # Cooldown
    cooldown_candles = IntParameter(6, 30, default=12, space="buy", optimize=True, load=True)

    # Exit
    exit_rsi_long = IntParameter(70, 90, default=78, space="sell", optimize=True, load=True)
    exit_rsi_short = IntParameter(10, 30, default=22, space="sell", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs
        for period in [20, 30, 40, 50, 60, 80, 100]:
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # Donchian Channels (multiple periods for hyperopt)
        for period in range(10, 31):
            dataframe[f"dc_upper_{period}"] = dataframe["high"].rolling(period).max()
            dataframe[f"dc_lower_{period}"] = dataframe["low"].rolling(period).min()
            dataframe[f"dc_mid_{period}"] = (dataframe[f"dc_upper_{period}"] + dataframe[f"dc_lower_{period}"]) / 2

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_sma"] = dataframe["atr"].rolling(20).mean()
        dataframe["atr_ratio"] = dataframe["atr"] / dataframe["atr_sma"]

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["vol_sma"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dc_period = self.donchian_period.value
        dc_upper = f"dc_upper_{dc_period}"
        dc_lower = f"dc_lower_{dc_period}"
        ema = f"ema_{self.ema_period.value}"

        adx_ok = dataframe["adx"] > self.adx_min.value

        # Volatility expansion: ATR above average * multiplier
        vol_expanding = dataframe["atr_ratio"] > self.atr_mult.value

        # Volume confirmation
        vol_ok = dataframe["vol_ratio"] > 1.2

        # Trend filter
        above_ema = dataframe["close"] > dataframe[ema]
        below_ema = dataframe["close"] < dataframe[ema]

        cooldown = self.cooldown_candles.value

        # LONG: breakout above Donchian upper + expanding vol + trend
        breakout_up = (dataframe["close"] > dataframe[dc_upper].shift(1)) & above_ema
        long_signal = breakout_up & vol_expanding & adx_ok & vol_ok
        long_cooled = long_signal & ~long_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[long_cooled, "enter_long"] = 1

        # SHORT: breakdown below Donchian lower + expanding vol + trend
        breakout_down = (dataframe["close"] < dataframe[dc_lower].shift(1)) & below_ema
        short_signal = breakout_down & vol_expanding & adx_ok & vol_ok
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
