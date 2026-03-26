"""
SOL_MFI_VWAP — Money Flow Index + VWAP Scalper

MFI is a volume-weighted RSI — better for crypto because it factors in
volume conviction behind price moves. Combined with VWAP (Volume Weighted
Average Price) as dynamic support/resistance.
Targets ~1 trade/day on 15m timeframe.
"""
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class SOL_MFI_VWAP(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

    minimal_roi = {
        "0": 0.14,
        "30": 0.07,
        "90": 0.03,
        "180": 0.01,
        "300": 0,
    }

    stoploss = -0.13
    trailing_stop = True
    trailing_stop_positive = 0.045
    trailing_stop_positive_offset = 0.09
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # Trend
    ema_fast = IntParameter(8, 21, default=10, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(21, 55, default=30, space="buy", optimize=True, load=True)

    # MFI
    mfi_buy = IntParameter(10, 35, default=25, space="buy", optimize=True, load=True)
    mfi_sell = IntParameter(65, 90, default=75, space="sell", optimize=True, load=True)

    # VWAP deviation (% above/below VWAP)
    vwap_buy_dev = DecimalParameter(-3.0, -0.5, default=-1.5, space="buy", optimize=True, load=True)
    vwap_sell_dev = DecimalParameter(0.5, 3.0, default=1.5, space="sell", optimize=True, load=True)

    # ADX
    adx_min = IntParameter(12, 28, default=18, space="buy", optimize=True, load=True)

    # Cooldown
    cooldown_candles = IntParameter(10, 60, default=30, space="buy", optimize=True, load=True)

    # Exit
    exit_mfi_long = IntParameter(70, 95, default=82, space="sell", optimize=True, load=True)
    exit_mfi_short = IntParameter(5, 30, default=18, space="sell", optimize=True, load=True)

    def _calc_vwap(self, dataframe: DataFrame) -> DataFrame:
        """Calculate rolling VWAP (96 candles = 24h on 15m)."""
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        vol_tp = typical_price * dataframe["volume"]
        # Rolling 24h VWAP
        window = 96
        dataframe["vwap"] = vol_tp.rolling(window).sum() / dataframe["volume"].rolling(window).sum()
        dataframe["vwap_dev"] = ((dataframe["close"] - dataframe["vwap"]) / dataframe["vwap"]) * 100
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for period in range(8, 56):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # MFI
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)

        # RSI (backup)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # VWAP
        dataframe = self._calc_vwap(dataframe)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["vol_sma"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        uptrend = dataframe[ema_fast] > dataframe[ema_slow]
        downtrend = dataframe[ema_fast] < dataframe[ema_slow]
        adx_ok = dataframe["adx"] > self.adx_min.value

        # MFI oversold + price below VWAP = institutional buying zone
        mfi_buy = dataframe["mfi"] < self.mfi_buy.value
        vwap_buy = dataframe["vwap_dev"] < self.vwap_buy_dev.value

        mfi_sell = dataframe["mfi"] > self.mfi_sell.value
        vwap_sell = dataframe["vwap_dev"] > self.vwap_sell_dev.value

        # Volume confirmation
        vol_ok = dataframe["vol_ratio"] > 0.9

        cooldown = self.cooldown_candles.value

        # LONG
        long_signal = uptrend & mfi_buy & vwap_buy & adx_ok & vol_ok
        long_cooled = long_signal & ~long_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[long_cooled, "enter_long"] = 1

        # SHORT
        short_signal = downtrend & mfi_sell & vwap_sell & adx_ok & vol_ok
        short_cooled = short_signal & ~short_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)
        dataframe.loc[short_cooled, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["mfi"] > self.exit_mfi_long.value, "exit_long"] = 1
        dataframe.loc[dataframe["mfi"] < self.exit_mfi_short.value, "exit_short"] = 1
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
