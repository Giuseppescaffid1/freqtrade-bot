"""
Adaptive ETH Strategy v7 — Regime-Aware Trading

Detects market regime using ADX + BB width and adapts:
- TRENDING (ADX > threshold): trend-following with EMA crossover
- RANGING (ADX < threshold): mean-reversion with BB bounces
- Dynamic stoploss: tighter in ranging, wider in trending

This makes the strategy profitable across both 2024 (sideways)
and 2025 (trending) market conditions.
"""

import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame


class AdaptiveETHv7(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

    minimal_roi = {
        "0": 0.15,
        "60": 0.06,
        "180": 0.03,
        "360": 0,
    }

    stoploss = -0.15
    trailing_stop = True
    trailing_stop_positive = 0.08
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True
    use_custom_stoploss = True

    # ── Trend Parameters ──
    ema_fast = IntParameter(5, 25, default=10, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(26, 80, default=50, space="buy", optimize=True, load=True)

    # ── Entry Filters ──
    rsi_buy_low = IntParameter(20, 45, default=35, space="buy", optimize=True, load=True)
    rsi_buy_high = IntParameter(55, 75, default=65, space="buy", optimize=True, load=True)
    rsi_sell_low = IntParameter(25, 45, default=35, space="sell", optimize=True, load=True)
    rsi_sell_high = IntParameter(55, 80, default=65, space="sell", optimize=True, load=True)
    volume_factor = DecimalParameter(0.5, 2.0, default=1.0, decimals=1, space="buy", optimize=True, load=True)

    # ── Regime Detection ──
    adx_trend_threshold = IntParameter(15, 35, default=25, space="buy", optimize=True, load=True)

    # ── Mean Reversion (ranging regime) ──
    bb_rsi_low = IntParameter(15, 40, default=30, space="buy", optimize=True, load=True)
    bb_rsi_high = IntParameter(60, 85, default=70, space="sell", optimize=True, load=True)

    # ── Dynamic Stoploss ──
    sl_ranging = DecimalParameter(-0.08, -0.02, default=-0.04, decimals=3, space="sell", optimize=True, load=True)
    sl_trending = DecimalParameter(-0.25, -0.08, default=-0.15, decimals=3, space="sell", optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        for period in range(5, 81):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]

        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        vol_ok = dataframe["volume"] > dataframe["vol_sma"] * self.volume_factor.value
        adx_threshold = self.adx_trend_threshold.value

        # ── TRENDING REGIME (ADX > threshold): trend following ──
        trending = dataframe["adx"] > adx_threshold

        trend_long = (
            trending
            & (dataframe[ema_fast] > dataframe[ema_slow])
            & (dataframe["rsi"] > self.rsi_buy_low.value)
            & (dataframe["rsi"] < self.rsi_buy_high.value)
            & vol_ok
        )

        trend_short = (
            trending
            & (dataframe[ema_fast] < dataframe[ema_slow])
            & (dataframe["rsi"] > self.rsi_sell_low.value)
            & (dataframe["rsi"] < self.rsi_sell_high.value)
            & vol_ok
        )

        # ── RANGING REGIME (ADX < threshold): mean reversion ──
        ranging = dataframe["adx"] <= adx_threshold

        range_long = (
            ranging
            & (dataframe["close"] <= dataframe["bb_lower"])
            & (dataframe["rsi"] < self.bb_rsi_low.value)
            & vol_ok
        )

        range_short = (
            ranging
            & (dataframe["close"] >= dataframe["bb_upper"])
            & (dataframe["rsi"] > self.bb_rsi_high.value)
            & vol_ok
        )

        dataframe.loc[trend_long | range_long, "enter_long"] = 1
        dataframe.loc[trend_short | range_short, "enter_short"] = 1

        dataframe.loc[trend_long, "enter_tag"] = "trend_long"
        dataframe.loc[range_long, "enter_tag"] = "range_long"
        dataframe.loc[trend_short, "enter_tag"] = "trend_short"
        dataframe.loc[range_short, "enter_tag"] = "range_short"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Mean reversion exits at BB mid
        dataframe.loc[
            (dataframe["adx"] <= self.adx_trend_threshold.value)
            & (dataframe["close"] > dataframe["bb_mid"]),
            "exit_long",
        ] = 1

        dataframe.loc[
            (dataframe["adx"] <= self.adx_trend_threshold.value)
            & (dataframe["close"] < dataframe["bb_mid"]),
            "exit_short",
        ] = 1

        return dataframe

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return self.stoploss

        last_adx = dataframe.iloc[-1].get("adx", 25)

        if last_adx > self.adx_trend_threshold.value:
            return self.sl_trending.value
        else:
            return self.sl_ranging.value

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
