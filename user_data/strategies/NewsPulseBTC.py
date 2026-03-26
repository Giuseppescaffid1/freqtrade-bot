"""
NewsPulseBTC — News-Driven Algorithmic Trading Strategy

Trades BTC/USDT perpetual futures on sentiment signals from the news_collector
daemon, confirmed by basic technical analysis filters.

Philosophy:
    "Trade the news, confirm with the chart."

    Unlike pure TA strategies (AdaptiveETHv6, QuantSOL), this strategy uses
    external sentiment as the PRIMARY signal and price action as CONFIRMATION.
    This is how institutional event-driven desks operate — the edge comes from
    reacting to information faster than the market prices it in.

Architecture:
    news_collector.py (daemon) → sentiment_data.json → this strategy reads it

Entry Logic:
    1. Sentiment signal: composite_score crosses threshold (±15)
    2. Trend confirmation: EMA alignment (don't fight the trend)
    3. Volume confirmation: above average (confirms market is reacting)
    4. Volatility filter: ATR-based, avoid entering in dead markets
    5. Cooldown: min candles between sentiment-driven entries

Exit Logic:
    - Sentiment reversal: composite flips direction
    - RSI extreme: overbought/oversold
    - Time decay: news impact fades, exit after max hold period
    - ROI + trailing stop + stoploss (safety net)

Risk Management:
    - 3x leverage (consistent with other bots)
    - Tighter stoploss than TA bots (news trades are high-conviction, short-lived)
    - Trailing stop locks in gains from rapid moves
    - Max 2 open trades (concentration on high-conviction signals)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame

log = logging.getLogger(__name__)

SENTIMENT_FILE = Path(__file__).parent.parent / "sentiment_data.json"


class NewsPulseBTC(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True

    # ── ROI: more aggressive than TA bots (news moves are fast) ──
    minimal_roi = {
        "0": 0.15,      # 15% immediate (strong news = strong move)
        "15": 0.08,      # 8% after 15 min
        "45": 0.04,      # 4% after 45 min
        "120": 0.015,    # 1.5% after 2h
        "240": 0,        # close at breakeven after 4h (news is stale)
    }

    # ── Tighter stoploss: news trades shouldn't sit in drawdown ──
    stoploss = -0.12

    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.08
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    # ── Sentiment thresholds ──
    sentiment_buy_threshold = IntParameter(
        10, 50, default=20, space="buy", optimize=True, load=True
    )
    sentiment_sell_threshold = IntParameter(
        10, 50, default=20, space="sell", optimize=True, load=True
    )
    sentiment_strong_threshold = IntParameter(
        30, 70, default=45, space="buy", optimize=True, load=True
    )

    # ── Trend filter (lighter than pure TA bots) ──
    ema_fast = IntParameter(8, 21, default=12, space="buy", optimize=True, load=True)
    ema_slow = IntParameter(21, 55, default=34, space="buy", optimize=True, load=True)
    use_trend_filter = IntParameter(
        0, 1, default=1, space="buy", optimize=True, load=True
    )

    # ── Volume confirmation ──
    volume_factor = DecimalParameter(
        0.8, 2.0, default=1.1, space="buy", optimize=True, load=True
    )

    # ── Cooldown: min candles between news-driven entries ──
    cooldown_candles = IntParameter(
        6, 40, default=12, space="buy", optimize=True, load=True
    )

    # ── Exit ──
    exit_rsi_long = IntParameter(
        70, 90, default=80, space="sell", optimize=True, load=True
    )
    exit_rsi_short = IntParameter(
        10, 30, default=20, space="sell", optimize=True, load=True
    )

    # ── FreqUI Chart Indicators ──
    plot_config = {
        "main_plot": {
            "ema_12": {"color": "#f59e0b", "type": "line"},
            "ema_34": {"color": "#3b82f6", "type": "line"},
            "bb_upper": {"color": "#6b728050", "type": "line"},
            "bb_lower": {"color": "#6b728050", "type": "line"},
        },
        "subplots": {
            "Sentiment": {
                "sentiment_score": {"color": "#a855f7", "type": "line", "fill_to": 0},
            },
            "RSI": {
                "rsi": {"color": "#f59e0b", "type": "line"},
            },
            "Volume Ratio": {
                "vol_ratio": {"color": "#10b981", "type": "bar"},
            },
        },
    }

    # ── Internal state ──
    _last_sentiment: dict = {}
    _last_sentiment_read: float = 0

    def _read_sentiment(self) -> dict:
        """
        Read sentiment data from the collector daemon's JSON output.
        Caches for 30 seconds to avoid excessive file I/O.
        """
        import time

        now = time.time()
        if now - self._last_sentiment_read < 30 and self._last_sentiment:
            return self._last_sentiment

        try:
            if SENTIMENT_FILE.exists():
                with open(SENTIMENT_FILE) as f:
                    data = json.load(f)

                # Check freshness — sentiment older than 10 min is stale
                ts = data.get("timestamp", "")
                if ts:
                    try:
                        data_time = datetime.fromisoformat(ts)
                        age = (datetime.now(timezone.utc) - data_time).total_seconds()
                        if age > 600:
                            log.warning(f"Sentiment data is {age/60:.0f}min old (stale)")
                            data["sentiment"]["stale"] = True
                    except (ValueError, TypeError):
                        pass

                self._last_sentiment = data
                self._last_sentiment_read = now
                return data
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Error reading sentiment: {e}")

        return {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ── Standard TA indicators ──
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # EMAs (range for hyperopt)
        for period in range(8, 56):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)

        # ADX for trend strength
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # ATR for volatility context
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"] * 100

        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_width"] = (bb["upperband"] - bb["lowerband"]) / bb["middleband"]

        # Volume ratio (current vs 20-period SMA)
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["vol_sma"]

        # MACD for momentum context
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        # ── Sentiment injection ──
        sentiment_data = self._read_sentiment()
        sentiment = sentiment_data.get("sentiment", {})

        composite = sentiment.get("composite_score", 0)
        stale = sentiment.get("stale", False)
        confidence = sentiment.get("confidence", 0)

        # If sentiment data is stale, dampen the signal
        if stale:
            composite *= 0.3

        # Weight by confidence (sources agreeing)
        effective_score = composite * max(0.3, confidence)

        # Inject as a constant column (updates every 30s via _read_sentiment cache)
        dataframe["sentiment_score"] = effective_score
        dataframe["sentiment_confidence"] = confidence
        dataframe["fear_greed"] = sentiment.get("fear_greed_value", 50)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = f"ema_{self.ema_fast.value}"
        ema_slow = f"ema_{self.ema_slow.value}"

        # ── Sentiment signals ──
        sent_bullish = dataframe["sentiment_score"] >= self.sentiment_buy_threshold.value
        sent_bearish = dataframe["sentiment_score"] <= -self.sentiment_sell_threshold.value
        sent_strong_bull = dataframe["sentiment_score"] >= self.sentiment_strong_threshold.value
        sent_strong_bear = dataframe["sentiment_score"] <= -self.sentiment_strong_threshold.value

        # ── Trend confirmation (optional via hyperopt) ──
        uptrend = dataframe[ema_fast] > dataframe[ema_slow]
        downtrend = dataframe[ema_fast] < dataframe[ema_slow]

        if self.use_trend_filter.value == 1:
            trend_long = uptrend
            trend_short = downtrend
        else:
            trend_long = True
            trend_short = True

        # ── Volume confirmation ──
        vol_ok = dataframe["vol_ratio"] > self.volume_factor.value

        # ── Volatility filter: market must be alive (ATR > 0.1%) ──
        vol_alive = dataframe["atr_pct"] > 0.1

        # ── Cooldown ──
        cooldown = self.cooldown_candles.value

        # ── LONG entries ──
        # Standard: sentiment bullish + trend + volume
        long_standard = sent_bullish & trend_long & vol_ok & vol_alive
        # Strong: strong sentiment overrides trend filter (news > chart)
        long_strong = sent_strong_bull & vol_alive

        long_signal = long_standard | long_strong
        long_cooled = long_signal & ~long_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)

        dataframe.loc[long_cooled, "enter_long"] = 1
        dataframe.loc[long_cooled & sent_strong_bull, "enter_tag"] = "strong_news_long"
        dataframe.loc[long_cooled & ~sent_strong_bull, "enter_tag"] = "news_long"

        # ── SHORT entries ──
        short_standard = sent_bearish & trend_short & vol_ok & vol_alive
        short_strong = sent_strong_bear & vol_alive

        short_signal = short_standard | short_strong
        short_cooled = short_signal & ~short_signal.shift(1).rolling(cooldown).max().fillna(0).astype(bool)

        dataframe.loc[short_cooled, "enter_short"] = 1
        dataframe.loc[short_cooled & sent_strong_bear, "enter_tag"] = "strong_news_short"
        dataframe.loc[short_cooled & ~sent_strong_bear, "enter_tag"] = "news_short"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long: RSI overbought OR sentiment flips negative
        dataframe.loc[
            (dataframe["rsi"] > self.exit_rsi_long.value)
            | (dataframe["sentiment_score"] < -10),
            "exit_long",
        ] = 1

        # Exit short: RSI oversold OR sentiment flips positive
        dataframe.loc[
            (dataframe["rsi"] < self.exit_rsi_short.value)
            | (dataframe["sentiment_score"] > 10),
            "exit_short",
        ] = 1

        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        return 3.0
