"""
News Sentiment Collector — Multi-Source Crypto Intelligence Engine

Runs as a standalone daemon. Aggregates sentiment from:
1. CryptoPanic API (curated crypto news + community votes)
2. Alternative.me Fear & Greed Index
3. CoinGecko trending + market data
4. RSS feeds (CoinDesk, CoinTelegraph, Decrypt)
5. Twitter/X API (optional, requires TWITTER_BEARER_TOKEN)

Writes aggregated sentiment to sentiment_data.json every cycle.
The NewsPulseBTC strategy reads this file in populate_indicators().

Usage:
    python user_data/strategies/news_collector.py [--once]
"""

import json
import time
import hashlib
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

# ── Config ──────────────────────────────────────────────────────────────────
SENTIMENT_FILE = Path(__file__).parent.parent / "sentiment_data.json"
POLL_INTERVAL = 120  # seconds between full cycles
REQUEST_TIMEOUT = 15

# API keys (optional — set as env vars)
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path(__file__).parent.parent.parent / "news_collector.log"
        ),
    ],
)
log = logging.getLogger("news_collector")

# ── Keyword scoring dictionaries ────────────────────────────────────────────
# Professional-grade keyword lists used by quant desks for headline scoring.
# Weighted by market-impact severity.

BULLISH_KEYWORDS = {
    # High impact (+3)
    "etf approved": 3, "etf approval": 3, "spot etf": 3,
    "institutional adoption": 3, "strategic reserve": 3,
    "legal tender": 3, "rate cut": 3, "fed pivot": 3,
    # Medium impact (+2)
    "bullish": 2, "surge": 2, "soar": 2, "rally": 2, "breakout": 2,
    "all-time high": 2, "ath": 2, "moon": 2, "pump": 2,
    "accumulation": 2, "buying": 2, "inflow": 2, "adoption": 2,
    "partnership": 2, "upgrade": 2, "halving": 2,
    "whale buying": 2, "short squeeze": 2,
    # Low impact (+1)
    "rise": 1, "gain": 1, "up": 1, "positive": 1, "growth": 1,
    "recovery": 1, "support": 1, "demand": 1, "accumulate": 1,
    "buy": 1, "launch": 1, "integration": 1, "expansion": 1,
}

BEARISH_KEYWORDS = {
    # High impact (-3)
    "ban": 3, "banned": 3, "sec lawsuit": 3, "indictment": 3,
    "hack": 3, "hacked": 3, "exploit": 3, "rug pull": 3,
    "bankruptcy": 3, "insolvent": 3, "collapse": 3,
    "rate hike": 3, "fed hawkish": 3,
    # Medium impact (-2)
    "bearish": 2, "crash": 2, "dump": 2, "plunge": 2, "selloff": 2,
    "sell-off": 2, "liquidation": 2, "outflow": 2, "withdraw": 2,
    "regulation": 2, "crackdown": 2, "investigation": 2,
    "whale selling": 2, "delisting": 2, "vulnerability": 2,
    # Low impact (-1)
    "fall": 1, "drop": 1, "decline": 1, "down": 1, "negative": 1,
    "risk": 1, "concern": 1, "fear": 1, "uncertainty": 1,
    "resistance": 1, "sell": 1, "warning": 1, "delay": 1,
}

# Coins to track sentiment for (BTC-focused but context from alts)
TRACKED_COINS = {"BTC", "bitcoin", "btc", "Bitcoin", "crypto", "cryptocurrency"}


def score_headline(title: str) -> int:
    """Score a headline from -10 to +10 based on keyword matching."""
    title_lower = title.lower()
    score = 0
    for keyword, weight in BULLISH_KEYWORDS.items():
        if keyword in title_lower:
            score += weight
    for keyword, weight in BEARISH_KEYWORDS.items():
        if keyword in title_lower:
            score -= weight
    return max(-10, min(10, score))


def is_btc_relevant(title: str) -> bool:
    """Check if headline is relevant to BTC/crypto market."""
    title_lower = title.lower()
    # Direct BTC mentions
    if any(coin.lower() in title_lower for coin in TRACKED_COINS):
        return True
    # Macro news that moves BTC
    macro_terms = ["fed ", "federal reserve", "interest rate", "inflation",
                   "etf", "sec ", "regulation", "institutional", "treasury"]
    if any(term in title_lower for term in macro_terms):
        return True
    return False


def time_decay_weight(published_at: datetime) -> float:
    """
    News impact decays exponentially. Professional desks use 2-4 hour half-life.
    We use 90-minute half-life for crypto (faster market).
    Returns weight between 0.0 and 1.0.
    """
    now = datetime.now(timezone.utc)
    age_minutes = (now - published_at).total_seconds() / 60
    if age_minutes < 0:
        age_minutes = 0
    half_life = 90  # minutes
    return 2 ** (-age_minutes / half_life)


# ── Source: CryptoPanic ─────────────────────────────────────────────────────
def fetch_cryptopanic() -> list[dict]:
    """
    CryptoPanic: curated crypto news aggregator.
    Free tier: 50 req/hour. Returns news with community sentiment votes.
    """
    items = []
    try:
        params = {
            "currencies": "BTC",
            "kind": "news",
            "filter": "important",
            "public": "true",
        }
        if CRYPTOPANIC_API_KEY:
            params["auth_token"] = CRYPTOPANIC_API_KEY

        url = "https://cryptopanic.com/api/free/v1/posts/"
        if CRYPTOPANIC_API_KEY:
            url = "https://cryptopanic.com/api/v1/posts/"

        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            log.warning(f"CryptoPanic HTTP {resp.status_code}")
            return items

        data = resp.json()
        for post in data.get("results", [])[:20]:
            title = post.get("title", "")
            published = post.get("published_at", "")

            # Parse timestamp
            try:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pub_dt = datetime.now(timezone.utc)

            # CryptoPanic has its own sentiment votes
            votes = post.get("votes", {})
            cp_sentiment = (
                votes.get("positive", 0) - votes.get("negative", 0)
                + votes.get("important", 0)
            )

            headline_score = score_headline(title)
            combined = headline_score + min(3, max(-3, cp_sentiment))
            weight = time_decay_weight(pub_dt)

            items.append({
                "source": "cryptopanic",
                "title": title,
                "score": combined,
                "weight": weight,
                "weighted_score": combined * weight,
                "published_at": pub_dt.isoformat(),
                "url": post.get("url", ""),
            })

        log.info(f"CryptoPanic: {len(items)} articles")
    except Exception as e:
        log.error(f"CryptoPanic error: {e}")

    return items


# ── Source: Fear & Greed Index ──────────────────────────────────────────────
def fetch_fear_greed() -> dict:
    """
    Alternative.me Fear & Greed Index (0-100).
    0 = Extreme Fear, 100 = Extreme Greed.
    Updated every ~15 minutes. Free, no API key.
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=2",
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return {"value": 50, "label": "Neutral", "normalized": 0.0}

        data = resp.json().get("data", [{}])[0]
        value = int(data.get("value", 50))
        label = data.get("value_classification", "Neutral")

        # Normalize to -1.0 (extreme fear) to +1.0 (extreme greed)
        normalized = (value - 50) / 50.0

        # Also get previous day for momentum
        prev = resp.json().get("data", [None, {}])
        prev_value = int(prev[1].get("value", value)) if len(prev) > 1 else value
        momentum = value - prev_value  # positive = getting greedier

        log.info(f"Fear & Greed: {value} ({label}), momentum: {momentum:+d}")
        return {
            "value": value,
            "label": label,
            "normalized": normalized,
            "momentum": momentum,
        }
    except Exception as e:
        log.error(f"Fear & Greed error: {e}")
        return {"value": 50, "label": "Neutral", "normalized": 0.0, "momentum": 0}


# ── Source: CoinGecko ───────────────────────────────────────────────────────
def fetch_coingecko_btc() -> dict:
    """
    CoinGecko: BTC market data (price change %, volume, market cap).
    Free tier: 10-30 req/min. No API key for basic endpoints.
    """
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin",
            params={"localization": "false", "tickers": "false",
                    "community_data": "false", "developer_data": "false"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return {}

        data = resp.json()
        market = data.get("market_data", {})

        return {
            "price_change_1h": market.get("price_change_percentage_1h_in_currency", {}).get("usd", 0),
            "price_change_24h": market.get("price_change_percentage_24h", 0),
            "price_change_7d": market.get("price_change_percentage_7d", 0),
            "total_volume_24h": market.get("total_volume", {}).get("usd", 0),
            "market_cap_change_24h": market.get("market_cap_change_percentage_24h", 0),
        }
    except Exception as e:
        log.error(f"CoinGecko error: {e}")
        return {}


# ── Source: RSS Feeds ───────────────────────────────────────────────────────
RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "theblock": "https://www.theblock.co/rss.xml",
}


def fetch_rss_feeds() -> list[dict]:
    """Parse RSS feeds from major crypto news outlets."""
    items = []
    for source, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
                "User-Agent": "NewsPulseBTC/1.0 (crypto trading research)"
            })
            if resp.status_code != 200:
                log.warning(f"RSS {source}: HTTP {resp.status_code}")
                continue

            root = ET.fromstring(resp.content)

            # Handle both RSS 2.0 and Atom
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            rss_items = root.findall(".//item")
            if not rss_items:
                rss_items = root.findall(".//atom:entry", ns)

            for item in rss_items[:15]:
                title = ""
                pub_date = ""

                # RSS 2.0
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    title = title_el.text.strip()

                pubdate_el = item.find("pubDate")
                if pubdate_el is not None and pubdate_el.text:
                    pub_date = pubdate_el.text

                # Atom fallback
                if not title:
                    title_el = item.find("atom:title", ns)
                    if title_el is not None and title_el.text:
                        title = title_el.text.strip()

                if not pub_date:
                    updated_el = item.find("atom:updated", ns)
                    if updated_el is not None and updated_el.text:
                        pub_date = updated_el.text

                if not title:
                    continue

                # Skip non-BTC-relevant headlines
                if not is_btc_relevant(title):
                    continue

                # Parse date
                try:
                    pub_dt = _parse_rss_date(pub_date)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

                # Skip articles older than 6 hours
                age = (datetime.now(timezone.utc) - pub_dt).total_seconds()
                if age > 6 * 3600:
                    continue

                headline_score = score_headline(title)
                weight = time_decay_weight(pub_dt)

                items.append({
                    "source": f"rss_{source}",
                    "title": title,
                    "score": headline_score,
                    "weight": weight,
                    "weighted_score": headline_score * weight,
                    "published_at": pub_dt.isoformat(),
                })

        except Exception as e:
            log.warning(f"RSS {source} error: {e}")

    log.info(f"RSS feeds: {len(items)} relevant articles")
    return items


def _parse_rss_date(date_str: str) -> datetime:
    """Parse various RSS date formats."""
    if not date_str:
        return datetime.now(timezone.utc)

    # ISO 8601
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    # RFC 2822 (common in RSS)
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass

    return datetime.now(timezone.utc)


# ── Source: Twitter/X (Optional) ────────────────────────────────────────────
def fetch_twitter() -> list[dict]:
    """
    Twitter/X API v2 — search recent tweets from crypto influencers.
    Requires TWITTER_BEARER_TOKEN env var ($100/mo Basic plan).
    Returns empty list if no token configured.
    """
    if not TWITTER_BEARER_TOKEN:
        return []

    items = []
    try:
        # Search for high-engagement BTC tweets from last hour
        query = "(bitcoin OR BTC OR crypto) (crash OR surge OR breaking OR etf OR hack OR ban OR rally) -is:retweet lang:en"
        params = {
            "query": query,
            "max_results": 20,
            "sort_order": "relevancy",
            "tweet.fields": "created_at,public_metrics,author_id",
        }
        headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

        resp = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            log.warning(f"Twitter HTTP {resp.status_code}: {resp.text[:200]}")
            return items

        data = resp.json()
        for tweet in data.get("data", []):
            text = tweet.get("text", "")
            created = tweet.get("created_at", "")
            metrics = tweet.get("public_metrics", {})

            try:
                pub_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pub_dt = datetime.now(timezone.utc)

            # Weight by engagement (likes + retweets = amplification)
            engagement = metrics.get("like_count", 0) + metrics.get("retweet_count", 0) * 3
            engagement_boost = min(2.0, 1.0 + engagement / 1000)

            headline_score = score_headline(text)
            weight = time_decay_weight(pub_dt) * engagement_boost

            items.append({
                "source": "twitter",
                "title": text[:200],
                "score": headline_score,
                "weight": weight,
                "weighted_score": headline_score * weight,
                "published_at": pub_dt.isoformat(),
            })

        log.info(f"Twitter: {len(items)} tweets")
    except Exception as e:
        log.error(f"Twitter error: {e}")

    return items


# ── Aggregation Engine ──────────────────────────────────────────────────────
def compute_aggregate_sentiment(news_items: list[dict], fear_greed: dict,
                                 coingecko: dict) -> dict:
    """
    Combine all sources into a single sentiment signal.

    Output scores:
    - composite_score: -100 to +100 (primary trading signal)
    - news_score: -100 to +100 (headline sentiment only)
    - fear_greed_score: -100 to +100 (market sentiment gauge)
    - momentum_score: -100 to +100 (rate of sentiment change)
    - signal: "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell"
    - confidence: 0.0 to 1.0 (how many sources agree)
    """

    # 1. News headline aggregate (time-weighted)
    if news_items:
        total_weight = sum(abs(n["weight"]) for n in news_items)
        if total_weight > 0:
            raw_news = sum(n["weighted_score"] for n in news_items) / total_weight
        else:
            raw_news = 0
        # Scale to -100..+100
        news_score = max(-100, min(100, raw_news * 15))
    else:
        news_score = 0

    # 2. Fear & Greed contribution
    fg_normalized = fear_greed.get("normalized", 0)  # -1 to +1
    fg_momentum = fear_greed.get("momentum", 0)  # -100 to +100
    fear_greed_score = fg_normalized * 60 + fg_momentum * 0.4

    # 3. CoinGecko market momentum
    cg_1h = coingecko.get("price_change_1h", 0)
    cg_24h = coingecko.get("price_change_24h", 0)
    market_momentum = cg_1h * 8 + cg_24h * 2  # 1h weighted more heavily

    # 4. Composite: weighted blend
    #    News (50%) + Fear/Greed (25%) + Market momentum (25%)
    composite = (
        news_score * 0.50
        + fear_greed_score * 0.25
        + market_momentum * 0.25
    )
    composite = max(-100, min(100, composite))

    # 5. Signal classification (institutional-style thresholds)
    if composite >= 40:
        signal = "strong_buy"
    elif composite >= 15:
        signal = "buy"
    elif composite <= -40:
        signal = "strong_sell"
    elif composite <= -15:
        signal = "sell"
    else:
        signal = "neutral"

    # 6. Confidence: agreement across sources
    signs = []
    if news_score != 0:
        signs.append(1 if news_score > 0 else -1)
    if fear_greed_score != 0:
        signs.append(1 if fear_greed_score > 0 else -1)
    if market_momentum != 0:
        signs.append(1 if market_momentum > 0 else -1)

    if signs:
        agreement = abs(sum(signs)) / len(signs)
    else:
        agreement = 0
    confidence = agreement

    # 7. Count active sources
    source_counts = {}
    for item in news_items:
        src = item["source"].split("_")[0]
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "composite_score": round(composite, 2),
        "news_score": round(news_score, 2),
        "fear_greed_score": round(fear_greed_score, 2),
        "market_momentum_score": round(market_momentum, 2),
        "signal": signal,
        "confidence": round(confidence, 2),
        "fear_greed_value": fear_greed.get("value", 50),
        "fear_greed_label": fear_greed.get("label", "Neutral"),
        "source_counts": source_counts,
        "total_articles": len(news_items),
    }


# ── Main Loop ───────────────────────────────────────────────────────────────
def collect_once() -> dict:
    """Run one collection cycle and return the full sentiment payload."""
    log.info("=" * 60)
    log.info("Starting collection cycle")

    # Fetch all sources
    news_items = []
    news_items.extend(fetch_cryptopanic())
    news_items.extend(fetch_rss_feeds())
    news_items.extend(fetch_twitter())

    fear_greed = fetch_fear_greed()
    coingecko = fetch_coingecko_btc()

    # Aggregate
    sentiment = compute_aggregate_sentiment(news_items, fear_greed, coingecko)

    # Build output
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentiment": sentiment,
        "top_headlines": sorted(
            news_items, key=lambda x: abs(x["weighted_score"]), reverse=True
        )[:10],
        "fear_greed": fear_greed,
        "coingecko": coingecko,
    }

    # Write atomically (write to temp, then rename)
    tmp_file = SENTIMENT_FILE.with_suffix(".tmp")
    with open(tmp_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    tmp_file.rename(SENTIMENT_FILE)

    log.info(
        f"Composite: {sentiment['composite_score']:+.1f} | "
        f"Signal: {sentiment['signal']} | "
        f"Confidence: {sentiment['confidence']:.0%} | "
        f"Sources: {sentiment['total_articles']} articles"
    )
    log.info(f"Written to {SENTIMENT_FILE}")

    return output


def main():
    """Run collector as daemon or single shot."""
    log.info("News Sentiment Collector starting")
    log.info(f"Output: {SENTIMENT_FILE}")
    log.info(f"CryptoPanic API: {'configured' if CRYPTOPANIC_API_KEY else 'free tier (limited)'}")
    log.info(f"Twitter API: {'configured' if TWITTER_BEARER_TOKEN else 'not configured (optional)'}")

    if "--once" in sys.argv:
        result = collect_once()
        print(json.dumps(result, indent=2, default=str))
        return

    while True:
        try:
            collect_once()
        except Exception as e:
            log.error(f"Collection cycle failed: {e}")

        log.info(f"Sleeping {POLL_INTERVAL}s until next cycle...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
