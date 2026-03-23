# AdaptiveETH — Freqtrade Algorithmic Trading Bot

Trend-following ETH/USDT strategy optimized via Bayesian hyperparameter optimization (Hyperopt/Optuna).

## Backtest Results (1 Year — Mar 2025 to Mar 2026)

| Metric | Value |
|--------|-------|
| Total Return | **+41.81%** |
| Total Trades | 191 |
| Win Rate | 53.4% |
| Max Drawdown | 5.94% |
| Sharpe Ratio | 1.77 |
| Sortino Ratio | 8.73 |
| Calmar Ratio | 36.97 |
| Profit Factor | 1.67 |
| Avg Trade Duration | 2h 47m |

*Results include Bybit's 0.06% trading fee on every entry/exit.*

## Strategy

- **Timeframe:** 15m
- **Pair:** ETH/USDT (Bybit Futures, Isolated Margin)
- **Leverage:** 3x
- **Entry signals:** EMA crossover + RSI pullback + MACD momentum + ADX trend strength + Volume confirmation
- **Exit:** ROI targets only (no exit signals — they were proven to reduce profits)
- **Risk:** -5.8% stop-loss + trailing stop

## Setup

```bash
# Install Freqtrade
pip install freqtrade

# Copy config
cp config.example.json config.json
# Edit config.json with your Bybit API keys

# Download data
freqtrade download-data --exchange bybit --pairs ETH/USDT:USDT --timeframes 15m --days 400 --trading-mode futures

# Backtest
freqtrade backtesting --strategy AdaptiveETHv6 --config config.json --timerange 20250323-20260323

# Run hyperopt (optimize parameters)
freqtrade hyperopt --strategy AdaptiveETHv6 --config config.json --hyperopt-loss ProfitDrawDownHyperOptLoss --spaces buy sell roi stoploss trailing --epochs 2000 -j 8 --min-trades 50

# Dry run (paper trading)
freqtrade trade --strategy AdaptiveETHv6 --config config.json

# FreqUI dashboard available at http://localhost:8082
```

## Strategy Versions

- **v4** (`AdaptiveETH.py`) — 1h timeframe, 7 trades, +5.09% (too conservative)
- **v5** (`AdaptiveETHv5.py`) — 15m timeframe, 193 trades, +35.80% (exit signals losing money)
- **v6** (`AdaptiveETHv6.py`) — 15m, 191 trades, **+41.81%** (exit signals disabled, ROI-only exits)

## Tech Stack

- Freqtrade 2026.2
- Python 3.12
- Bybit API (CCXT)
- TA-Lib
- Hyperopt (Optuna)
- FreqUI Dashboard
