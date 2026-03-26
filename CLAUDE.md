# Freqtrade Bot — Claude Code Instructions

## Project Overview
Algorithmic crypto trading bot project running three strategies on Bybit Futures via Freqtrade.

### Bots
| Bot | Strategy | Pair | Timeframe | Port | Credentials | Service |
|-----|----------|------|-----------|------|-------------|---------|
| ETH | AdaptiveETHv6 | ETH/USDT:USDT | 15m | 8082 | freqtrade / freqtrade123 | freqtrade-bot |
| SOL | QuantSOL | SOL/USDT:USDT | 5m | 8083 | quantsol / quantsol2024 | freqtrade-sol |
| BTC | NewsPulseBTC | BTC/USDT:USDT | 5m | 8084 | newspulse / newspulse2026 | freqtrade-btc-news |

### Key Paths
- **Strategies:** `user_data/strategies/`
- **News collector:** `user_data/strategies/news_collector.py` (daemon, systemd: news-collector)
- **Sentiment data:** `user_data/sentiment_data.json` (updated every 2 min by collector)
- **Configs:** `config.json` (ETH), `config_sol.json` (SOL), `config_btc_news.json` (BTC) — NEVER commit these (contain secrets)
- **Hyperopt results:** `user_data/hyperopt_results/`
- **Custom losses:** `user_data/hyperopts/`
- **Venv:** `venv/`
- **Logs:** `freqtrade_trade.log` (ETH), `freqtrade_sol.log` (SOL), `freqtrade_btc_news.log` (BTC), `news_collector.log`

### Strategies

**AdaptiveETHv6** (ETH):
- 6-layer entry: EMA 10/56 crossover + RSI 43-55 + ADX > 29 + MACD + BB + volume 0.9x
- 2-year backtest: +249.5% return, 589 trades, Sharpe 3.06
- Exit: OR-logic (trend reversal | RSI overbought)

**QuantSOL** (SOL):
- RSI mean-reversion within EMA 11/30 trend, MACD confirmation, 15-candle cooldown
- Custom DailyTradeHyperOptLoss for ~1 trade/day frequency targeting
- 2-year backtest: +106.18%, 774 trades, Sharpe 1.03, 38.9% CAGR

**NewsPulseBTC** (BTC) — News/Sentiment-Driven:
- Two-component system: news_collector daemon + Freqtrade strategy
- Sources: RSS feeds (CoinDesk, CoinTelegraph, Decrypt, TheBlock), Fear & Greed Index, CoinGecko market data, Twitter/X (optional)
- Keyword-weighted headline scoring (bullish/bearish dictionaries, 90-min time decay)
- Composite signal: news 50% + fear/greed 25% + market momentum 25%
- Entry: sentiment threshold ± trend confirmation (EMA) + volume + ATR volatility filter
- Strong sentiment overrides trend filter (news > chart)
- Exit: sentiment reversal + RSI extreme + 4h max hold (news becomes stale)
- Tighter risk: -12% stoploss, 4% trailing (news trades are fast)

### Common Commands
```bash
# Activate venv
source venv/bin/activate

# Backtest
freqtrade backtesting --strategy QuantSOL --config config_sol.json --timerange 20240101-20260315 --timeframe 5m
freqtrade backtesting --strategy AdaptiveETHv6 --config config.json --timerange 20240101-20260315 --timeframe 15m

# Hyperopt
freqtrade hyperopt --strategy QuantSOL --config config_sol.json --hyperopt-loss DailyTradeHyperOptLoss --spaces buy sell roi stoploss trailing --epochs 800 -j 4

# Restart services
echo 'Laracolletta123!' | sudo -S systemctl restart freqtrade-bot       # ETH
echo 'Laracolletta123!' | sudo -S systemctl restart freqtrade-sol       # SOL
echo 'Laracolletta123!' | sudo -S systemctl restart freqtrade-btc-news  # BTC
echo 'Laracolletta123!' | sudo -S systemctl restart news-collector      # Sentiment daemon

# Check status
curl -s -u freqtrade:freqtrade123 http://127.0.0.1:8082/api/v1/status
curl -s -u quantsol:quantsol2024 http://127.0.0.1:8083/api/v1/status
curl -s -u newspulse:newspulse2026 http://127.0.0.1:8084/api/v1/status

# Check sentiment data
cat user_data/sentiment_data.json | python3 -m json.tool
```

### Showcase Pages
- Landing: `/var/www/showcase/index.html` (served at :8090)
- ETH page: `/var/www/showcase/eth.html`
- SOL page: `/var/www/showcase/sol.html`
- Status JSON: `/var/www/showcase/sol_status.json`

## Rules
1. **Working directory:** `/home/giuseppe/freqtrade-bot/`
2. **Never commit** `config.json` or `config_sol.json` (contain API keys and tokens)
3. **Always activate venv** before running freqtrade commands: `source venv/bin/activate`
4. **Restart services** after strategy changes using systemd
5. **Sudo password:** `Laracolletta123!` — use `echo 'Laracolletta123!' | sudo -S <cmd>`
6. **Read before editing** — always check current file state
7. **Push to GitHub** after significant changes
8. **Don't break running bots** — test changes via backtest before deploying
9. **Use DailyTradeHyperOptLoss** for QuantSOL hyperopt (not SharpeHyperOptLoss)
10. You have full permission to read, write, edit files, run bash commands, and manage services in this project without asking for confirmation.
