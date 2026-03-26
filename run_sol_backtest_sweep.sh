#!/bin/bash
# SOL Strategy Backtest Sweep — runs all strategies in parallel
# Results saved to /tmp/sol_backtest_results/

set -e
cd /home/giuseppe/freqtrade-bot
source venv/bin/activate

OUTDIR="/tmp/sol_backtest_results"
mkdir -p "$OUTDIR"
TIMERANGE="20240101-20260326"

echo "Starting SOL backtest sweep at $(date)"
echo "Timerange: $TIMERANGE"
echo "Output: $OUTDIR"
echo ""

# Function to run a backtest and save output
run_bt() {
    local strategy=$1
    local timeframe=$2
    local label="${strategy}_${timeframe}"
    echo "[START] $label"
    freqtrade backtesting \
        --strategy "$strategy" \
        --config config_sol.json \
        --timerange "$TIMERANGE" \
        --timeframe "$timeframe" \
        --export none \
        2>&1 | tee "$OUTDIR/${label}.log" | tail -30 > "$OUTDIR/${label}_summary.log"
    echo "[DONE] $label"
}

# Run all backtests in parallel (max 4 at a time to not overload CPU)
run_bt QuantSOL 5m &
run_bt SOL_QuantSOL_15m 15m &
run_bt SOL_StochRSI 15m &
run_bt SOL_CCI_BB 5m &
wait

run_bt SOL_MFI_VWAP 15m &
run_bt SOL_Breakout 1h &
run_bt SOL_MACD_Divergence 15m &
run_bt SOL_EMA_Ribbon 15m &
wait

echo ""
echo "============================================"
echo "ALL BACKTESTS COMPLETE at $(date)"
echo "============================================"
echo ""

# Summary extraction
for f in "$OUTDIR"/*_summary.log; do
    label=$(basename "$f" _summary.log)
    echo "=== $label ==="
    grep -E "TOTAL|Total|Profit|Drawdown|trades|Win|Sharpe" "$f" 2>/dev/null || echo "(no summary found)"
    echo ""
done
