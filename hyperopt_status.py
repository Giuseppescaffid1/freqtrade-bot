#!/usr/bin/env python3
"""Writes hyperopt progress to a JSON file for the dashboard."""
import json, time, glob, os, sys

RESULTS_DIR = "/home/giuseppe/freqtrade-bot/user_data/hyperopt_results"
OUTPUT = "/var/www/showcase/hyperopt_status.json"
TARGET_EPOCHS = 1000

while True:
    try:
        # Find latest v6 hyperopt file
        files = sorted(glob.glob(f"{RESULTS_DIR}/strategy_AdaptiveETHv6*.fthypt"), key=os.path.getmtime, reverse=True)
        if not files:
            status = {"running": False, "epochs": 0, "target": TARGET_EPOCHS, "pct": 0, "best_profit": 0, "best_trades": 0}
        else:
            fpath = files[0]
            # Count lines = epochs
            with open(fpath, 'r') as f:
                lines = f.readlines()
            epochs = len(lines)

            # Parse best result from lines
            best_profit = 0
            best_trades = 0
            best_pct = "0%"
            for line in lines:
                try:
                    entry = json.loads(line)
                    if entry.get('is_best', False):
                        results = entry.get('results_metrics', {})
                        profit = results.get('profit_total', 0) * 100
                        trades = results.get('trade_count', len(results.get('trades', [])))
                        if profit > best_profit:
                            best_profit = round(profit, 2)
                            best_trades = trades
                            best_pct = f"+{best_profit:.2f}%"
                except:
                    continue

            # Check if hyperopt is still running
            is_running = False
            try:
                import subprocess
                result = subprocess.run(['pgrep', '-f', 'hyperopt.*AdaptiveETHv6'], capture_output=True, text=True)
                is_running = result.returncode == 0
            except:
                pass

            pct = min(100, round(epochs / TARGET_EPOCHS * 100, 1))

            # ETA
            mtime = os.path.getmtime(fpath)
            ctime = os.path.getctime(fpath)
            elapsed = time.time() - ctime
            if epochs > 0 and is_running:
                per_epoch = elapsed / epochs
                remaining = (TARGET_EPOCHS - epochs) * per_epoch
                eta_min = round(remaining / 60, 1)
            else:
                eta_min = 0

            status = {
                "running": is_running,
                "epochs": epochs,
                "target": TARGET_EPOCHS,
                "pct": pct,
                "best_profit": best_profit,
                "best_pct": f"+{best_profit:.2f}%",
                "best_trades": best_trades,
                "eta_min": eta_min,
                "elapsed_min": round(elapsed / 60, 1),
                "updated": int(time.time()),
            }

        with open(OUTPUT, 'w') as f:
            json.dump(status, f)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

    time.sleep(5)
