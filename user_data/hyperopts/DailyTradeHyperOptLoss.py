"""
Custom HyperOpt Loss — targets ~1 trade per day while maximizing profit.

Penalizes results that deviate from the target trade frequency.
"""
from datetime import datetime
from freqtrade.optimize.hyperopt import IHyperOptLoss
from pandas import DataFrame


class DailyTradeHyperOptLoss(IHyperOptLoss):
    """
    Loss function that balances profit with trade frequency.
    Targets approximately 1 trade per day.
    """

    TARGET_TRADES_PER_DAY = 1.0

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        *args,
        **kwargs,
    ) -> float:
        total_days = (max_date - min_date).days or 1
        trades_per_day = trade_count / total_days

        # Profit component (negative = better in hyperopt)
        total_profit = results["profit_ratio"].sum()

        # Trade frequency penalty: penalize deviation from target
        target = DailyTradeHyperOptLoss.TARGET_TRADES_PER_DAY
        freq_ratio = trades_per_day / target if trades_per_day > 0 else 0.01

        # Log-based penalty: symmetric around target (1.0)
        import math
        freq_penalty = abs(math.log(freq_ratio)) * 2.0

        # Win rate bonus
        if trade_count > 0:
            win_rate = len(results[results["profit_ratio"] > 0]) / trade_count
        else:
            return 100.0  # No trades = worst

        # Too few trades penalty (hard floor)
        if trade_count < 100:
            scarcity_penalty = (100 - trade_count) * 0.1
        else:
            scarcity_penalty = 0

        # Combined loss: lower is better
        # - Profit: more negative = better (we negate it so profit lowers loss)
        # - freq_penalty: deviation from 1 trade/day
        # - win_rate: higher = lower loss
        # - scarcity_penalty: under 100 trades gets punished
        loss = -total_profit + freq_penalty + scarcity_penalty - (win_rate * 2.0)

        return loss
