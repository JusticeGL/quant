from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.baseline.backtest import run_topk_backtest
from alpha_lab.baseline.config import load_phase2_config


def test_topk_backtest_uses_next_day_open_lots_t_plus_one_and_fees() -> None:
    config = load_phase2_config(Path("config"))
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    rows = []
    for offset, trade_date in enumerate(dates):
        for instrument, base in (("A", 10.0), ("B", 20.0)):
            rows.append(
                {
                    "trade_date": trade_date,
                    "instrument": instrument,
                    "open": base + offset,
                    "close": base + offset + 0.5,
                    "volume": 10000.0,
                    "suspend": False,
                    "limit_up": False,
                    "limit_down": False,
                    "is_st": False,
                }
            )
    market = pd.DataFrame(rows)
    predictions = pd.DataFrame(
        {
            "datetime": [dates[0], dates[0], dates[1], dates[1]],
            "instrument": ["A", "B", "A", "B"],
            "score": [2.0, 1.0, 1.0, 2.0],
            "label": [0.01, 0.0, 0.0, 0.01],
        }
    )
    strategy = config.baseline.strategy.model_copy(update={"top_k": 1})

    result = run_topk_backtest(
        predictions,
        market,
        strategy=strategy,
        costs=config.costs,
        annualization_days=252,
        allowed_end=dates[-1].date(),
    )

    assert result.trades["side"].tolist()[:3] == ["buy", "sell", "buy"]
    assert all(result.trades["shares"] % strategy.lot_size == 0)
    assert result.metrics["total_fees"] > 0
    assert result.constraints["blocked_t_plus_one"] == 0
