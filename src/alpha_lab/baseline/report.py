from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_reports(manifest: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    markdown_path = output_dir / "baseline_report.md"
    html_path = output_dir / "baseline_report.html"
    signal = manifest["signal_analysis"]
    backtest = manifest["backtest"]
    limitations = manifest["limitations"]
    limitation_markdown = "".join(f"- {item}" + "\n" for item in limitations)
    artifacts_json = json.dumps(
        manifest["artifacts"], ensure_ascii=False, indent=2, sort_keys=True
    )
    constraints_json = json.dumps(
        backtest["constraints"], ensure_ascii=False, indent=2, sort_keys=True
    )
    train = manifest["splits"]["train"]
    validation = manifest["splits"]["validation"]
    markdown = f"""# Alpha158 + LightGBM Phase 2 Baseline

> Engineering validation only. This is not an investment-performance claim.

## Reproducibility identity

| Field | Value |
|---|---|
| Run ID | `{manifest["run_id"]}` |
| Data snapshot | `{manifest["data_snapshot_id"]}` |
| Qlib content SHA256 | `{manifest["qlib_content_sha256"]}` |
| Config SHA256 | `{manifest["config_sha256"]}` |
| Split policy SHA256 | `{manifest["split_policy_sha256"]}` |
| Cost policy SHA256 | `{manifest["cost_policy_sha256"]}` |
| Git commit | `{manifest["git"]["commit"]}` |
| Git dirty | `{str(manifest["git"]["dirty"]).lower()}` |
| Random seed | `{manifest["random_seed"]}` |
| Reproducibility SHA256 | `{manifest["reproducibility_sha256"]}` |

## Dataset and protocol

- Feature set: Qlib `Alpha158` ({manifest["feature_count"]} features).
- Label: `{manifest["label"]["expression"]}` ({manifest["label"]["name"]}).
- Train: `{train["start"]}` through `{train["end"]}`.
- Validation: `{validation["start"]}` through `{validation["end"]}`.
- Locked test: not loaded, scored, evaluated, or reported.
- Model: deterministic single-thread LightGBM regression.
- Strategy: validation-only Top-{manifest["strategy"]["top_k"]} backtest.

## Signal analysis (validation)

| Metric | Value |
|---|---:|
| Rows | {signal["row_count"]} |
| Coverage | {_number(signal["coverage"])} |
| Mean IC | {_number(signal["mean_ic"])} |
| Mean RankIC | {_number(signal["mean_rank_ic"])} |
| ICIR | {_number(signal["icir"])} |
| RankICIR | {_number(signal["rank_icir"])} |
| Positive IC ratio | {_number(signal["positive_ic_ratio"])} |
| Top-bottom spread | {_number(signal["mean_top_bottom_spread"])} |
| RMSE | {_number(signal["rmse"])} |

## Top-K backtest (validation)

| Metric | Value |
|---|---:|
| Initial cash | {_number(backtest["metrics"]["initial_cash"])} |
| Final NAV | {_number(backtest["metrics"]["final_nav"])} |
| Total return | {_number(backtest["metrics"]["total_return"])} |
| Annualized return | {_number(backtest["metrics"]["annualized_return"])} |
| Annualized volatility | {_number(backtest["metrics"]["annualized_volatility"])} |
| Sharpe (zero risk-free rate) | {_number(backtest["metrics"]["sharpe"])} |
| Maximum drawdown | {_number(backtest["metrics"]["max_drawdown"])} |
| Turnover ratio | {_number(backtest["metrics"]["turnover_ratio"])} |
| Total fees | {_number(backtest["metrics"]["total_fees"])} |
| Trades | {backtest["metrics"]["trade_count"]} |

Constraint counters:

```json
{constraints_json}
```

## Limitations

{limitation_markdown}
## Artifact inventory

```json
{artifacts_json}
```
"""
    markdown_path.write_text(markdown, encoding="utf-8")

    identity_table = _html_table(
        [
            ("Run ID", manifest["run_id"]),
            ("Data snapshot", manifest["data_snapshot_id"]),
            ("Config SHA256", manifest["config_sha256"]),
            ("Git commit", manifest["git"]["commit"]),
            ("Git dirty", manifest["git"]["dirty"]),
            ("Random seed", manifest["random_seed"]),
            ("Reproducibility SHA256", manifest["reproducibility_sha256"]),
        ]
    )
    signal_table = _html_table(
        [(key, value) for key, value in signal.items() if key != "daily"]
    )
    backtest_table = _html_table(list(backtest["metrics"].items()))
    constraint_table = _html_table(list(backtest["constraints"].items()))
    limitation_html = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in limitations
    )
    manifest_html = html.escape(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    )
    html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 2 Alpha158 Baseline</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1000px;
margin: 40px auto; padding: 0 20px; color: #17202a; }}
h1, h2 {{ color: #12355b; }}
.warning {{ background: #fff3cd; padding: 14px;
border-left: 5px solid #e0a800; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
th {{ background: #eef4f8; }}
code, pre {{ background: #f6f8fa; padding: 2px 5px; }}
pre {{ padding: 14px; overflow: auto; }}
</style></head><body>
<h1>Alpha158 + LightGBM Phase 2 Baseline</h1>
<p class="warning"><strong>Engineering validation only.</strong>
This is not an investment-performance claim.</p>
<h2>Reproducibility identity</h2>{identity_table}
<h2>Signal analysis (validation)</h2>{signal_table}
<h2>Top-K backtest (validation)</h2>{backtest_table}
<h2>Constraint counters</h2>{constraint_table}
<h2>Limitations</h2><ul>{limitation_html}</ul>
<h2>Machine-readable manifest</h2><pre>{manifest_html}</pre>
</body></html>"""
    html_path.write_text(html_report, encoding="utf-8")
    return markdown_path, html_path


def _number(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, int)):
        return f"{value:.8f}"
    return str(value)


def _html_table(rows: list[tuple[str, object]]) -> str:
    body = "".join(
        f"<tr><th>{html.escape(str(key))}</th>"
        f"<td>{html.escape(_number(value))}</td></tr>"
        for key, value in rows
    )
    return f"<table><tbody>{body}</tbody></table>"
