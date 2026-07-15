from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd

from alpha_lab.baseline.config import CostConfig, CostRule, StrategyConfig


@dataclass
class Position:
    shares: int
    acquired_on: date


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]
    constraints: dict[str, int]


def run_topk_backtest(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    *,
    strategy: StrategyConfig,
    costs: CostConfig,
    annualization_days: int,
    allowed_end: date,
) -> BacktestResult:
    frame = _prepare_market(market, allowed_end)
    dates = [value.date() for value in sorted(frame["trade_date"].unique())]
    if len(dates) < 3:
        raise ValueError("backtest needs at least three market dates")

    signals = predictions.dropna(subset=["score"]).copy()
    signals["datetime"] = pd.to_datetime(signals["datetime"]).dt.date
    target_by_trade_date: dict[date, list[str]] = {}
    exit_by_trade_date: dict[date, set[str]] = {}
    explicit_execution_dates = {"entry_date", "exit_date"}.issubset(signals.columns)
    if explicit_execution_dates:
        signals["entry_date"] = pd.to_datetime(
            signals["entry_date"], errors="raise"
        ).dt.date
        signals["exit_date"] = pd.to_datetime(
            signals["exit_date"], errors="raise"
        ).dt.date
        invalid = signals[["entry_date", "exit_date"]].isna().any(axis=1) | ~(
            (signals["datetime"] < signals["entry_date"])
            & (signals["entry_date"] < signals["exit_date"])
        )
        if invalid.any():
            raise ValueError("prediction execution dates are missing or not ordered")
    date_position = {value: index for index, value in enumerate(dates)}
    for signal_date, part in signals.groupby("datetime", sort=True):
        ordered = part.sort_values(
            ["score", "instrument"], ascending=[False, True], kind="stable"
        )
        selected = ordered.head(strategy.top_k)
        if explicit_execution_dates:
            selected_dates = set(selected["entry_date"]) | set(selected["exit_date"])
            if any(
                trade_date not in date_position or trade_date > allowed_end
                for trade_date in selected_dates
            ):
                raise ValueError("prediction execution date is outside market data")
            for trade_date, entries in selected.groupby("entry_date", sort=True):
                target_by_trade_date.setdefault(trade_date, []).extend(
                    str(value) for value in entries["instrument"]
                )
            for trade_date, exits in selected.groupby("exit_date", sort=True):
                exit_by_trade_date.setdefault(trade_date, set()).update(
                    str(value) for value in exits["instrument"]
                )
        else:
            index = date_position.get(signal_date)
            if index is None or index + 1 >= len(dates):
                continue
            trade_date = dates[index + 1]
            target_by_trade_date[trade_date] = [
                str(value) for value in selected["instrument"]
            ]

    cash = strategy.initial_cash
    positions: dict[str, Position] = {}
    last_close: dict[str, float] = {}
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    constraints = {
        "blocked_suspend": 0,
        "blocked_limit": 0,
        "blocked_st": 0,
        "blocked_unknown_status": 0,
        "blocked_t_plus_one": 0,
        "blocked_lot_or_cash": 0,
        "inferred_price_limit_checks": 0,
    }

    by_date = {
        value.date(): part.set_index("instrument")
        for value, part in frame.groupby("trade_date", sort=True)
    }
    active_dates = sorted(set(target_by_trade_date) | set(exit_by_trade_date))
    if not active_dates:
        raise ValueError("no validation signal has an executable next trading day")

    for trade_date in [value for value in dates if value >= active_dates[0]]:
        day = by_date[trade_date]
        target: list[str] | None
        if explicit_execution_dates and (
            trade_date in target_by_trade_date or trade_date in exit_by_trade_date
        ):
            target = sorted(
                (set(positions) - exit_by_trade_date.get(trade_date, set()))
                | set(target_by_trade_date.get(trade_date, []))
            )
        else:
            target = target_by_trade_date.get(trade_date)
        turnover = 0.0
        fees = 0.0
        if target is not None:
            for instrument in sorted(set(positions) - set(target)):
                position = positions[instrument]
                row = _market_row(day, instrument)
                reason = _blocked_reason(
                    row, "sell", strategy, constraints, position.acquired_on, trade_date
                )
                if reason is not None:
                    constraints[reason] += 1
                    continue
                notional = position.shares * float(row["open"])
                fee = _transaction_fee(notional, "sell", costs.rule_for(trade_date))
                cash += notional - fee
                turnover += notional
                fees += fee
                trade_rows.append(
                    _trade_row(
                        trade_date,
                        instrument,
                        "sell",
                        position.shares,
                        float(row["open"]),
                        fee,
                    )
                )
                del positions[instrument]

            marked_equity = cash + _position_value(
                positions, day, "open", last_prices=last_close
            )
            target_value = marked_equity / strategy.top_k
            for instrument in target:
                if instrument in positions:
                    continue
                row = _market_row(day, instrument)
                reason = _blocked_reason(
                    row, "buy", strategy, constraints, None, trade_date
                )
                if reason is not None:
                    constraints[reason] += 1
                    continue
                price = float(row["open"])
                budget = min(target_value, cash)
                shares = int(budget // (price * strategy.lot_size)) * strategy.lot_size
                while shares > 0:
                    notional = shares * price
                    fee = _transaction_fee(notional, "buy", costs.rule_for(trade_date))
                    if notional + fee <= cash:
                        break
                    shares -= strategy.lot_size
                if shares <= 0:
                    constraints["blocked_lot_or_cash"] += 1
                    continue
                notional = shares * price
                fee = _transaction_fee(notional, "buy", costs.rule_for(trade_date))
                cash -= notional + fee
                turnover += notional
                fees += fee
                positions[instrument] = Position(shares=shares, acquired_on=trade_date)
                trade_rows.append(
                    _trade_row(trade_date, instrument, "buy", shares, price, fee)
                )

        close_value = _position_value(positions, day, "close", last_prices=last_close)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "cash": cash,
                "position_value": close_value,
                "nav": cash + close_value,
                "holding_count": len(positions),
                "turnover_notional": turnover,
                "fees": fees,
            }
        )
        last_close.update(
            (str(instrument), float(value))
            for instrument, value in day["close"].items()
        )

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(
        trade_rows,
        columns=["trade_date", "instrument", "side", "shares", "price", "fee"],
    )
    metrics = _backtest_metrics(daily, strategy.initial_cash, annualization_days)
    metrics["trade_count"] = len(trades)
    metrics["total_fees"] = float(daily["fees"].sum())
    metrics["turnover_ratio"] = float(
        daily["turnover_notional"].sum() / daily["nav"].mean()
    )
    return BacktestResult(
        daily=daily, trades=trades, metrics=metrics, constraints=constraints
    )


def _prepare_market(market: pd.DataFrame, allowed_end: date) -> pd.DataFrame:
    required = {
        "trade_date",
        "instrument",
        "open",
        "close",
        "volume",
        "suspend",
        "limit_up",
        "limit_down",
        "is_st",
    }
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"market data is missing backtest fields: {missing}")
    frame = market.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame.loc[frame["trade_date"].dt.date <= allowed_end].copy()
    frame = frame.sort_values(["instrument", "trade_date"], kind="stable")
    frame["previous_close"] = frame.groupby("instrument")["close"].shift(1)
    return frame.sort_values(["trade_date", "instrument"], kind="stable")


def _market_row(day: pd.DataFrame, instrument: str) -> pd.Series[Any]:
    if instrument not in day.index:
        raise ValueError(f"market row missing for {instrument}")
    row = day.loc[instrument]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"duplicate market rows for {instrument}")
    return row


def _blocked_reason(
    row: pd.Series[Any],
    side: Literal["buy", "sell"],
    strategy: StrategyConfig,
    constraints: dict[str, int],
    acquired_on: date | None,
    trade_date: date,
) -> str | None:
    if pd.isna(row["suspend"]):
        return "blocked_unknown_status"
    if bool(row["suspend"]) or float(row["volume"]) <= 0:
        return "blocked_suspend"
    if side == "buy" and strategy.exclude_st:
        if pd.isna(row["is_st"]):
            return "blocked_unknown_status"
        if bool(row["is_st"]):
            return "blocked_st"
    if side == "sell" and acquired_on is not None and acquired_on >= trade_date:
        return "blocked_t_plus_one"

    explicit_key = "limit_up" if side == "buy" else "limit_down"
    explicit = row[explicit_key]
    if not pd.isna(explicit):
        return "blocked_limit" if bool(explicit) else None
    if not strategy.infer_missing_price_limits:
        return "blocked_unknown_status"
    constraints["inferred_price_limit_checks"] += 1
    previous_close = row["previous_close"]
    if pd.isna(previous_close) or float(previous_close) <= 0:
        return "blocked_unknown_status"
    is_st = False if pd.isna(row["is_st"]) else bool(row["is_st"])
    ratio = strategy.st_limit_ratio if is_st else strategy.main_board_limit_ratio
    open_return = float(row["open"]) / float(previous_close) - 1.0
    threshold = ratio - strategy.price_limit_tolerance
    if side == "buy" and open_return >= threshold:
        return "blocked_limit"
    if side == "sell" and open_return <= -threshold:
        return "blocked_limit"
    return None


def _transaction_fee(
    notional: float, side: Literal["buy", "sell"], rule: CostRule
) -> float:
    commission = max(rule.minimum_commission, notional * rule.commission_rate)
    if side == "buy":
        rate = rule.stamp_duty_rate_buy + rule.transfer_fee_rate_buy
    else:
        rate = rule.stamp_duty_rate_sell + rule.transfer_fee_rate_sell
    return commission + notional * rate


def _position_value(
    positions: dict[str, Position],
    day: pd.DataFrame,
    field: Literal["open", "close"],
    *,
    last_prices: dict[str, float] | None = None,
) -> float:
    total = 0.0
    for instrument, position in positions.items():
        if instrument not in day.index and last_prices is not None:
            try:
                total += position.shares * last_prices[instrument]
            except KeyError as error:
                raise ValueError(
                    f"no current or prior price for held instrument {instrument}"
                ) from error
            continue
        row = _market_row(day, instrument)
        total += position.shares * float(row[field])
    return total


def _trade_row(
    trade_date: date,
    instrument: str,
    side: Literal["buy", "sell"],
    shares: int,
    price: float,
    fee: float,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "instrument": instrument,
        "side": side,
        "shares": shares,
        "price": price,
        "fee": fee,
    }


def _backtest_metrics(
    daily: pd.DataFrame, initial_cash: float, annualization_days: int
) -> dict[str, float | None]:
    if daily.empty:
        raise ValueError("backtest produced no daily observations")
    nav = daily["nav"].astype(float)
    returns = nav.pct_change().dropna()
    total_return = float(nav.iloc[-1] / initial_cash - 1.0)
    periods = max(len(returns), 1)
    annualized_return = float(
        (1.0 + total_return) ** (annualization_days / periods) - 1.0
    )
    volatility = (
        float(returns.std(ddof=1) * math.sqrt(annualization_days))
        if len(returns) >= 2
        else None
    )
    sharpe = None
    if volatility is not None and volatility > 0:
        sharpe = float(
            returns.mean() / returns.std(ddof=1) * math.sqrt(annualization_days)
        )
    drawdown = nav / nav.cummax() - 1.0
    return {
        "initial_cash": initial_cash,
        "final_nav": float(nav.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }
