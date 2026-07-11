from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.factors.contract import (
    FACTOR_KEY_COLUMNS,
    FactorCandidate,
    validate_factor_output,
)

FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "akshare",
        "baostock",
        "httpx",
        "importlib",
        "os",
        "pathlib",
        "requests",
        "socket",
        "subprocess",
        "tushare",
        "urllib",
    }
)
FORBIDDEN_CALLS = frozenset(
    {
        "open",
        "eval",
        "exec",
        "read_bytes",
        "read_csv",
        "read_excel",
        "read_feather",
        "read_json",
        "read_parquet",
        "read_pickle",
        "read_text",
        "urlopen",
        "write_bytes",
        "write_text",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_json",
        "to_parquet",
        "to_pickle",
    }
)
LABEL_NAMES = frozenset({"label", "labels", "target", "future_return", "LABEL"})
TEST_NAMES = frozenset({"test", "test_data", "test_end", "test_set", "test_start"})


@dataclass(frozen=True)
class LeakageIssue:
    code: str
    message: str
    line: int | None


@dataclass(frozen=True)
class LeakageReport:
    passed: bool
    static_issues: tuple[LeakageIssue, ...]
    prefix_invariant: bool
    future_perturbation_invariant: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "static_issues": [asdict(issue) for issue in self.static_issues],
            "prefix_invariant": self.prefix_invariant,
            "future_perturbation_invariant": self.future_perturbation_invariant,
        }


def audit_factor(candidate: FactorCandidate, market: pd.DataFrame) -> LeakageReport:
    static_issues = tuple(
        inspect_factor_source(candidate.source_path, set(candidate.metadata.inputs))
    )
    prefix_invariant = False
    perturbation_invariant = False
    if not static_issues:
        prefix_invariant = _prefix_invariance(candidate, market)
        perturbation_invariant = _future_perturbation_invariance(candidate, market)
    return LeakageReport(
        passed=not static_issues and prefix_invariant and perturbation_invariant,
        static_issues=static_issues,
        prefix_invariant=prefix_invariant,
        future_perturbation_invariant=perturbation_invariant,
    )


def inspect_factor_source(
    source_path: Path, declared_inputs: set[str]
) -> list[LeakageIssue]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    visitor = _FactorAstVisitor(declared_inputs)
    visitor.visit(tree)
    return visitor.issues


class _FactorAstVisitor(ast.NodeVisitor):
    def __init__(self, declared_inputs: set[str]) -> None:
        self.declared_inputs = declared_inputs
        self.issues: list[LeakageIssue] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                self._add("forbidden_import", f"forbidden import: {root}", node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in FORBIDDEN_IMPORT_ROOTS:
            self._add("forbidden_import", f"forbidden import: {root}", node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in LABEL_NAMES:
            self._add("label_access", f"forbidden label name: {node.id}", node)
        if node.id.lower() in TEST_NAMES:
            self._add("test_condition", f"test-specific name: {node.id}", node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value in LABEL_NAMES:
            self._add("label_access", f"forbidden label field: {node.value}", node)
        if isinstance(node.value, str) and node.value.lower() in TEST_NAMES:
            self._add("test_condition", f"test-specific value: {node.value}", node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        field = _literal_string(node.slice)
        if (
            field is not None
            and isinstance(node.value, ast.Name)
            and node.value.id in {"frame", "ordered"}
        ):
            allowed = {*FACTOR_KEY_COLUMNS, *self.declared_inputs}
            if field not in allowed:
                self._add(
                    "undeclared_input",
                    f"field is not declared in metadata: {field}",
                    node,
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in FORBIDDEN_CALLS:
            self._add("forbidden_io", f"forbidden I/O call: {name}", node)
        if name in {"shift", "pct_change"}:
            periods = _call_integer_argument(node, "periods", 0, default=1)
            if periods is not None and periods < 0:
                self._add(
                    "future_shift",
                    f"negative {name} periods are forbidden: {periods}",
                    node,
                )
        if name == "rolling":
            if not _has_argument(node, "min_periods", 1):
                self._add(
                    "missing_min_periods",
                    "rolling() must declare min_periods",
                    node,
                )
            center = _call_boolean_argument(node, "center", default=False)
            if center:
                self._add(
                    "centered_window", "centered rolling windows are forbidden", node
                )
        self.generic_visit(node)

    def _add(self, code: str, message: str, node: ast.AST) -> None:
        self.issues.append(
            LeakageIssue(code=code, message=message, line=getattr(node, "lineno", None))
        )


def _prefix_invariance(candidate: FactorCandidate, market: pd.DataFrame) -> bool:
    dates = sorted(pd.to_datetime(market["trade_date"]).dt.normalize().unique())
    if len(dates) < candidate.metadata.lookback + 3:
        return False
    cutoff = pd.Timestamp(
        dates[max(candidate.metadata.lookback + 1, len(dates) * 2 // 3)]
    )
    full = validate_factor_output(candidate, market)
    truncated_market = market.loc[
        pd.to_datetime(market["trade_date"]).dt.normalize() <= cutoff
    ].copy()
    truncated = validate_factor_output(candidate, truncated_market)
    historical = full.loc[full["trade_date"] <= cutoff]
    return _same_values(historical, truncated)


def _future_perturbation_invariance(
    candidate: FactorCandidate, market: pd.DataFrame
) -> bool:
    dates = sorted(pd.to_datetime(market["trade_date"]).dt.normalize().unique())
    if len(dates) < candidate.metadata.lookback + 3:
        return False
    cutoff = pd.Timestamp(
        dates[max(candidate.metadata.lookback + 1, len(dates) * 2 // 3)]
    )
    original = validate_factor_output(candidate, market)
    perturbed = market.copy(deep=True)
    future_mask = pd.to_datetime(perturbed["trade_date"]).dt.normalize() > cutoff
    for column in candidate.metadata.inputs:
        if pd.api.types.is_numeric_dtype(perturbed[column]):
            perturbed.loc[future_mask, column] = (
                pd.to_numeric(perturbed.loc[future_mask, column], errors="coerce")
                * 1.137
                + 0.731
            )
    changed = validate_factor_output(candidate, perturbed)
    return _same_values(
        original.loc[original["trade_date"] <= cutoff],
        changed.loc[changed["trade_date"] <= cutoff],
    )


def _same_values(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    columns = [*FACTOR_KEY_COLUMNS, "value"]
    first = left[columns].sort_values(list(FACTOR_KEY_COLUMNS), kind="stable")
    second = right[columns].sort_values(list(FACTOR_KEY_COLUMNS), kind="stable")
    if len(first) != len(second):
        return False
    if (
        not first[list(FACTOR_KEY_COLUMNS)]
        .reset_index(drop=True)
        .equals(second[list(FACTOR_KEY_COLUMNS)].reset_index(drop=True))
    ):
        return False
    return bool(
        np.allclose(
            first["value"].to_numpy(dtype=float),
            second["value"].to_numpy(dtype=float),
            equal_nan=True,
            rtol=1e-12,
            atol=1e-12,
        )
    )


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _literal_string(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_integer_argument(
    node: ast.Call, keyword: str, position: int, *, default: int
) -> int | None:
    value: ast.expr | None = None
    for item in node.keywords:
        if item.arg == keyword:
            value = item.value
            break
    if value is None and len(node.args) > position:
        value = node.args[position]
    if value is None:
        return default
    if isinstance(value, ast.Constant) and isinstance(value.value, int):
        return value.value
    if (
        isinstance(value, ast.UnaryOp)
        and isinstance(value.op, ast.USub)
        and isinstance(value.operand, ast.Constant)
        and isinstance(value.operand.value, int)
    ):
        return -value.operand.value
    return None


def _call_boolean_argument(
    node: ast.Call, keyword: str, *, default: bool
) -> bool | None:
    for item in node.keywords:
        if item.arg == keyword:
            if isinstance(item.value, ast.Constant) and isinstance(
                item.value.value, bool
            ):
                return item.value.value
            return None
    return default


def _has_argument(node: ast.Call, keyword: str, position: int) -> bool:
    return (
        any(item.arg == keyword for item in node.keywords) or len(node.args) > position
    )
