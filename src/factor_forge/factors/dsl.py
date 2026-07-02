from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factor_forge.exceptions import DSLValidationError
from . import operators as ops


FIELD_ALIASES = {
    "open": "adj_open",
    "high": "adj_high",
    "low": "adj_low",
    "close": "adj_close",
    "volume": "volume_shares",
    "amount": "amount_cny",
    "market_cap": "total_mv_cny",
    "industry": "industry_l1_code",
}

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_COMPARE = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


@dataclass
class DSLContext:
    panel: pd.DataFrame
    values: dict[str, Any]
    min_group_size: int = 10
    allowed_fields: set[str] | None = None


class FormulaEvaluator:
    """Evaluate the deliberately small V1 formula language using Python's AST."""

    def __init__(self, context: DSLContext):
        self.context = context

    def evaluate(self, formula: str) -> pd.Series:
        try:
            tree = ast.parse(formula, mode="eval")
        except SyntaxError as exc:
            raise DSLValidationError(f"Invalid formula syntax: {exc}") from exc
        result = self._eval(tree.body)
        if np.isscalar(result):
            result = pd.Series(result, index=self.context.panel.index, dtype=float)
        if not isinstance(result, pd.Series):
            raise DSLValidationError("A factor formula must produce a Series")
        return result.reindex(self.context.panel.index).replace([np.inf, -np.inf], np.nan)

    def _eval(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise DSLValidationError("Only numeric and boolean literals are allowed")
        if isinstance(node, ast.Name):
            if node.id in self.context.values:
                return self.context.values[node.id]
            field = FIELD_ALIASES.get(node.id, node.id)
            if field in self.context.panel.columns:
                if self.context.allowed_fields is not None and field not in self.context.allowed_fields:
                    raise DSLValidationError(
                        f"Field {node.id!r} is used by the formula but is not declared in data.required_fields"
                    )
                return self.context.panel[field]
            raise DSLValidationError(f"Unknown field or feature: {node.id}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.Not):
                return ~value if isinstance(value, pd.Series) else not value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            fn = _COMPARE.get(type(node.ops[0]))
            if fn:
                return fn(self._eval(node.left), self._eval(node.comparators[0]))
        if isinstance(node, ast.BoolOp):
            values = [self._eval(item) for item in node.values]
            fn = operator.and_ if isinstance(node.op, ast.And) else operator.or_
            result = values[0]
            for value in values[1:]:
                result = fn(result, value)
            return result
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [self._eval(arg) for arg in node.args]
            kwargs = {item.arg: self._eval(item.value) for item in node.keywords}
            return self._call(node.func.id, args, kwargs)
        raise DSLValidationError(
            f"Unsupported syntax in V1 DSL: {node.__class__.__name__}"
        )

    def _call(self, name: str, args: list, kwargs: dict):
        def window(minimum: int = 1) -> int:
            value = kwargs.pop("window", args.pop(1) if len(args) > 1 else None)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < minimum:
                raise DSLValidationError(f"{name} window/period must be an integer >= {minimum}")
            return int(value)
        if name == "ret":
            return ops.ret(args[0], window(1))
        if name == "lag":
            return ops.lag(args[0], window(0))
        if name == "delta":
            return ops.delta(args[0], window(1))
        if name in {"mean", "std", "slope", "max", "min"}:
            fn = {
                "mean": ops.ts_mean, "std": ops.ts_std, "slope": ops.slope,
                "max": ops.ts_max, "min": ops.ts_min,
            }[name]
            return fn(args[0], window(2 if name == "slope" else 1))
        if name in {"cs_rank", "cs_percentile", "cs_zscore"}:
            fn = ops.cs_zscore if name == "cs_zscore" else ops.cs_rank
            return fn(
                args[0], by=kwargs.get("by"), min_group_size=self.context.min_group_size
            )
        if name == "group_mean":
            if "by" not in kwargs:
                raise DSLValidationError("group_mean requires by=<group field>")
            return ops.group_mean(
                args[0], kwargs["by"], min_group_size=self.context.min_group_size
            )
        if name == "abs":
            return abs(args[0])
        if name == "log":
            return np.log(args[0].where(args[0] > 0))
        if name == "clip":
            lower = kwargs.get("lower", args[1] if len(args) > 1 else None)
            upper = kwargs.get("upper", args[2] if len(args) > 2 else None)
            return args[0].clip(lower=lower, upper=upper)
        if name == "where":
            if len(args) != 3:
                raise DSLValidationError("where requires condition, true_value, false_value")
            return pd.Series(np.where(args[0], args[1], args[2]), index=self.context.panel.index)
        raise DSLValidationError(f"Operator is not in the V1 registry: {name}")


def infer_lookback(formula: str, values: dict[str, Any], feature_lookbacks: dict[str, int]) -> int:
    """Infer the maximum number of prior rows required by a valid expression."""
    try:
        node = ast.parse(formula, mode="eval").body
    except SyntaxError as exc:
        raise DSLValidationError(f"Invalid formula syntax: {exc}") from exc

    def scalar(item: ast.AST) -> int:
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            value = item.value
        elif isinstance(item, ast.Name) and item.id in values:
            value = values[item.id]
        else:
            raise DSLValidationError("Window arguments must be numeric literals or declared parameters")
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise DSLValidationError("Window arguments must be non-negative integers")
        return int(value)

    def visit(item: ast.AST) -> int:
        if isinstance(item, ast.Name):
            return feature_lookbacks.get(item.id, 0)
        if isinstance(item, ast.Constant):
            return 0
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
            child = max((visit(arg) for arg in item.args[:1]), default=0)
            name = item.func.id
            window_node = next((kw.value for kw in item.keywords if kw.arg == "window"), None)
            if window_node is None and len(item.args) > 1:
                window_node = item.args[1]
            if name in {"ret", "lag", "delta"}:
                if window_node is None: raise DSLValidationError(f"{name} requires a period")
                return child + scalar(window_node)
            if name in {"mean", "std", "slope", "max", "min"}:
                if window_node is None: raise DSLValidationError(f"{name} requires a window")
                return child + max(scalar(window_node) - 1, 0)
            return max([visit(arg) for arg in item.args] + [visit(kw.value) for kw in item.keywords] + [0])
        children = list(ast.iter_child_nodes(item))
        return max((visit(child) for child in children), default=0)

    return visit(node)
