"""Independent frozen-box breakout process research module."""

from .engine import BreakoutProcessEngine
from .fast import BreakoutEventBuilder
from .models import (
    BoxState,
    BreakoutConfig,
    BreakoutRunResult,
    ColumnMap,
    FactorStage,
    OperatorContext,
)
from .operators import FactorOperator, OperatorRegistry, default_operator_registry

__all__ = [
    "BoxState",
    "BreakoutConfig",
    "BreakoutEventBuilder",
    "BreakoutProcessEngine",
    "BreakoutRunResult",
    "ColumnMap",
    "FactorOperator",
    "FactorStage",
    "OperatorContext",
    "OperatorRegistry",
    "default_operator_registry",
]
