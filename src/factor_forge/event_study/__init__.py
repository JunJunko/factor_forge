"""Matched-control event studies for frozen label-free observations."""

from .config import EventStudyConfig, load_event_study_config
from .mechanism_features import (
    TURNOVER_CONCENTRATION_AGGREGATE_FIELDS,
    build_turnover_concentration_aggregate_features,
    turnover_concentration_prefix_audit,
)
from .runner import EventStudyRunner

__all__ = [
    "EventStudyConfig",
    "EventStudyRunner",
    "TURNOVER_CONCENTRATION_AGGREGATE_FIELDS",
    "build_turnover_concentration_aggregate_features",
    "load_event_study_config",
    "turnover_concentration_prefix_audit",
]
