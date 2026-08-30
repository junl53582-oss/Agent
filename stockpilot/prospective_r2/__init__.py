"""Prospective Alpha V1r2 operational hardening.

This package coordinates evidence capture only.  It deliberately exposes no
model-training or execution entry point.
"""

from .config import OperationalSettings, ReadinessThresholds
from .orchestrator import DailyDependencies, run_official_daily

__all__ = [
    "DailyDependencies",
    "OperationalSettings",
    "ReadinessThresholds",
    "run_official_daily",
]
