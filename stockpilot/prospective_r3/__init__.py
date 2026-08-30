"""Evidence-derived prospective alpha certification revision.

This package is operational infrastructure only.  It has no model-training
entry point and never authorizes execution.
"""

from .config import OperationalSettings, ReadinessThresholds

__all__ = ["OperationalSettings", "ReadinessThresholds"]
