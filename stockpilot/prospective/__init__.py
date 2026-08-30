"""Forward-only PIT observation and alpha-validation infrastructure.

This package deliberately contains no model-training entry point.  It turns
prospectively witnessed source data into immutable observations, PIT features,
and mature labels that a future challenger may consume only after factual
readiness gates pass.
"""

from .readiness import ReadinessStatus, derive_readiness

__all__ = ["ReadinessStatus", "derive_readiness"]
