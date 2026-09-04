"""User-facing daily stock prediction product."""

from .product import (
    DailyPredictionError,
    DailyPredictionSettings,
    history,
    latest,
    predict_daily,
)

__all__ = [
    "DailyPredictionError",
    "DailyPredictionSettings",
    "history",
    "latest",
    "predict_daily",
]
