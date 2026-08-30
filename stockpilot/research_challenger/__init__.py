"""Canonical, research-only ranking challenger framework.

This package is intentionally model-version agnostic.  V31 is the first protocol
using it; future experiments must vary frozen configs/artifacts rather than copy
the implementation into another versioned source directory.
"""

from .config import ChallengerSettings

__all__ = ["ChallengerSettings"]
