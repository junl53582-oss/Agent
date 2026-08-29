from dataclasses import replace
from pathlib import Path

from stockpilot.prediction_forward_r2 import (
    ForwardR2Settings,
    V30R1_INFERENCE_ARTIFACTS,
    V30_INFERENCE_ARTIFACTS,
    verify_publishable_inference_bundle,
)


def test_publishable_bundle_is_a_strict_inference_subset():
    bundle = verify_publishable_inference_bundle()
    assert bundle["intact"] is True
    assert bundle["full_historical_validation_recertified"] is False
    assert bundle["checked_files"] == len(V30_INFERENCE_ARTIFACTS) + len(
        V30R1_INFERENCE_ARTIFACTS
    )


def test_publishable_bundle_detects_missing_model(tmp_path: Path):
    settings = replace(ForwardR2Settings(), parent_v30_root=tmp_path / "missing")
    try:
        verify_publishable_inference_bundle(settings)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing frozen parent lock must fail closed")
