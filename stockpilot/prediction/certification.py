from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PredictionCertificationResult:
    data_verified: bool = False
    pit_verified: bool = False
    label_maturity_verified: bool = False
    leakage_test_passed: bool = False
    purged_walk_forward_passed: bool = False
    calibration_passed: bool = False
    baseline_beaten: bool = False
    stability_passed: bool = False
    regime_passed: bool = False
    probability_quality_passed: bool = False
    cost_aware_stress_passed: bool = False
    production_prediction_ready: bool = False
    execution_authorized: bool = False
    future_126d_confirmed: bool = False
    future_confirmation_status: str = "NOT_STARTED"
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def evaluate(cls, *, future_126d_confirmed: bool = False, **checks: bool) -> "PredictionCertificationResult":
        required = (
            "data_verified", "pit_verified", "label_maturity_verified", "leakage_test_passed",
            "purged_walk_forward_passed", "calibration_passed", "baseline_beaten",
            "stability_passed", "regime_passed", "probability_quality_passed",
            "cost_aware_stress_passed",
        )
        values = {name: bool(checks.get(name, False)) for name in required}
        ready = all(values.values())
        reasons = tuple(name for name, passed in values.items() if not passed)
        future_status = "CONFIRMED" if future_126d_confirmed else "COLLECTING"
        return cls(
            **values,
            production_prediction_ready=ready,
            execution_authorized=False,
            future_126d_confirmed=bool(future_126d_confirmed),
            future_confirmation_status=future_status,
            reasons=reasons,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_certification(path: Path) -> PredictionCertificationResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reasons"] = tuple(payload.get("reasons", ()))
    return PredictionCertificationResult(**payload)

