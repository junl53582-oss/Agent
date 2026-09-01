"""Self-verifying operational activation layered over frozen runtime 009.

This module is the only canonical operational CLI for Gen2 prospective
research-only predictions. The frozen 009 module remains historical runtime
implementation evidence; every state-changing operation here verifies the full
effective lock chain before delegating to it.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from stockpilot.prospective_r2.calendar import load_verified_calendar
from stockpilot.prospective_r2.integrity import (
    read_verified_json,
    sha256_file,
    verify_immutable,
    write_immutable_json,
)
from . import prospective_gen2_runtime as runtime
from .gen02_correctness import verify_final as verify_correctness_lock
from .prospective_gen2 import (
    CORRECTNESS_INTERPRETATION_LOCK,
    EXPECTED_CORRECTNESS_LOCK,
    EXPECTED_INTERPRETATION_LOCK,
    EXPECTED_V1R4_LOCK,
    EXPECTED_V6_LOCK,
    verify_human_freeze,
)


FAILED_ACTIVATION_010_DIR = (
    runtime.AMENDMENT_009 / "experiments/010_runtime_self_verification_activation"
)
FAILED_ACTIVATION_010R1_DIR = (
    runtime.AMENDMENT_009 / "experiments/010r1_runtime_self_verification_activation"
)
FAILED_ACTIVATION_010R2_DIR = (
    runtime.AMENDMENT_009 / "experiments/010r2_runtime_self_verification_activation"
)
ACTIVATION_DIR = (
    runtime.AMENDMENT_009 / "experiments/010r3_runtime_self_verification_activation"
)
ACTIVATION_LOCK = ACTIVATION_DIR / "plan.lock.json"
ACTIVE_RESEARCH_PATH = Path("artifacts/active_research.json")
V1R4_LOCK = Path("artifacts/prospective_alpha_v1r4/plan.lock.json")
V6_LOCK = Path("artifacts/research_v6/plan.lock.json")
CORRECTNESS_LOCK = Path(
    "artifacts/research_challenger/gen02/experiments/005_correctness_hardening/plan.lock.json"
)
V31_OPERATIONAL_LOCK = Path(
    "artifacts/research_v31/experiments/002_ci_verifier_fix/plan.lock.json"
)
EXPECTED_V31_OPERATIONAL_LOCK = (
    "6858eb261532b305c4470dc3b3907def3499fc1251017a4a821caf784f34600f"
)


class EffectiveRuntimeLockError(RuntimeError):
    """Raised before any operational side effect when the lock graph is invalid."""


def _verify_lock_surface(path: Path, expected_digest: str | None = None) -> dict:
    """Verify a sidecar-bound lock and every file directly bound by it."""
    digest = verify_immutable(path)
    mismatches: list[str] = []
    if expected_digest is not None and digest != expected_digest:
        mismatches.append(f"DIGEST:{path.as_posix()}")
    payload = read_verified_json(path)
    for name, expected in payload.get("files", {}).items():
        candidate = Path(name)
        if not candidate.is_absolute() and not candidate.is_file():
            candidate = path.parent / candidate
        if not candidate.is_file() or sha256_file(candidate) != expected:
            mismatches.append(name)
    return {
        "intact": not mismatches,
        "mismatches": mismatches,
        "lock_sha256": digest,
        "payload": payload,
    }


def verify_activation(path: Path = ACTIVATION_LOCK) -> dict:
    """Verify 010 without requiring its lock to contain its own digest."""
    result = _verify_lock_surface(path)
    payload = result.pop("payload")
    if payload.get("lock_id") != "GEN02-RUNTIME-SELF-VERIFICATION-ACTIVATION-010R3":
        result["mismatches"].append("ACTIVATION_LOCK_ID")
    base = runtime.verify_amendment()
    if payload.get("runtime_009_lock_sha256") != base.get("lock_sha256"):
        result["mismatches"].append("RUNTIME_009_PARENT_DIGEST")
    result["intact"] = not result["mismatches"]
    result.update(
        {
            "runtime_009_lock_sha256": payload.get("runtime_009_lock_sha256"),
            "production_prediction_ready": False,
            "execution_authorized": False,
        }
    )
    return result


def _verify_portable_v1r4_lock() -> dict:
    """Verify committed V1r4 evidence without requiring ignored research data.

    The recursive V1r4 verifier intentionally checks V6's local frozen datasets.
    Those large files are absent in a clean checkout. Operational data readiness
    remains enforced by 009 seal/preflight; this verifier proves the committed
    V1r4 lock bytes, sidecar, direct files and parent digest relationships.
    """
    result = _verify_lock_surface(V1R4_LOCK, EXPECTED_V1R4_LOCK)
    payload = result.pop("payload")
    parents = payload.get("parent_locks", {})
    files = payload.get("files", {})
    expected_relationships = {
        "parent_v1r3_lock_sha256": files.get(
            "artifacts/prospective_alpha_v1r3/plan.lock.json"
        ),
        "v18_lock_sha256": files.get("artifacts/research_v18/plan.lock.json"),
        "v20r2_lock_sha256": files.get("artifacts/research_v20r2/plan.lock.json"),
        "v30r1_forward_r2_lock_sha256": files.get(
            "artifacts/prediction_forward/v30r1_r2/plan.lock.json"
        ),
        "v6_lock_sha256": files.get("artifacts/research_v6/plan.lock.json"),
    }
    if parents.get("all_parent_locks_intact") is not True:
        result["mismatches"].append("V1R4_PARENT_STATUS")
    for key, expected in expected_relationships.items():
        if expected is None or parents.get(key) != expected:
            result["mismatches"].append(f"V1R4_PARENT_RELATION:{key}")
    result["intact"] = not result["mismatches"]
    result["v1r4_lock_sha256"] = result["lock_sha256"]
    result["runtime_data_readiness_deferred_to_seal_preflight"] = True
    return result


def verify_effective_runtime_freeze(
    settings: runtime.RuntimeSettings | None = None,
    *,
    activation_lock_path: Path = ACTIVATION_LOCK,
) -> dict:
    """Verify every trust root required by the active Gen2 runtime."""
    settings = settings or runtime.RuntimeSettings()
    failures: list[str] = []

    def capture(name: str, callback: Callable[[], dict], intact_key: str = "intact") -> dict:
        try:
            value = callback()
            if value.get(intact_key) is not True:
                failures.append(f"{name}:{value.get('mismatches', [])}")
            return value
        except Exception as error:  # fail closed while preserving component diagnostics
            failures.append(f"{name}:{type(error).__name__}:{error}")
            return {intact_key: False, "mismatches": [f"{type(error).__name__}:{error}"]}

    activation = capture("010", lambda: verify_activation(activation_lock_path))
    runtime_009 = capture("009", lambda: runtime.verify_amendment(settings))
    human_008 = capture("008", lambda: verify_human_freeze(settings))
    correctness = capture("correctness", verify_correctness_lock)
    interpretation = capture(
        "interpretation",
        lambda: _verify_lock_surface(
            Path(CORRECTNESS_INTERPRETATION_LOCK), EXPECTED_INTERPRETATION_LOCK
        ),
    )
    v31 = capture(
        "v31",
        lambda: _verify_lock_surface(
            V31_OPERATIONAL_LOCK, EXPECTED_V31_OPERATIONAL_LOCK
        ),
    )
    v1r4 = capture("v1r4", _verify_portable_v1r4_lock)
    v6_digest = sha256_file(V6_LOCK) if V6_LOCK.is_file() else None
    v6 = {"intact": v6_digest == EXPECTED_V6_LOCK, "lock_sha256": v6_digest}
    if not v6["intact"]:
        failures.append("v6:LOCK_DIGEST_MISMATCH")
    if correctness.get("lock_sha256") != EXPECTED_CORRECTNESS_LOCK:
        failures.append("correctness:LOCK_DIGEST_MISMATCH")
        correctness["intact"] = False

    # 007 is immutable parent evidence. 008 is the portable active amendment
    # and binds the 007 lock bytes plus current code and policy files.
    original_007 = settings.human_lock_path
    try:
        original_007_digest = verify_immutable(original_007)
        parent_007 = read_verified_json(settings.parent_008_lock_path).get(
            "parent_007_lock_sha256"
        )
        human_007_intact = original_007_digest == parent_007
    except Exception as error:
        failures.append(f"007:{type(error).__name__}:{error}")
        original_007_digest = None
        human_007_intact = False
    if not human_007_intact:
        failures.append("007:PARENT_BINDING_MISMATCH")

    try:
        calendar = load_verified_calendar(settings.calendar_path)
        calendar.sessions()
        calendar_intact = True
        calendar_digest = sha256_file(settings.calendar_path)
    except Exception as error:
        failures.append(f"calendar:{type(error).__name__}:{error}")
        calendar_intact = False
        calendar_digest = None

    try:
        runtime._verify_policy_hashes(settings)
        challenger_spec_intact = True
    except Exception as error:
        failures.append(f"challenger_spec:{type(error).__name__}:{error}")
        challenger_spec_intact = False

    effective = not failures
    return {
        "human_007_lock_intact": human_007_intact,
        "human_007_lock_sha256": original_007_digest,
        "parent_008_lock_intact": human_008.get("intact") is True,
        "parent_008_lock_sha256": human_008.get("lock_sha256"),
        "runtime_009_lock_intact": runtime_009.get("intact") is True,
        "runtime_009_lock_sha256": runtime_009.get("lock_sha256"),
        "self_verification_010_intact": activation.get("intact") is True,
        "self_verification_010_lock_sha256": activation.get("lock_sha256"),
        "correctness_lock_intact": correctness.get("intact") is True,
        "correctness_lock_sha256": correctness.get("lock_sha256"),
        "interpretation_lock_intact": interpretation.get("intact") is True,
        "interpretation_lock_sha256": interpretation.get("lock_sha256"),
        "v31_operational_lock_intact": v31.get("intact") is True,
        "v31_operational_lock_sha256": v31.get("lock_sha256"),
        "v1r4_lock_intact": v1r4.get("intact") is True,
        "v1r4_lock_sha256": v1r4.get("v1r4_lock_sha256"),
        "v1r4_runtime_data_readiness_deferred_to_seal_preflight": v1r4.get(
            "runtime_data_readiness_deferred_to_seal_preflight"
        )
        is True,
        "v6_lock_intact": v6["intact"],
        "v6_lock_sha256": v6_digest,
        "challenger_spec_intact": challenger_spec_intact,
        "calendar_evidence_intact": calendar_intact,
        "calendar_sha256": calendar_digest,
        "effective_operational_lock_intact": effective,
        "operational_lock_intact": effective,
        "failures": failures,
        "provider_requests_made": 0,
        "v6_champion": True,
        "gen2_promoted": False,
        "automatic_promotion_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }


def _guard(
    settings: runtime.RuntimeSettings,
    verifier: Callable[[runtime.RuntimeSettings], dict] | None = None,
) -> dict:
    try:
        result = (verifier or verify_effective_runtime_freeze)(settings)
    except Exception as error:
        raise EffectiveRuntimeLockError(
            "GEN2_EFFECTIVE_RUNTIME_LOCK_INVALID:"
            f"{type(error).__name__}:{error}"
        ) from error
    if result.get("effective_operational_lock_intact") is not True:
        raise EffectiveRuntimeLockError(
            "GEN2_EFFECTIVE_RUNTIME_LOCK_INVALID:"
            f"{result.get('failures', result.get('mismatches', []))}"
        )
    return result


def seal_inputs(
    target_date: str,
    *,
    now: datetime | None = None,
    settings=None,
    effective_verifier=None,
):
    settings = settings or runtime.RuntimeSettings()
    _guard(settings, effective_verifier)
    return runtime.seal_inputs(target_date, now=now, settings=settings)


def preflight(
    target_date: str,
    *,
    now: datetime | None = None,
    settings=None,
    effective_verifier=None,
):
    settings = settings or runtime.RuntimeSettings()
    try:
        lock = _guard(settings, effective_verifier)
    except Exception as error:
        return {
            "target_date": target_date,
            "parent_008_lock_intact": False,
            "runtime_009_lock_intact": False,
            "self_verification_010_intact": False,
            "effective_operational_lock_intact": False,
            "operational_lock_intact": False,
            "daily_prediction_allowed": False,
            "status": "GEN2_EFFECTIVE_RUNTIME_LOCK_INVALID",
            "failures": [f"{type(error).__name__}:{error}"],
            "provider_requests_made": 0,
            "production_prediction_ready": False,
            "execution_authorized": False,
        }
    result = runtime.preflight(target_date, now=now, settings=settings)
    return {**result, **lock}


def generate_prediction(
    target_date: str,
    *,
    now: datetime | None = None,
    settings=None,
    scorer=None,
    effective_verifier=None,
):
    settings = settings or runtime.RuntimeSettings()
    _guard(settings, effective_verifier)
    return runtime.generate_prediction(target_date, now=now, settings=settings, scorer=scorer)


def settle_prediction(
    prediction_date: str,
    market_path: Path,
    *,
    now: datetime | None = None,
    test_as_of_override=None,
    settings=None,
    official_alpha_requested=False,
    effective_verifier=None,
):
    settings = settings or runtime.RuntimeSettings()
    _guard(settings, effective_verifier)
    return runtime.settle_prediction(
        prediction_date,
        market_path,
        now=now,
        test_as_of_override=test_as_of_override,
        settings=settings,
        official_alpha_requested=official_alpha_requested,
    )


def derived_status(*, settings=None, effective_verifier=None) -> dict:
    settings = settings or runtime.RuntimeSettings()
    lock = _guard(settings, effective_verifier)
    checkpoint = runtime.review_checkpoint(settings)
    return {
        **checkpoint,
        **lock,
        "active_version": "prospective-alpha-v1r4",
        "ranking_incumbent": "V6",
        "gen02_historical_status": "HISTORICAL_RESEARCH_CLOSED",
        "gen02_promotion_status": "NOT_PROMOTED",
        "gen02_prospective_status": "PROSPECTIVE_RESEARCH_ONLY_APPROVED",
        "gen02_original_diagnostic_portfolio": "equal_top30",
        "gen02_frozen_prospective_portfolio": "sector_balanced_top20",
        "gen02_frozen_prospective_top_k": 20,
        "gen02_frozen_prospective_horizon": 20,
        "gen02_frozen_prospective_rebalance_trading_days": 20,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }


def write_derived_active_research_view(
    path: Path = ACTIVE_RESEARCH_PATH,
    *,
    settings=None,
    effective_verifier=None,
) -> dict:
    """Update only critical 010 fields from verified evidence, atomically."""
    status = derived_status(settings=settings, effective_verifier=effective_verifier)
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(
        {
            "gen02_prospective_runtime_entrypoint": (
                "python -m stockpilot.research_challenger.prospective_gen2_runtime_locked"
            ),
            "gen02_prospective_input_sealing_entrypoint": (
                "python -m stockpilot.research_challenger.prospective_gen2_runtime_locked "
                "seal-inputs --date YYYY-MM-DD"
            ),
            "gen02_prospective_preflight_entrypoint": (
                "python -m stockpilot.research_challenger.prospective_gen2_runtime_locked "
                "preflight --date YYYY-MM-DD"
            ),
            "gen02_prospective_prediction_entrypoint": (
                "python -m stockpilot.research_challenger.prospective_gen2_runtime_locked "
                "predict --date YYYY-MM-DD"
            ),
            "gen02_prospective_settlement_entrypoint": (
                "python -m stockpilot.research_challenger.prospective_gen2_runtime_locked "
                "settle --date YYYY-MM-DD --market PATH"
            ),
            "gen02_effective_runtime_status": "SELF_VERIFYING_RUNTIME_010R3_ACTIVE",
            "gen02_runtime_self_verification_revision": "010r3",
            "gen02_self_verification_010_lock_sha256": status[
                "self_verification_010_lock_sha256"
            ],
            "gen02_parent_008_lock_intact": status["parent_008_lock_intact"],
            "gen02_parent_008_operational_lock_sha256": status[
                "parent_008_lock_sha256"
            ],
            "gen02_runtime_009_lock_intact": status["runtime_009_lock_intact"],
            "gen02_self_verification_010_intact": status[
                "self_verification_010_intact"
            ],
            "gen02_effective_operational_lock_intact": status[
                "effective_operational_lock_intact"
            ],
            "gen02_effective_operational_lock_sha256": status[
                "self_verification_010_lock_sha256"
            ],
            # Legacy ambiguous keys remain for compatibility, but now point to
            # the active 010 boundary rather than presenting parent 008 as the
            # complete operational freeze.
            "gen02_prospective_operational_lock_sha256": status[
                "self_verification_010_lock_sha256"
            ],
            "gen02_effective_amendment_lock_sha256": status[
                "self_verification_010_lock_sha256"
            ],
            "gen02_original_diagnostic_portfolio": status[
                "gen02_original_diagnostic_portfolio"
            ],
            "gen02_frozen_prospective_portfolio": status[
                "gen02_frozen_prospective_portfolio"
            ],
            "gen02_frozen_prospective_top_k": status["gen02_frozen_prospective_top_k"],
            "gen02_frozen_prospective_horizon": status[
                "gen02_frozen_prospective_horizon"
            ],
            "gen02_frozen_prospective_rebalance_trading_days": status[
                "gen02_frozen_prospective_rebalance_trading_days"
            ],
            "gen02_prospective_prediction_count": status[
                "prediction_trading_days"
            ],
            "gen02_prospective_settled_20d_count": status["settled_20d_dates"],
            "production_prediction_ready": False,
            "execution_authorized": False,
        }
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return current


def freeze_activation(*, now: datetime | None = None) -> dict:
    """Freeze 010r3 once, without including the resulting lock in its own map."""
    now = now or datetime.now(timezone.utc)
    if ACTIVATION_DIR.exists() and any(ACTIVATION_DIR.iterdir()):
        raise RuntimeError("SELF_VERIFICATION_ACTIVATION_010R3_ALREADY_EXISTS")
    failed_lock = FAILED_ACTIVATION_010_DIR / "plan.lock.json"
    failure_receipt = FAILED_ACTIVATION_010_DIR / "failure_receipt.json"
    if not failed_lock.is_file() or not failure_receipt.is_file():
        raise RuntimeError("FAILED_ACTIVATION_010_EVIDENCE_MISSING")
    verify_immutable(failed_lock)
    verify_immutable(failure_receipt)
    failed_010r1_lock = FAILED_ACTIVATION_010R1_DIR / "plan.lock.json"
    ci_failure_receipt = FAILED_ACTIVATION_010R1_DIR / "ci_failure_receipt.json"
    if not failed_010r1_lock.is_file() or not ci_failure_receipt.is_file():
        raise RuntimeError("FAILED_ACTIVATION_010R1_EVIDENCE_MISSING")
    verify_immutable(failed_010r1_lock)
    verify_immutable(ci_failure_receipt)
    failed_010r2_lock = FAILED_ACTIVATION_010R2_DIR / "plan.lock.json"
    ci_failure_010r2 = FAILED_ACTIVATION_010R2_DIR / "ci_failure_receipt.json"
    if not failed_010r2_lock.is_file() or not ci_failure_010r2.is_file():
        raise RuntimeError("FAILED_ACTIVATION_010R2_EVIDENCE_MISSING")
    verify_immutable(failed_010r2_lock)
    verify_immutable(ci_failure_010r2)
    base = runtime.verify_amendment()
    human = verify_human_freeze()
    correctness = verify_correctness_lock()
    if (
        base.get("intact") is not True
        or human.get("intact") is not True
        or correctness.get("intact") is not True
    ):
        raise RuntimeError("GEN2_PARENT_LOCK_INVALID_BEFORE_010_FREEZE")
    v1r4 = _verify_portable_v1r4_lock()
    if v1r4["intact"] is not True:
        raise RuntimeError(f"V1R4_LOCK_INVALID_BEFORE_010R3_FREEZE:{v1r4['mismatches']}")
    if sha256_file(V6_LOCK) != EXPECTED_V6_LOCK:
        raise RuntimeError("V6_LOCK_INVALID_BEFORE_010_FREEZE")
    v31 = _verify_lock_surface(V31_OPERATIONAL_LOCK, EXPECTED_V31_OPERATIONAL_LOCK)
    if v31["intact"] is not True:
        raise RuntimeError(f"V31_LOCK_INVALID_BEFORE_010R3_FREEZE:{v31['mismatches']}")

    ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "amendment_id": "GEN02-RUNTIME-SELF-VERIFICATION-ACTIVATION-010R3",
        "classification": "OPERATIONAL_SELF_VERIFICATION_ONLY",
        "canonical_entrypoint": (
            "python -m stockpilot.research_challenger.prospective_gen2_runtime_locked"
        ),
        "reason": (
            "all formal runtime commands must verify 007/008/009/010 and required "
            "trust roots before side effects"
        ),
        "runtime_009_lock_sha256": base["lock_sha256"],
        "failed_activation_010_lock_sha256": sha256_file(failed_lock),
        "failed_activation_010_preserved": True,
        "failed_activation_010r1_lock_sha256": sha256_file(failed_010r1_lock),
        "failed_activation_010r1_preserved": True,
        "failed_activation_010r2_lock_sha256": sha256_file(failed_010r2_lock),
        "failed_activation_010r2_preserved": True,
        "v31_operational_lock_sha256": v31["lock_sha256"],
        "v1r4_portable_lock_surface_sha256": v1r4["lock_sha256"],
        "runtime_data_readiness_gate": "009_SEAL_AND_PREFLIGHT",
        "model_changed": False,
        "feature_policy_changed": False,
        "training_policy_changed": False,
        "portfolio_policy_changed": False,
        "cost_policy_changed": False,
        "prospective_start_date_changed": False,
        "historical_tuning_runs": 0,
        "provider_requests": 0,
        "2026_holdout_opened": False,
        "automatic_promotion_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    audit = {
        "operational_commands_self_verified": [
            "verify",
            "status",
            "seal-inputs",
            "preflight",
            "predict",
            "settle",
        ],
        "original_009_preserved": True,
        "failed_activation_010_preserved": True,
        "failed_activation_010_reason": "RELATIVE_LOCK_MEMBER_RESOLUTION_BUG",
        "failed_activation_010r1_preserved": True,
        "failed_activation_010r1_reason": "FROZEN_V31_WORKFLOW_MEMBER_CHANGED",
        "failed_activation_010r2_preserved": True,
        "failed_activation_010r2_reason": "CLEAN_CHECKOUT_REQUIRED_IGNORED_V6_DATA",
        "activation_self_reference_strategy": (
            "plan lock hashes code, tests, parents and protocol/audit; sidecar hashes "
            "plan lock; plan lock never hashes itself"
        ),
        "first_real_prediction_generated": False,
        "provider_requests": {"market": 0, "financial": 0, "benchmark": 0},
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    protocol_hash = write_immutable_json(
        ACTIVATION_DIR / "protocol_amendment.json", protocol
    )
    audit_hash = write_immutable_json(ACTIVATION_DIR / "audit.json", audit)
    files = [
        Path("stockpilot/research_challenger/prospective_gen2_runtime_locked.py"),
        Path("tests/test_research_challenger_gen2_runtime_lock_activation.py"),
        Path(".github/workflows/prospective-integrity.yml"),
        runtime.AMENDMENT_009 / "plan.lock.json",
        failed_lock,
        failure_receipt,
        failed_010r1_lock,
        ci_failure_receipt,
        failed_010r2_lock,
        ci_failure_010r2,
        runtime.AMENDMENT_008,
        runtime.HUMAN_DIR / "plan.lock.json",
        runtime.HUMAN_DIR / "decision.json",
        runtime.HUMAN_DIR / "challenger_spec.json",
        CORRECTNESS_LOCK,
        Path(CORRECTNESS_INTERPRETATION_LOCK),
        V1R4_LOCK,
        V6_LOCK,
        V31_OPERATIONAL_LOCK,
        runtime.RuntimeSettings().calendar_path,
        ACTIVATION_DIR / "protocol_amendment.json",
        ACTIVATION_DIR / "audit.json",
    ]
    lock = {
        "lock_id": "GEN02-RUNTIME-SELF-VERIFICATION-ACTIVATION-010R3",
        "created_at_utc": runtime._utc(now),
        "runtime_009_lock_sha256": base["lock_sha256"],
        "protocol_amendment_sha256": protocol_hash,
        "audit_sha256": audit_hash,
        "files": {path.as_posix(): sha256_file(path) for path in files},
        "automatic_promotion_allowed": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    lock_hash = write_immutable_json(ACTIVATION_LOCK, lock)
    manifest = {
        "protocol_amendment.json": protocol_hash,
        "audit.json": audit_hash,
        "plan.lock.json": lock_hash,
    }
    manifest_hash = write_immutable_json(ACTIVATION_DIR / "artifact_manifest.json", manifest)
    verified = verify_effective_runtime_freeze()
    if verified["effective_operational_lock_intact"] is not True:
        raise RuntimeError(
            f"GEN2_EFFECTIVE_RUNTIME_LOCK_INVALID_AFTER_FREEZE:{verified['failures']}"
        )
    write_derived_active_research_view()
    return {
        "status": "SELF_VERIFYING_RUNTIME_010R3_ACTIVE",
        "lock_sha256": lock_hash,
        "artifact_manifest_sha256": manifest_hash,
        "runtime_009_lock_sha256": base["lock_sha256"],
        "effective_operational_lock_intact": True,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-verifying Gen2 research-only runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("seal-inputs", "preflight", "predict"):
        item = sub.add_parser(name)
        item.add_argument("--date", required=True)
    settle = sub.add_parser("settle")
    settle.add_argument("--date", required=True)
    settle.add_argument("--market", type=Path, required=True)
    sub.add_parser("status")
    sub.add_parser("verify")
    sub.add_parser("freeze-010r3")
    args = parser.parse_args(argv)
    if args.command == "seal-inputs":
        result = seal_inputs(args.date)
    elif args.command == "preflight":
        result = preflight(args.date)
    elif args.command == "predict":
        result = generate_prediction(args.date)
    elif args.command == "settle":
        result = settle_prediction(args.date, args.market)
    elif args.command == "verify":
        result = verify_effective_runtime_freeze()
        if result["effective_operational_lock_intact"] is not True:
            raise EffectiveRuntimeLockError(
                f"GEN2_EFFECTIVE_RUNTIME_LOCK_INVALID:{result['failures']}"
            )
    elif args.command == "freeze-010r3":
        result = freeze_activation()
    else:
        result = derived_status()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
