from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from stockpilot.prospective_r2.integrity import (
    canonical_json_bytes,
    read_verified_json,
    sha256_bytes,
    sha256_file,
    verify_immutable,
    write_immutable_json,
)
from stockpilot.prospective_r2.observation import universe_hash
from stockpilot.prospective_r2.sources import load_pit_context

from .config import OperationalSettings


@dataclass(frozen=True)
class ObservationCertification:
    observation_id: str
    target_date: str
    certified_at: str
    reservation_verified: bool
    official_lock_verified: bool
    calendar_verified: bool
    membership_verified: bool
    industry_verified: bool
    observation_receipt_verified: bool
    earnings_source_verified: bool
    raw_evidence_verified: bool
    normalized_evidence_verified: bool
    coverage_verified: bool
    actual_universe_coverage: float
    qualifying_observation: bool
    certification_failures: tuple[str, ...]
    certification_method: str = "EVIDENCE_RECOMPUTED_NOT_SELF_DECLARED"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["certification_failures"] = list(self.certification_failures)
        return value


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _verified_lock(settings: OperationalSettings, verifier: Callable | None) -> dict:
    if verifier is None:
        from .freeze import verify_lock

        verifier = verify_lock
    evidence = verifier(settings)
    expected = sha256_file(settings.plan_lock_path)
    if evidence.get("v1r3_lock_sha256") != expected:
        raise RuntimeError("official V1r3 lock evidence does not match the approved lock")
    if evidence.get("frozen_inputs_intact") is not True:
        raise RuntimeError("official V1r3 frozen inputs are not intact")
    return evidence


def _strip_runtime_receipt_fields(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key not in {"receipt_path", "receipt_sha256"}
    }


def _verify_reservation(
    observation: dict,
    settings: OperationalSettings,
    approved_lock_hash: str,
) -> None:
    expected_path = settings.attempts_root / f"{observation['target_date']}.json"
    if not _same_path(observation["attempt_path"], expected_path):
        raise RuntimeError("observation is not linked to the canonical daily reservation")
    if sha256_file(expected_path) != observation["attempt_sha256"]:
        raise RuntimeError("daily reservation hash mismatch")
    import json

    attempt = json.loads(expected_path.read_text(encoding="utf-8"))
    required = {
        "target_date": observation["target_date"],
        "observation_attempt_id": observation["observation_id"],
        "parent_lock_sha256": approved_lock_hash,
        "retry_allowed": False,
        "automatic_retry": False,
        "manual_retry": False,
    }
    mismatches = [name for name, expected in required.items() if attempt.get(name) != expected]
    if mismatches:
        raise RuntimeError(f"daily reservation linkage mismatch: {mismatches}")


def _verify_observation_receipt(observation: dict, settings: OperationalSettings) -> None:
    expected = settings.observations_root / observation["observation_id"] / "observation.json"
    if not _same_path(observation["observation_path"], expected):
        raise RuntimeError("observation receipt is outside the canonical runtime namespace")
    if verify_immutable(expected) != observation["observation_sha256"]:
        raise RuntimeError("observation receipt hash mismatch")
    stored = read_verified_json(expected)
    supplied = {
        key: value
        for key, value in observation.items()
        if key not in {"observation_path", "observation_sha256"}
    }
    if canonical_json_bytes(stored) != canonical_json_bytes(supplied):
        raise RuntimeError("loaded observation differs from its immutable receipt")


def _verify_source(
    observation: dict,
    universe: set[str],
    minimum_coverage: float,
) -> tuple[bool, bool, bool, bool, float]:
    source = observation.get("sources", {}).get("earnings_expectations")
    if not isinstance(source, dict):
        raise RuntimeError("earnings expectations source receipt is missing")
    if source.get("source") != "earnings_expectations":
        raise RuntimeError("earnings source identity mismatch")
    if source.get("source_status") != "SUCCESS" or source.get("success") is not True:
        raise RuntimeError("earnings source did not complete successfully")
    if source.get("silent_fallback_used") is not False:
        raise RuntimeError("silent fallback is forbidden")
    if int(source.get("conflicting_duplicate_count", -1)) != 0:
        raise RuntimeError("earnings source contains conflicting duplicates")

    receipt_path = Path(source["receipt_path"])
    if verify_immutable(receipt_path) != source["receipt_sha256"]:
        raise RuntimeError("earnings source receipt hash mismatch")
    stored_receipt = read_verified_json(receipt_path)
    if canonical_json_bytes(stored_receipt) != canonical_json_bytes(
        _strip_runtime_receipt_fields(source)
    ):
        raise RuntimeError("observation source metadata differs from source receipt")

    raw_paths = source.get("raw_paths") or []
    raw_hashes = source.get("raw_response_sha256") or []
    if not raw_paths or len(raw_paths) != len(raw_hashes):
        raise RuntimeError("earnings raw evidence list is incomplete")
    for path, expected in zip(raw_paths, raw_hashes):
        if verify_immutable(path) != expected:
            raise RuntimeError("earnings raw evidence hash mismatch")
    raw_verified = True

    normalized_path = Path(source["normalized_path"])
    if verify_immutable(normalized_path) != source["normalized_data_sha256"]:
        raise RuntimeError("earnings normalized evidence hash mismatch")
    frame = pd.read_csv(normalized_path, dtype={"symbol": str})
    required = {"symbol", "forecast_eps_1"}
    if required - set(frame.columns):
        raise RuntimeError("earnings normalized evidence schema mismatch")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    if frame["symbol"].duplicated().any():
        raise RuntimeError("earnings normalized evidence has duplicate symbols")
    valid = set(
        frame.loc[pd.to_numeric(frame["forecast_eps_1"], errors="coerce").notna(), "symbol"]
    ) & universe
    actual_coverage = len(valid) / len(universe) if universe else 0.0
    metadata_consistent = (
        int(source.get("row_count", -1)) == len(frame)
        and abs(float(source.get("universe_coverage", -1)) - actual_coverage) <= 1e-12
    )
    if not metadata_consistent:
        raise RuntimeError("earnings receipt coverage/row count differs from recomputed evidence")
    normalized_verified = True
    coverage_verified = actual_coverage >= minimum_coverage
    return True, raw_verified, normalized_verified, coverage_verified, actual_coverage


def certify_observation(
    observation: dict,
    settings: OperationalSettings,
    *,
    lock_verifier: Callable | None = None,
    context_loader: Callable = load_pit_context,
    certified_at: datetime | None = None,
) -> ObservationCertification:
    """Recompute qualification from files; self-declared booleans are ignored."""
    failures: list[str] = []
    flags = {
        "reservation_verified": False,
        "official_lock_verified": False,
        "calendar_verified": False,
        "membership_verified": False,
        "industry_verified": False,
        "observation_receipt_verified": False,
        "earnings_source_verified": False,
        "raw_evidence_verified": False,
        "normalized_evidence_verified": False,
        "coverage_verified": False,
    }
    actual_coverage = 0.0

    try:
        locks = _verified_lock(settings, lock_verifier)
        approved_lock_hash = locks["v1r3_lock_sha256"]
        if observation.get("lock_sha256") != approved_lock_hash:
            raise RuntimeError("observation official lock hash mismatch")
        flags["official_lock_verified"] = True
    except Exception as error:
        failures.append(f"official_lock:{type(error).__name__}:{error}")
        locks = {}
        approved_lock_hash = ""

    try:
        _verify_reservation(observation, settings, approved_lock_hash)
        flags["reservation_verified"] = True
    except Exception as error:
        failures.append(f"reservation:{type(error).__name__}:{error}")

    try:
        _verify_observation_receipt(observation, settings)
        flags["observation_receipt_verified"] = True
    except Exception as error:
        failures.append(f"observation_receipt:{type(error).__name__}:{error}")

    try:
        from stockpilot.prospective_r2.calendar import load_verified_calendar

        calendar = load_verified_calendar(settings.calendar_path)
        target = pd.Timestamp(observation["target_date"]).normalize()
        if not calendar.start <= target <= calendar.end or not calendar.is_session(target):
            raise RuntimeError("target is outside the approved XSHG session set")
        if observation.get("trading_calendar_hash") != calendar.file_sha256:
            raise RuntimeError("observation trading calendar hash mismatch")
        # The calendar is trusted only because the verified V1r3/V1r2 lock binds it.
        if not flags["official_lock_verified"]:
            raise RuntimeError("calendar has no intact official lock root")
        flags["calendar_verified"] = True
    except Exception as error:
        failures.append(f"calendar:{type(error).__name__}:{error}")

    universe: set[str] = set()
    try:
        panel, proof = context_loader(observation["target_date"], settings)
        universe = set(panel["symbol"].astype(str).str.zfill(6))
        if observation.get("universe_hash") != universe_hash(universe):
            raise RuntimeError("recomputed PIT universe hash mismatch")
        if observation.get("pit_membership_snapshot_hash") != proof[
            "membership_snapshot_sha256"
        ]:
            raise RuntimeError("recomputed PIT membership snapshot hash mismatch")
        if sha256_file(proof["membership_source_path"]) != proof["membership_source_sha256"]:
            raise RuntimeError("PIT membership source provenance mismatch")
        flags["membership_verified"] = True
        if observation.get("pit_industry_mapping_hash") != proof["industry_mapping_sha256"]:
            raise RuntimeError("recomputed PIT industry mapping hash mismatch")
        if sha256_file(proof["industry_source_path"]) != proof["industry_source_sha256"]:
            raise RuntimeError("PIT industry source provenance mismatch")
        flags["industry_verified"] = True
    except Exception as error:
        failures.append(f"pit_context:{type(error).__name__}:{error}")

    try:
        (
            flags["earnings_source_verified"],
            flags["raw_evidence_verified"],
            flags["normalized_evidence_verified"],
            flags["coverage_verified"],
            actual_coverage,
        ) = _verify_source(
            observation,
            universe,
            settings.thresholds.minimum_expectation_coverage,
        )
        if not flags["coverage_verified"]:
            failures.append(
                f"coverage:actual={actual_coverage:.12g}<minimum="
                f"{settings.thresholds.minimum_expectation_coverage:.12g}"
            )
    except Exception as error:
        failures.append(f"earnings_source:{type(error).__name__}:{error}")

    qualifying = all(flags.values()) and not failures
    moment = certified_at or datetime.now(timezone.utc)
    return ObservationCertification(
        observation_id=str(observation.get("observation_id", "UNKNOWN")),
        target_date=str(observation.get("target_date", "UNKNOWN")),
        certified_at=moment.astimezone(timezone.utc).isoformat(),
        actual_universe_coverage=actual_coverage,
        qualifying_observation=qualifying,
        certification_failures=tuple(failures),
        **flags,
    )


def persist_certification(
    certification: ObservationCertification,
    settings: OperationalSettings,
) -> dict:
    target = settings.certifications_root / f"{certification.target_date}.json"
    payload = certification.to_dict()
    if target.exists():
        existing = read_verified_json(target)
        left = {key: value for key, value in existing.items() if key != "certified_at"}
        right = {key: value for key, value in payload.items() if key != "certified_at"}
        if canonical_json_bytes(left) != canonical_json_bytes(right):
            raise RuntimeError("immutable certification evidence changed")
        return existing | {
            "certification_path": target.as_posix(),
            "certification_sha256": verify_immutable(target),
        }
    digest = write_immutable_json(target, payload)
    return payload | {
        "certification_path": target.as_posix(),
        "certification_sha256": digest,
    }
