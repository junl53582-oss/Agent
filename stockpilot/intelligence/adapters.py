from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import CanonicalPrediction

ADAPTER_VERSION = "prediction-v1-phase1-adapters-1.0.0"
CSI300_UNIVERSE_ID = "CSI300_VALIDATED_SCOPE"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _path_text(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(value: Any) -> str | None:
    return None if _missing(value) else str(value)


def _float(value: Any) -> float | None:
    return None if _missing(value) else float(value)


def _int(value: Any) -> int | None:
    return None if _missing(value) else int(value)


def _bool(value: Any) -> bool:
    if _missing(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no"}:
            return False
        if normalized in {"true", "1", "yes"}:
            return True
        raise ValueError(f"invalid boolean value: {value}")
    return bool(value)


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction artifact missing columns {missing}: {source}")


def _fail_closed_execution(value: Any) -> bool:
    if _bool(value):
        raise ValueError("source artifact unexpectedly authorizes execution")
    return False


def _optional_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    return sha256_file(candidate) if candidate.exists() else None


def adapt_v6_snapshot(path: str | Path) -> tuple[CanonicalPrediction, ...]:
    """Read a V6 snapshot without enriching fields absent from V6 evidence."""
    source = Path(path)
    source_hash = sha256_file(source)
    frame = pd.read_csv(source, dtype={"symbol": str})
    _require_columns(
        frame,
        {
            "date",
            "symbol",
            "score",
            "pred_rank",
            "generated_at_utc",
            "model",
            "training_cutoff",
            "execution_authorized",
        },
        source,
    )
    records = []
    for row in frame.to_dict(orient="records"):
        records.append(
            CanonicalPrediction(
                symbol=str(row["symbol"]),
                stock_name=None,
                industry=None,
                prediction_date=str(row["date"]),
                prediction_timestamp=str(row["generated_at_utc"]),
                universe_id=CSI300_UNIVERSE_ID,
                eligibility_status="ELIGIBLE_IN_SOURCE_SNAPSHOT",
                data_coverage=None,
                feature_coverage=None,
                raw_rank_score=_float(row["score"]),
                return_1d_pred=None,
                return_5d_pred=None,
                return_20d_pred=None,
                up_prob_1d=None,
                up_prob_5d=None,
                up_prob_20d=None,
                rank_1d=None,
                rank_5d=None,
                rank_20d=None,
                market_rank=_int(row["pred_rank"]),
                industry_rank=None,
                stock_score=None,
                ranking_score=None,
                expected_return_score=None,
                probability_score=None,
                agreement_score=None,
                industry_score=None,
                regime_score=None,
                confidence_score=None,
                risk_score=None,
                positive_drivers=None,
                negative_drivers=None,
                risk_drivers=None,
                model_version=str(row["model"]),
                feature_version=None,
                training_cutoff=_text(row["training_cutoff"]),
                data_snapshot_hash=None,
                model_manifest_hash=None,
                production_prediction_ready=False,
                execution_authorized=_fail_closed_execution(row["execution_authorized"]),
                source_kind="V6",
                source_artifact_path=_path_text(source),
                source_artifact_hash=source_hash,
                source_snapshot_hash=source_hash,
                adapter_version=ADAPTER_VERSION,
            )
        )
    return tuple(records)


def _adapt_probability_snapshot(
    path: str | Path,
    *,
    source_kind: str,
    model_manifest_path: str | Path | None = None,
    feature_snapshot_path: str | Path | None = None,
) -> tuple[CanonicalPrediction, ...]:
    source = Path(path)
    source_hash = sha256_file(source)
    frame = pd.read_csv(source, dtype={"symbol": str})
    _require_columns(
        frame,
        {
            "date",
            "symbol",
            "name",
            "ranking_component",
            "p_up_1d",
            "p_up_5d",
            "p_up_20d",
            "rank_1d",
            "rank_5d",
            "rank_20d",
            "expected_return_5d",
            "expected_return_20d",
            "confidence_score",
            "prediction_ready",
            "model_version",
            "training_cutoff",
            "generated_at_utc",
            "execution_authorized",
        },
        source,
    )
    manifest_hash = _optional_hash(model_manifest_path)
    data_hash = _optional_hash(feature_snapshot_path)
    records = []
    for row in frame.to_dict(orient="records"):
        records.append(
            CanonicalPrediction(
                symbol=str(row["symbol"]),
                stock_name=_text(row["name"]),
                industry=None,
                prediction_date=str(row["date"]),
                prediction_timestamp=str(row["generated_at_utc"]),
                universe_id=CSI300_UNIVERSE_ID,
                eligibility_status="ELIGIBLE_IN_SOURCE_SNAPSHOT",
                data_coverage=None,
                feature_coverage=None,
                raw_rank_score=_float(row["ranking_component"]),
                return_1d_pred=None,
                return_5d_pred=_float(row["expected_return_5d"]),
                return_20d_pred=_float(row["expected_return_20d"]),
                up_prob_1d=_float(row["p_up_1d"]),
                up_prob_5d=_float(row["p_up_5d"]),
                up_prob_20d=_float(row["p_up_20d"]),
                rank_1d=_int(row["rank_1d"]),
                rank_5d=_int(row["rank_5d"]),
                rank_20d=_int(row["rank_20d"]),
                market_rank=_int(row["rank_5d"]),
                industry_rank=None,
                stock_score=None,
                ranking_score=None,
                expected_return_score=None,
                probability_score=None,
                agreement_score=None,
                industry_score=None,
                regime_score=None,
                confidence_score=_float(row["confidence_score"]),
                risk_score=None,
                positive_drivers=None,
                negative_drivers=None,
                risk_drivers=None,
                model_version=str(row["model_version"]),
                feature_version=None,
                training_cutoff=_text(row["training_cutoff"]),
                data_snapshot_hash=data_hash,
                model_manifest_hash=manifest_hash,
                production_prediction_ready=_bool(row["prediction_ready"]),
                execution_authorized=_fail_closed_execution(row["execution_authorized"]),
                source_kind=source_kind,
                source_artifact_path=_path_text(source),
                source_artifact_hash=source_hash,
                source_snapshot_hash=source_hash,
                adapter_version=ADAPTER_VERSION,
            )
        )
    return tuple(records)


def adapt_v30r1_snapshot(
    path: str | Path,
    *,
    model_manifest_path: str | Path | None = "artifacts/prediction_v30r1/models/manifest.json",
) -> tuple[CanonicalPrediction, ...]:
    return _adapt_probability_snapshot(
        path, source_kind="V30R1", model_manifest_path=model_manifest_path
    )


def adapt_forward_r2_snapshot(
    path: str | Path,
    *,
    model_manifest_path: str | Path | None = "artifacts/prediction_v30r1/models/manifest.json",
    feature_snapshot_path: str | Path | None = None,
) -> tuple[CanonicalPrediction, ...]:
    source = Path(path)
    if feature_snapshot_path is None:
        candidate = source.parent.parent / "features" / source.name
        feature_snapshot_path = candidate if candidate.exists() else None
    return _adapt_probability_snapshot(
        source,
        source_kind="V30R1_FORWARD_R2",
        model_manifest_path=model_manifest_path,
        feature_snapshot_path=feature_snapshot_path,
    )
