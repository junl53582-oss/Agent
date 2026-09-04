"""Make the frozen DAILY PIT + Gen2 ranking visible as a daily product."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from stockpilot.daily_pit import pipeline as daily_pipeline
from stockpilot.daily_pit import runtime as daily_runtime
from stockpilot.forward_evidence.monitor import (
    ForwardEvidenceSettings,
    _baseline_guard,
    _prediction_regime,
)
from stockpilot.prospective_r2.calendar import load_verified_calendar
from stockpilot.prospective_r2.integrity import (
    canonical_json_bytes,
    read_verified_json,
    verify_immutable,
    write_immutable_bytes,
    write_immutable_frame,
    write_immutable_json,
)
from stockpilot.provider_lineage_alignment import acquire_lineage_aligned_market

SHANGHAI = ZoneInfo("Asia/Shanghai")
MODEL_ID = "GEN2-LGBM-20D-SECTOR-BALANCED-TOP20"
PRODUCT_VERSION = "DAILY_STOCK_PREDICTION_V1"
EVIDENCE_LEVEL = "WEAK_REGIME_DEPENDENT"


class DailyPredictionError(RuntimeError):
    """Fail-closed error with one user-facing product status."""

    def __init__(self, status: str, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"{status}:{detail}" if detail else status)


@dataclass(frozen=True)
class DailyPredictionSettings:
    root: Path = Path("artifacts/daily_predictions/gen2")
    runtime_settings: daily_runtime.DailyRuntimeSettings = field(
        default_factory=daily_runtime.DailyRuntimeSettings
    )
    earliest_prediction_time: time = time(18, 30)
    model_id: str = MODEL_ID
    horizon_sessions: int = 20
    baseline_sha: str = "0119ca98e4db9156ec1008b8155fa4342131943d"
    verify_git_boundary: bool = True
    require_product_protocol: bool = True

    @property
    def prediction_root(self) -> Path:
        return self.root / "predictions"

    @property
    def attempt_root(self) -> Path:
        return self.root / "attempts"

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"

    @property
    def protocol_path(self) -> Path:
        return self.root / "product_protocol.json"


def _utc(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc).isoformat()


def _git_sha(settings: DailyPredictionSettings) -> str:
    if not settings.verify_git_boundary:
        return settings.baseline_sha
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def _baseline(settings: DailyPredictionSettings) -> dict[str, Any]:
    forward = ForwardEvidenceSettings(
        baseline_sha=settings.baseline_sha,
        runtime_settings=settings.runtime_settings,
        verify_git_boundary=settings.verify_git_boundary,
    )
    result = _baseline_guard(forward)
    if result["model_id"] != settings.model_id:
        raise DailyPredictionError("MODEL_INVALID", "OBSERVATION_MODEL_CHANGED")
    return result


def _expected_product_protocol(settings: DailyPredictionSettings) -> dict[str, Any]:
    return {
        "product_version": PRODUCT_VERSION,
        "phase": "DAILY_STOCK_PREDICTION_PRODUCT",
        "baseline_sha": settings.baseline_sha,
        "model_id": settings.model_id,
        "target_horizon_sessions": settings.horizon_sessions,
        "target_semantics": "cross-sectional relative-strength ranking",
        "top_k": {"primary": 20, "display": [10, 20]},
        "signal_strength": {
            "VERY_HIGH": "percentile >= 95",
            "HIGH": "90 <= percentile < 95",
            "MEDIUM": "70 <= percentile < 90",
            "LOW": "percentile < 70",
            "probability_semantics": False,
        },
        "model_evidence_level": EVIDENCE_LEVEL,
        "data_window": "18:30 Asia/Shanghai or later",
        "real_provider_confirmation_required": True,
        "feature_contribution_status": "NOT_PERSISTED_BY_FROZEN_RUNTIME",
        "research_only": True,
        "execution_authorized": False,
        "broker_requests_allowed": 0,
        "real_orders_allowed": False,
        "real_trades_allowed": False,
    }


def _verify_product_protocol(settings: DailyPredictionSettings) -> str | None:
    if not settings.require_product_protocol:
        return None
    try:
        digest = verify_immutable(settings.protocol_path)
        actual = read_verified_json(settings.protocol_path)
    except Exception as error:
        raise DailyPredictionError("LOCK_INVALID", f"PRODUCT_PROTOCOL:{error}") from error
    if actual != _expected_product_protocol(settings):
        raise DailyPredictionError("LOCK_INVALID", "PRODUCT_PROTOCOL_CHANGED")
    return digest


def _guard(
    settings: DailyPredictionSettings,
    verifier: Callable[[DailyPredictionSettings], dict[str, Any]] | None,
) -> dict[str, Any]:
    try:
        result = (verifier or _baseline)(settings)
    except DailyPredictionError:
        raise
    except Exception as error:
        raise DailyPredictionError("LOCK_INVALID", str(error)) from error
    if result.get("lock_status") != "VALID":
        raise DailyPredictionError("LOCK_INVALID", str(result.get("failures", [])))
    product_protocol = _verify_product_protocol(settings)
    if product_protocol is not None:
        result = {**result, "product_protocol_hash": product_protocol}
    return result


def _session_gate(target_date: str, now: datetime, settings: DailyPredictionSettings) -> None:
    try:
        sessions = load_verified_calendar(settings.runtime_settings.calendar_path).sessions()
    except Exception as error:
        raise DailyPredictionError("INPUT_INVALID", f"CALENDAR:{error}") from error
    target = pd.Timestamp(target_date)
    if target not in sessions:
        raise DailyPredictionError("NO_PREDICTION", "NOT_VERIFIED_TRADING_SESSION")
    local = now.astimezone(SHANGHAI)
    if target_date != local.date().isoformat():
        raise DailyPredictionError("NO_PREDICTION", "TARGET_DATE_MUST_BE_TODAY")
    if local.timetz().replace(tzinfo=None) < settings.earliest_prediction_time:
        raise DailyPredictionError(
            "DATA_WINDOW_NOT_OPEN", f"NEXT_ELIGIBLE_TIME={target_date}T18:30:00+08:00"
        )


def _core_paths(settings: DailyPredictionSettings, target_date: str) -> tuple[Path, Path, Path]:
    root = settings.runtime_settings.prediction_root / target_date
    return root / "prediction.json", root / "prediction.csv", root / "manifest.json"


def _verify_core_prediction(settings: DailyPredictionSettings, target_date: str) -> dict[str, Any]:
    receipt_path, ranking_path, manifest_path = _core_paths(settings, target_date)
    try:
        manifest_hash = verify_immutable(manifest_path)
        manifest = read_verified_json(manifest_path)
        receipt = read_verified_json(receipt_path)
        ranking_hash = verify_immutable(ranking_path)
    except Exception as error:
        raise DailyPredictionError("PREDICTION_CONFLICT", str(error)) from error
    if manifest.get("prediction.csv") != ranking_hash:
        raise DailyPredictionError("PREDICTION_CONFLICT", "CORE_RANKING_HASH_MISMATCH")
    if receipt.get("model_id") != settings.model_id:
        raise DailyPredictionError("MODEL_INVALID", "CORE_MODEL_ID_MISMATCH")
    if receipt.get("future_label_fields_present") is not False:
        raise DailyPredictionError("PREDICTION_CONFLICT", "FUTURE_LABEL_FIELD_PRESENT")
    return {
        "receipt": receipt,
        "ranking_path": ranking_path,
        "manifest_hash": manifest_hash,
    }


def _ensure_core_prediction(
    target_date: str,
    now: datetime,
    settings: DailyPredictionSettings,
    *,
    confirm_real_provider_acquisition: bool,
    acquisition_runner: Callable[..., dict[str, Any]],
    materializer: Callable[..., dict[str, Any]],
    sealer: Callable[..., dict[str, Any]],
    predictor: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    receipt_path, _, _ = _core_paths(settings, target_date)
    if receipt_path.is_file():
        return _verify_core_prediction(settings, target_date) | {
            "idempotent_core": True,
            "provider_requests": 0,
        }
    _session_gate(target_date, now, settings)
    daily_dir = settings.runtime_settings.daily_input_root / target_date
    provider_requests = 0
    if not (daily_dir / "market_manifest.json").is_file():
        if not confirm_real_provider_acquisition:
            raise DailyPredictionError(
                "PROVIDER_BLOCKED", "EXPLICIT_REAL_PROVIDER_CONFIRMATION_REQUIRED"
            )
        try:
            acquired = acquisition_runner(
                target_date,
                [],
                now=now,
                settings=settings.runtime_settings.pit_settings(),
            )
        except Exception as error:
            raise DailyPredictionError("PROVIDER_BLOCKED", str(error)) from error
        provider_requests = int(acquired.get("provider_requests_made", 0))
    if not (daily_dir / "manifest.json").is_file():
        try:
            materializer(target_date, settings=settings.runtime_settings.pit_settings())
        except Exception as error:
            raise DailyPredictionError("FEATURE_INVALID", str(error)) from error
    seal_path = settings.runtime_settings.input_seal_root / f"{target_date}.json"
    if not seal_path.is_file():
        try:
            sealer(target_date, now=now, settings=settings.runtime_settings)
        except Exception as error:
            raise DailyPredictionError("INPUT_INVALID", f"SEAL:{error}") from error
    try:
        gate = daily_runtime.preflight(target_date, now=now, settings=settings.runtime_settings)
    except Exception as error:
        raise DailyPredictionError("INPUT_INVALID", f"PREFLIGHT:{error}") from error
    if gate.get("daily_prediction_allowed") is not True:
        failures = str(gate.get("failures", []))
        status = "LOCK_INVALID" if "LOCK" in failures.upper() else "INPUT_INVALID"
        raise DailyPredictionError(status, f"PREFLIGHT:{failures}")
    try:
        predictor(target_date, now=now, settings=settings.runtime_settings)
    except Exception as error:
        raise DailyPredictionError("MODEL_INVALID", str(error)) from error
    return _verify_core_prediction(settings, target_date) | {
        "idempotent_core": False,
        "provider_requests": provider_requests,
    }


def _signal_strength(percentile: float) -> str:
    if percentile >= 95:
        return "VERY_HIGH"
    if percentile >= 90:
        return "HIGH"
    if percentile >= 70:
        return "MEDIUM"
    return "LOW"


def _display_frame(core_ranking: Path) -> pd.DataFrame:
    frame = pd.read_csv(core_ranking, dtype={"symbol": str})
    required = {
        "symbol",
        "score",
        "rank",
        "industry",
        "broad_sector",
        "selected_for_new_portfolio",
    }
    if required - set(frame):
        raise DailyPredictionError(
            "PREDICTION_CONFLICT", f"CORE_RANKING_SCHEMA:{sorted(required - set(frame))}"
        )
    if frame["symbol"].duplicated().any() or frame["rank"].duplicated().any():
        raise DailyPredictionError("PREDICTION_CONFLICT", "DUPLICATE_SYMBOL_OR_RANK")
    frame = frame.sort_values(["rank", "symbol"], kind="mergesort").reset_index(drop=True)
    count = len(frame)
    if count < 20:
        raise DailyPredictionError("PREDICTION_CONFLICT", "UNIVERSE_SMALLER_THAN_TOP20")
    frame["percentile"] = (
        (count - pd.to_numeric(frame["rank"], errors="raise")) / max(count - 1, 1) * 100
    )
    frame["eligible"] = True
    frame["selected_top10"] = frame["rank"].le(10)
    frame["selected_top20"] = frame["rank"].le(20)
    frame["selected_for_frozen_portfolio"] = frame["selected_for_new_portfolio"].astype(bool)
    frame["signal_strength"] = frame["percentile"].map(_signal_strength)
    return frame[
        [
            "symbol",
            "score",
            "rank",
            "percentile",
            "industry",
            "broad_sector",
            "eligible",
            "selected_top10",
            "selected_top20",
            "selected_for_frozen_portfolio",
            "signal_strength",
        ]
    ]


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "rank",
        "symbol",
        "score",
        "percentile",
        "broad_sector",
        "eligible",
        "signal_strength",
    ]
    header = "| Rank | Symbol | Score | Percentile | Sector | Eligibility | Signal Strength |"
    divider = "|---:|:---|---:|---:|:---|:---:|:---|"
    rows = [header, divider]
    for row in frame[columns].itertuples(index=False):
        rows.append(
            f"| {int(row.rank)} | {row.symbol} | {float(row.score):.6f} | "
            f"{float(row.percentile):.2f}% | {row.broad_sector} | "
            f"{'YES' if row.eligible else 'NO'} | {row.signal_strength} |"
        )
    return "\n".join(rows)


def _report(value: dict[str, Any], ranking: pd.DataFrame) -> str:
    distribution = value["prediction_distribution"]
    regime = value["market_regime_context"]
    return f"""# DAILY STOCK PREDICTION

## Prediction Date

{value["prediction_date"]}

## Target Horizon

20 trading sessions — a cross-sectional relative-strength ranking, not a next-day
direction forecast or target-price forecast.

## Model

`{value["model_id"]}`

## Status

`PREDICTION_AVAILABLE`

## Top 10 Predictions

{_markdown_table(ranking.head(10))}

## Top 20

{_markdown_table(ranking.head(20))}

Top20 is the frozen model-ranking display. The separate
`selected_for_frozen_portfolio` field preserves the existing sector-balanced,
20-session portfolio decision and is never execution authorization.

## Prediction Distribution

- Universe size: {distribution["universe_size"]}
- Eligible stock count: {distribution["eligible_count"]}
- Score min: {distribution["score_min"]:.6f}
- Score max: {distribution["score_max"]:.6f}
- Score median: {distribution["score_median"]:.6f}
- Score dispersion: {distribution["score_std"]:.6f}

## Market / Regime Context

- Market regime: {regime.get("market_regime", "NOT_AVAILABLE")}
- Volatility regime: {regime.get("volatility_regime", "NOT_AVAILABLE")}
- Positive 20D breadth: {regime.get("positive_20d_breadth", "NOT_AVAILABLE")}
- Classification uses prediction-time information only: {regime.get("classified_from_prediction_time_data_only", False)}

## Model Confidence / Evidence

- Historical Gen2 assessment: **WEAK / REGIME DEPENDENT**
- `VERY_HIGH/HIGH/MEDIUM/LOW` means relative daily model rank strength only.
- It is not an estimated probability of a price rise.
- No target price or expected percentage return is produced.

## Safety

- Research only: true
- Execution authorized: false
- Broker requests: 0
- Real orders: false
- Real trades: false

## Audit Identity

- Prediction ID: `{value["prediction_id"]}`
- Git SHA: `{value["git_sha"]}`
- Model hash: `{value["model_hash"]}`
- Feature manifest hash: `{value["feature_manifest_hash"]}`
- Input seal hash: `{value["input_seal_hash"]}`
"""


def _no_prediction_report(
    target_date: str,
    status: str,
    detail: str,
    now: datetime,
    settings: DailyPredictionSettings,
) -> dict[str, Any]:
    attempt_id = _utc(now).replace(":", "").replace("+", "_")
    directory = settings.attempt_root / target_date / attempt_id
    value = {
        "product_version": PRODUCT_VERSION,
        "prediction_date": target_date,
        "prediction_timestamp": _utc(now),
        "status": "NO_PREDICTION",
        "reason_code": status,
        "reason": detail,
        "next_eligible_prediction_time": (
            f"{target_date}T18:30:00+08:00" if status == "DATA_WINDOW_NOT_OPEN" else None
        ),
        "model_id": settings.model_id,
        "target_horizon_sessions": settings.horizon_sessions,
        "research_only": True,
        "execution_authorized": False,
        "broker_requests": 0,
        "real_orders": False,
        "real_trades": False,
    }
    json_hash = write_immutable_json(directory / "status.json", value)
    report = f"""# DAILY STOCK PREDICTION

## Prediction Date

{target_date}

## Status

`NO_PREDICTION`

## Reason

`{status}` — {detail}

## Next Eligible Prediction Time

{value["next_eligible_prediction_time"] or "WAIT_FOR_NEXT_VALID_INPUT_OR_SESSION"}

## Safety

- Research only: true
- Execution authorized: false
- Broker requests: 0
- Real orders: false
- Real trades: false
"""
    report_hash = write_immutable_bytes(
        directory / f"DAILY_STOCK_PREDICTION_REPORT_{target_date}.md",
        report.encode("utf-8"),
    )
    manifest_hash = write_immutable_json(
        directory / "manifest.json",
        {"status.json": json_hash, "prediction_report.md": report_hash},
    )
    return value | {
        "attempt_id": attempt_id,
        "artifact_path": str(directory),
        "manifest_hash": manifest_hash,
        "provider_requests": 0,
    }


def publish_prediction(
    target_date: str,
    core: dict[str, Any],
    baseline: dict[str, Any],
    now: datetime,
    settings: DailyPredictionSettings,
    *,
    regime_classifier: Callable[[ForwardEvidenceSettings, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    destination = settings.prediction_root / target_date
    existing = destination / "prediction.json"
    core_manifest_hash = core["manifest_hash"]
    if existing.is_file():
        value = read_verified_json(existing)
        if value.get("core_prediction_manifest_hash") != core_manifest_hash:
            raise DailyPredictionError("PREDICTION_CONFLICT", "PRODUCT_CORE_IDENTITY_CHANGED")
        return value | {"idempotent": True, "artifact_path": str(destination)}
    frame = _display_frame(core["ranking_path"])
    receipt = core["receipt"]
    model_hash = receipt.get("training_evidence", {}).get("model_signature")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise DailyPredictionError("MODEL_INVALID", "MODEL_SIGNATURE_MISSING")
    daily_dir = settings.runtime_settings.daily_input_root / target_date
    feature_manifest_hash = verify_immutable(daily_dir / "manifest.json")
    input_seal_hash = verify_immutable(
        settings.runtime_settings.input_seal_root / f"{target_date}.json"
    )
    source = read_verified_json(daily_dir / "source_receipt.json")
    prediction_id = f"DAILY-GEN2-{target_date}-{core_manifest_hash[:16]}"
    try:
        regime = (regime_classifier or _prediction_regime)(
            ForwardEvidenceSettings(
                baseline_sha=settings.baseline_sha,
                runtime_settings=settings.runtime_settings,
                verify_git_boundary=settings.verify_git_boundary,
            ),
            target_date,
        )
    except Exception as error:  # noqa: BLE001 - optional display context cannot change ranking
        regime = {"status": "NOT_AVAILABLE", "reason": f"{type(error).__name__}:{error}"}
    rows = frame.to_dict("records")
    value = {
        "product_version": PRODUCT_VERSION,
        "prediction_id": prediction_id,
        "prediction_date": target_date,
        "prediction_timestamp": receipt["created_at_utc"],
        "target_horizon_sessions": settings.horizon_sessions,
        "target_semantics": "20-session cross-sectional relative-strength ranking",
        "git_sha": baseline.get("git_sha", _git_sha(settings)),
        "baseline_sha": settings.baseline_sha,
        "model_id": receipt["model_id"],
        "model_hash": model_hash,
        "model_spec_hash": receipt["model_spec_hash"],
        "model_evidence_level": EVIDENCE_LEVEL,
        "feature_manifest_hash": feature_manifest_hash,
        "input_seal_hash": input_seal_hash,
        "effective_lock": baseline["effective_lock_identity"],
        "core_prediction_manifest_hash": core_manifest_hash,
        "universe_count": len(frame),
        "eligible_count": int(frame["eligible"].sum()),
        "predictions": rows,
        "top10": frame.head(10)["symbol"].tolist(),
        "top20": frame.head(20)["symbol"].tolist(),
        "top20_semantics": "raw model ranking display; frozen portfolio policy remains separate",
        "portfolio_action": receipt["portfolio_action"],
        "selected_for_frozen_portfolio": frame.loc[
            frame["selected_for_frozen_portfolio"], "symbol"
        ].tolist(),
        "prediction_distribution": {
            "universe_size": len(frame),
            "eligible_count": int(frame["eligible"].sum()),
            "score_min": float(frame["score"].min()),
            "score_max": float(frame["score"].max()),
            "score_median": float(frame["score"].median()),
            "score_std": float(frame["score"].std(ddof=1)),
        },
        "market_regime_context": regime,
        "provider_evidence": {
            "provider": source.get("provider_sources"),
            "acquisition_timestamp": source.get("acquired_at_utc"),
            "source_manifest_hash": verify_immutable(daily_dir / "market_manifest.json"),
            "source_hash": verify_immutable(daily_dir / "market.csv"),
            "target_date": source.get("target_date"),
        },
        "status": "PREDICTION_AVAILABLE",
        "research_only": True,
        "execution_authorized": False,
        "broker_requests": 0,
        "real_orders": False,
        "real_trades": False,
    }
    prediction_hash = write_immutable_json(existing, value)
    ranking_hash = write_immutable_frame(destination / "ranking.csv", frame, ["rank", "symbol"])
    top10_hash = write_immutable_frame(
        destination / "top10.csv", frame.head(10), ["rank", "symbol"]
    )
    top20_hash = write_immutable_frame(
        destination / "top20.csv", frame.head(20), ["rank", "symbol"]
    )
    report_hash = write_immutable_bytes(
        destination / f"DAILY_STOCK_PREDICTION_REPORT_{target_date}.md",
        _report(value, frame).encode("utf-8"),
    )
    manifest = {
        "prediction.json": prediction_hash,
        "ranking.csv": ranking_hash,
        "top10.csv": top10_hash,
        "top20.csv": top20_hash,
        "prediction_report.md": report_hash,
        "core_prediction_manifest_hash": core_manifest_hash,
    }
    manifest_hash = write_immutable_json(destination / "prediction_manifest.json", manifest)
    _atomic_json(
        settings.latest_path,
        {
            "prediction_date": target_date,
            "prediction_id": prediction_id,
            "artifact_path": str(destination),
            "prediction_manifest_hash": manifest_hash,
            "prediction_json_hash": prediction_hash,
        },
    )
    return value | {
        "idempotent": False,
        "artifact_path": str(destination),
        "prediction_manifest_hash": manifest_hash,
    }


def predict_daily(
    target_date: str,
    *,
    confirm_real_provider_acquisition: bool = False,
    now: datetime | None = None,
    settings: DailyPredictionSettings | None = None,
    baseline_verifier: Callable[[DailyPredictionSettings], dict[str, Any]] | None = None,
    acquisition_runner: Callable[..., dict[str, Any]] = acquire_lineage_aligned_market,
    materializer: Callable[..., dict[str, Any]] = daily_pipeline.materialize_features,
    sealer: Callable[..., dict[str, Any]] = daily_runtime.seal_inputs,
    predictor: Callable[..., dict[str, Any]] = daily_runtime.generate_prediction,
    regime_classifier: Callable[[ForwardEvidenceSettings, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    settings = settings or DailyPredictionSettings()
    now = now or datetime.now(timezone.utc)
    try:
        baseline = _guard(settings, baseline_verifier)
        core = _ensure_core_prediction(
            target_date,
            now,
            settings,
            confirm_real_provider_acquisition=confirm_real_provider_acquisition,
            acquisition_runner=acquisition_runner,
            materializer=materializer,
            sealer=sealer,
            predictor=predictor,
        )
        result = publish_prediction(
            target_date,
            core,
            baseline,
            now,
            settings,
            regime_classifier=regime_classifier,
        )
        return result | {
            "provider_requests": core["provider_requests"],
            "broker_requests": 0,
        }
    except DailyPredictionError as error:
        return _no_prediction_report(target_date, error.status, error.detail, now, settings)


def _load_product_prediction(directory: Path) -> dict[str, Any]:
    value = read_verified_json(directory / "prediction.json")
    manifest = read_verified_json(directory / "prediction_manifest.json")
    expected = {
        "prediction.json": directory / "prediction.json",
        "ranking.csv": directory / "ranking.csv",
        "top10.csv": directory / "top10.csv",
        "top20.csv": directory / "top20.csv",
        "prediction_report.md": directory / f"DAILY_STOCK_PREDICTION_REPORT_{directory.name}.md",
    }
    for key, path in expected.items():
        if verify_immutable(path) != manifest.get(key):
            raise DailyPredictionError("PREDICTION_CONFLICT", f"PRODUCT_MANIFEST:{key}")
    return value


def latest(settings: DailyPredictionSettings | None = None) -> dict[str, Any]:
    settings = settings or DailyPredictionSettings()
    if not settings.latest_path.is_file():
        return {
            "status": "NO_PREDICTION",
            "prediction_date": None,
            "reason": "NO_FORMAL_DAILY_PREDICTION_EXISTS",
            "reason_code": "NO_PREDICTION",
            "next_eligible_prediction_time": None,
            "model_id": settings.model_id,
            "target_horizon_sessions": settings.horizon_sessions,
            "research_only": True,
            "execution_authorized": False,
        }
    pointer = json.loads(settings.latest_path.read_text(encoding="utf-8"))
    directory = Path(pointer["artifact_path"])
    value = _load_product_prediction(directory)
    if value["prediction_id"] != pointer["prediction_id"]:
        raise DailyPredictionError("PREDICTION_CONFLICT", "LATEST_POINTER_ID_MISMATCH")
    return value | {"artifact_path": str(directory)}


def history(
    settings: DailyPredictionSettings | None = None, *, limit: int = 20
) -> list[dict[str, Any]]:
    settings = settings or DailyPredictionSettings()
    rows = []
    for directory in (
        sorted(settings.prediction_root.iterdir(), reverse=True)
        if settings.prediction_root.is_dir()
        else []
    ):
        if not directory.is_dir():
            continue
        value = _load_product_prediction(directory)
        maturity = settings.runtime_settings.settlement_root / directory.name / "settlement.json"
        rows.append(
            {
                "date": value["prediction_date"],
                "status": value["status"],
                "prediction_id": value["prediction_id"],
                "top1": value["top10"][0],
                "top5": value["top10"][:5],
                "universe": value["universe_count"],
                "model": value["model_id"],
                "maturity_status": "SETTLED" if maturity.is_file() else "PENDING_MATURITY",
            }
        )
        if len(rows) >= limit:
            break
    return rows


def explain(
    symbol: str,
    target_date: str,
    settings: DailyPredictionSettings | None = None,
) -> dict[str, Any]:
    settings = settings or DailyPredictionSettings()
    directory = settings.prediction_root / target_date
    if not directory.is_dir():
        return {
            "status": "NO_PREDICTION",
            "prediction_date": target_date,
            "symbol": str(symbol).zfill(6),
            "reason": "NO_FORMAL_DAILY_PREDICTION_EXISTS_FOR_DATE",
            "feature_contribution_status": "NOT_AVAILABLE",
        }
    value = _load_product_prediction(directory)
    normalized = str(symbol).zfill(6)
    row = next((item for item in value["predictions"] if item["symbol"] == normalized), None)
    if row is None:
        return {
            "status": "NO_PREDICTION",
            "prediction_date": target_date,
            "symbol": normalized,
            "reason": f"SYMBOL_NOT_IN_UNIVERSE:{normalized}",
            "feature_contribution_status": "NOT_AVAILABLE",
        }
    return {
        "prediction_date": target_date,
        "symbol": normalized,
        "rank": row["rank"],
        "score": row["score"],
        "percentile": row["percentile"],
        "signal_strength": row["signal_strength"],
        "feature_contribution_status": "FEATURE_CONTRIBUTION_NOT_AVAILABLE",
        "reason": "the frozen runtime did not persist per-row LightGBM pred_contrib values",
        "interpretation": "model score ranking, not a causal explanation or rise probability",
    }


def _format_prediction(value: dict[str, Any]) -> str:
    if value["status"] != "PREDICTION_AVAILABLE":
        return "\n".join(
            [
                "# STOCKPILOT DAILY PREDICTION",
                "",
                f"Date: {value.get('prediction_date')}",
                "Status: NO_PREDICTION",
                f"Reason: {value.get('reason_code')} - {value.get('reason')}",
                f"Next eligible time: {value.get('next_eligible_prediction_time')}",
                "Execution: DISABLED",
            ]
        )
    lines = [
        "# STOCKPILOT DAILY PREDICTION",
        "",
        f"Date: {value['prediction_date']}",
        "Status: PREDICTION_AVAILABLE",
        f"Model: {value['model_id']}",
        "Horizon: 20 trading sessions",
        "",
        "Top 10:",
    ]
    for item in value["predictions"][:10]:
        lines.append(
            f"{item['rank']}. {item['symbol']} — {item['score']:.6f} — "
            f"{item['percentile']:.2f}% — {item['signal_strength']}"
        )
    lines.extend(
        [
            "",
            f"Top20 artifact: {Path(value['artifact_path']) / 'top20.csv'}",
            f"Prediction ID: {value['prediction_id']}",
            "PIT: PASSED",
            f"Evidence Level: {EVIDENCE_LEVEL}",
            "Research Only: TRUE",
            "Execution: DISABLED",
        ]
    )
    return "\n".join(lines)


def _format_history(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "NO_FORMAL_DAILY_PREDICTIONS"
    lines = ["date       status                top1    universe maturity"]
    for row in rows:
        lines.append(
            f"{row['date']} {row['status']:<21} {row['top1']} "
            f"{row['universe']:<8} {row['maturity_status']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="StockPilot daily Gen2 stock prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("date")
    predict.add_argument("--confirm-real-provider-acquisition", action="store_true")
    predict.add_argument("--json", action="store_true")
    latest_parser = commands.add_parser("latest")
    latest_parser.add_argument("--json", action="store_true")
    history_parser = commands.add_parser("history")
    history_parser.add_argument("--limit", type=int, default=20)
    history_parser.add_argument("--json", action="store_true")
    explain_parser = commands.add_parser("explain")
    explain_parser.add_argument("symbol")
    explain_parser.add_argument("date")
    explain_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "predict":
        value: Any = predict_daily(
            args.date,
            confirm_real_provider_acquisition=args.confirm_real_provider_acquisition,
        )
        output = (
            json.dumps(value, ensure_ascii=False, indent=2)
            if args.json
            else _format_prediction(value)
        )
    elif args.command == "latest":
        value = latest()
        output = (
            json.dumps(value, ensure_ascii=False, indent=2)
            if args.json
            else _format_prediction(value)
        )
    elif args.command == "history":
        value = history(limit=args.limit)
        output = (
            json.dumps(value, ensure_ascii=False, indent=2) if args.json else _format_history(value)
        )
    else:
        value = explain(args.symbol, args.date)
        output = json.dumps(value, ensure_ascii=False, indent=2)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
