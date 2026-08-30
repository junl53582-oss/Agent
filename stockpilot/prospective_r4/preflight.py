from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stockpilot.prospective_r2.calendar import load_verified_calendar
from stockpilot.prospective_r2.integrity import (
    read_verified_json,
    sha256_file,
    verify_immutable,
    write_immutable_json,
)
from stockpilot.prospective_r2.sources import load_pit_context

from .config import OperationalSettings


SHANGHAI = ZoneInfo("Asia/Shanghai")
EXPECTED_V6_MODEL = "research_v6_sector_balanced_ensemble"
EXPECTED_V6_LOCK = "94edfc9e05bd30a58a14e7e11a988a1b7fb0d5358e462df1b20cb23dca4c0f4d"


class DailyPreflightBlocked(RuntimeError):
    def __init__(self, result: dict):
        super().__init__(result["status"])
        self.result = result


@dataclass(frozen=True)
class PreflightResult:
    version: str
    target_date: str
    shanghai_now: str
    shanghai_session: bool
    time_gate_passed: bool
    forward_market_ready: bool
    v6_ranking_ready: bool
    input_evidence_verified: bool
    frozen_inputs_intact: bool
    reservation_exists: bool
    daily_run_allowed: bool
    status: str
    failures: tuple[str, ...]
    provider_requests_made: int = 0
    execution_authorized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def _validate_session(target_date: str, now: datetime, settings: OperationalSettings) -> tuple[bool, str | None]:
    local = now.astimezone(SHANGHAI)
    target = pd.Timestamp(target_date).normalize()
    actual = pd.Timestamp(local.date()).normalize()
    if target < actual:
        return False, "HISTORICAL_DATE_FORBIDDEN"
    if target > actual:
        return False, "FUTURE_DATE_FORBIDDEN"
    calendar = load_verified_calendar(settings.calendar_path)
    if not calendar.is_session(target):
        return False, "NOT_VERIFIED_SHANGHAI_TRADING_SESSION"
    return True, None


def _validate_time(now: datetime, settings: OperationalSettings) -> tuple[bool, str | None]:
    local = now.astimezone(SHANGHAI)
    if local.timetz().replace(tzinfo=None) < settings.earliest_daily_run_time:
        if local.timetz().replace(tzinfo=None) < pd.Timestamp("15:00").time():
            return False, "TRADING_SESSION_NOT_CLOSED"
        return False, "UPSTREAM_DATA_WINDOW_NOT_OPEN"
    return True, None


def _paths(target_date: str, settings: OperationalSettings) -> tuple[Path, Path, Path, Path]:
    market = Path(settings.prediction_market_template.format(date=target_date))
    ranking = Path(settings.prediction_ranking_template.format(date=target_date))
    market_manifest = market.with_name(market.name.replace(".csv", ".manifest.json"))
    market_failures = market.with_name(market.name.replace(".csv", ".failures.csv"))
    return market, ranking, market_manifest, market_failures


def _inspect_market(target_date: str, settings: OperationalSettings) -> dict:
    market, _, manifest_path, failures_path = _paths(target_date, settings)
    if not market.exists():
        raise FileNotFoundError("FORWARD_MARKET_INPUT_NOT_READY")
    if not manifest_path.exists() or not failures_path.exists():
        raise RuntimeError("FORWARD_MARKET_SOURCE_EVIDENCE_MISSING")
    frame = pd.read_csv(market, dtype={"symbol": str})
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "amount"}
    if required - set(frame.columns):
        raise ValueError("FORWARD_MARKET_SCHEMA_INVALID")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("FORWARD_MARKET_DUPLICATE_DATE_SYMBOL")
    target = pd.Timestamp(target_date).normalize()
    if frame.empty or frame["date"].max() != target or (frame["date"] > target).any():
        raise ValueError("FORWARD_MARKET_DATE_COVERAGE_INVALID")
    numeric = frame[["open", "high", "low", "close", "volume", "amount"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy()).all() or (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("FORWARD_MARKET_PRICE_INVALID")
    panel, proof = load_pit_context(target_date, settings)
    expected = set(panel["symbol"].astype(str).str.zfill(6))
    current = set(frame.loc[frame["date"].eq(target), "symbol"])
    covered = len(expected & current)
    if covered < settings.minimum_forward_market_symbols or covered != len(expected):
        raise ValueError("FORWARD_MARKET_PIT_UNIVERSE_INCOMPLETE")
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    if (
        str(manifest.get("date_max")) != target_date
        or int(manifest.get("output_rows", -1)) != len(frame)
        or int(manifest.get("output_symbols", -1)) != frame["symbol"].nunique()
        or manifest.get("price_positive") is not True
    ):
        raise ValueError("FORWARD_MARKET_MANIFEST_MISMATCH")
    return {
        "path": market.as_posix(),
        "sha256": sha256_file(market),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "failures_path": failures_path.as_posix(),
        "failures_sha256": sha256_file(failures_path),
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()),
        "target_date_symbols": len(current),
        "pit_universe_symbols": len(expected),
        "pit_universe_covered": covered,
        "membership_snapshot_sha256": proof["membership_snapshot_sha256"],
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
    }


def _inspect_ranking(target_date: str, now: datetime, settings: OperationalSettings) -> dict:
    _, ranking, _, _ = _paths(target_date, settings)
    if not ranking.exists():
        raise FileNotFoundError("V6_RANKING_INPUT_NOT_READY")
    frame = pd.read_csv(ranking, dtype={"symbol": str})
    required = {
        "date", "symbol", "score", "pred_rank", "generated_at_utc", "model",
        "protocol_status", "execution_authorized", "plan_lock_sha256", "training_cutoff",
    }
    if required - set(frame.columns):
        raise ValueError("V6_RANKING_SCHEMA_INVALID")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    target = pd.Timestamp(target_date).normalize()
    if frame.empty or not frame["date"].eq(target).all():
        raise ValueError("V6_RANKING_DATE_MISMATCH")
    if frame["symbol"].duplicated().any() or frame["pred_rank"].duplicated().any():
        raise ValueError("V6_RANKING_DUPLICATE_SYMBOL_OR_RANK")
    scores = pd.to_numeric(frame["score"], errors="coerce")
    if not np.isfinite(scores).all():
        raise ValueError("V6_RANKING_SCORE_INVALID")
    generated = pd.to_datetime(frame["generated_at_utc"], utc=True, errors="raise")
    if generated.nunique() != 1 or generated.iloc[0] > pd.Timestamp(now).tz_convert("UTC"):
        raise ValueError("V6_RANKING_GENERATION_TIME_INVALID")
    if set(frame["model"].astype(str)) != {EXPECTED_V6_MODEL}:
        raise ValueError("V6_RANKING_MODEL_IDENTITY_INVALID")
    if set(frame["plan_lock_sha256"].astype(str).str.lower()) != {EXPECTED_V6_LOCK}:
        raise ValueError("V6_RANKING_LOCK_INVALID")
    if set(frame["protocol_status"].astype(str)) != {"retrospective_research"}:
        raise ValueError("V6_RANKING_PROTOCOL_INVALID")
    if _as_bool(frame["execution_authorized"]).isna().any() or _as_bool(frame["execution_authorized"]).any():
        raise ValueError("V6_RANKING_EXECUTION_FLAG_INVALID")
    if (pd.to_datetime(frame["training_cutoff"], errors="raise") >= target).any():
        raise ValueError("V6_RANKING_TRAINING_CUTOFF_INVALID")
    panel, proof = load_pit_context(target_date, settings)
    expected = set(panel["symbol"].astype(str).str.zfill(6))
    actual = set(frame["symbol"])
    if not actual <= expected:
        raise ValueError("V6_RANKING_OUTSIDE_PIT_UNIVERSE")
    coverage = len(actual) / len(expected) if expected else 0.0
    if len(actual) < settings.minimum_v6_ranking_symbols or coverage < settings.minimum_v6_ranking_coverage:
        raise ValueError("V6_RANKING_COVERAGE_INCOMPLETE")
    return {
        "path": ranking.as_posix(),
        "sha256": sha256_file(ranking),
        "rows": len(frame),
        "symbols": len(actual),
        "pit_universe_symbols": len(expected),
        "coverage": coverage,
        "membership_snapshot_sha256": proof["membership_snapshot_sha256"],
        "prediction_date": target_date,
        "generated_at_utc": generated.iloc[0].isoformat(),
        "training_cutoff": str(pd.to_datetime(frame["training_cutoff"]).max().date()),
        "model": EXPECTED_V6_MODEL,
        "plan_lock_sha256": EXPECTED_V6_LOCK,
    }


def seal_prediction_inputs(
    target_date: str,
    *,
    now: datetime | None = None,
    settings: OperationalSettings | None = None,
    lock_verifier: Callable | None = None,
) -> dict:
    """Validate already-produced local inputs and bind them; never calls a provider."""
    settings = settings or OperationalSettings()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    session, session_error = _validate_session(target_date, now, settings)
    time_ok, time_error = _validate_time(now, settings)
    if not session:
        raise RuntimeError(session_error)
    if not time_ok:
        raise RuntimeError(time_error)
    verifier = lock_verifier
    if verifier is None:
        from .freeze import verify_lock

        verifier = verify_lock
    locks = verifier(settings)
    if locks.get("frozen_inputs_intact") is not True:
        raise RuntimeError("FROZEN_INPUTS_NOT_INTACT")
    if (settings.attempts_root / f"{target_date}.json").exists():
        raise RuntimeError("DAILY_ATTEMPT_ALREADY_RESERVED")
    value = {
        "version": settings.version,
        "target_date": target_date,
        "sealed_at_utc": now.astimezone(timezone.utc).isoformat(),
        "forward_market": _inspect_market(target_date, settings),
        "v6_ranking": _inspect_ranking(target_date, now, settings),
        "v6_model_retrained": False,
        "provider_requests_made": 0,
        "execution_authorized": False,
    }
    path = settings.input_evidence_root / f"{target_date}.json"
    digest = write_immutable_json(path, value)
    return value | {"evidence_path": path.as_posix(), "evidence_sha256": digest}


def _verify_input_evidence(target_date: str, now: datetime, settings: OperationalSettings) -> tuple[bool, bool, bool, list[str]]:
    path = settings.input_evidence_root / f"{target_date}.json"
    failures: list[str] = []
    try:
        evidence = read_verified_json(path)
        verify_immutable(path)
        if evidence.get("target_date") != target_date or evidence.get("provider_requests_made") != 0:
            raise RuntimeError("PREDICTION_INPUT_EVIDENCE_METADATA_INVALID")
    except Exception as error:
        return False, False, False, [f"{type(error).__name__}:{error}"]
    market_ready = False
    ranking_ready = False
    try:
        market = _inspect_market(target_date, settings)
        if evidence.get("forward_market") != market:
            raise RuntimeError("FORWARD_MARKET_EVIDENCE_HASH_MISMATCH")
        market_ready = True
    except Exception as error:
        failures.append(f"{type(error).__name__}:{error}")
    try:
        ranking = _inspect_ranking(target_date, now, settings)
        if evidence.get("v6_ranking") != ranking:
            raise RuntimeError("V6_RANKING_EVIDENCE_HASH_MISMATCH")
        ranking_ready = True
    except Exception as error:
        failures.append(f"{type(error).__name__}:{error}")
    return market_ready and ranking_ready, market_ready, ranking_ready, failures


def run_preflight(
    *,
    target_date: str | None = None,
    now: datetime | None = None,
    settings: OperationalSettings | None = None,
    lock_verifier: Callable | None = None,
) -> dict:
    """Pure read-only gate.  It never reserves, writes, or constructs providers."""
    settings = settings or OperationalSettings()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(SHANGHAI)
    target_date = target_date or local.date().isoformat()
    failures: list[str] = []
    verifier = lock_verifier
    if verifier is None:
        from .freeze import verify_lock

        verifier = verify_lock
    try:
        locks = verifier(settings)
        frozen = locks.get("frozen_inputs_intact") is True
    except Exception as error:
        frozen = False
        failures.append(f"FROZEN_INPUTS_NOT_INTACT:{type(error).__name__}:{error}")
    session, session_error = _validate_session(target_date, now, settings)
    if session_error:
        failures.append(session_error)
    time_ok, time_error = _validate_time(now, settings)
    if time_error:
        failures.append(time_error)
    evidence_ok = market_ok = ranking_ok = False
    if session and time_ok and frozen:
        evidence_ok, market_ok, ranking_ok, input_failures = _verify_input_evidence(
            target_date, now, settings
        )
        failures.extend(input_failures)
    reservation_exists = (settings.attempts_root / f"{target_date}.json").exists()
    if reservation_exists:
        failures.append("DAILY_ATTEMPT_ALREADY_RESERVED")
    allowed = all((session, time_ok, frozen, evidence_ok, market_ok, ranking_ok)) and not reservation_exists
    if allowed:
        status = "DAILY_RUN_ALLOWED"
    elif not session:
        status = session_error or "SESSION_INVALID"
    elif not time_ok:
        status = time_error or "UPSTREAM_DATA_WINDOW_NOT_OPEN"
    elif not frozen:
        status = "FROZEN_INPUTS_NOT_INTACT"
    elif reservation_exists:
        status = "DAILY_ATTEMPT_ALREADY_RESERVED"
    else:
        status = "DAILY_PREFLIGHT_BLOCKED_PREDICTION_INPUT"
    return PreflightResult(
        version=settings.version,
        target_date=target_date,
        shanghai_now=local.isoformat(),
        shanghai_session=session,
        time_gate_passed=time_ok,
        forward_market_ready=market_ok,
        v6_ranking_ready=ranking_ok,
        input_evidence_verified=evidence_ok,
        frozen_inputs_intact=frozen,
        reservation_exists=reservation_exists,
        daily_run_allowed=allowed,
        status=status,
        failures=tuple(failures),
    ).to_dict()
