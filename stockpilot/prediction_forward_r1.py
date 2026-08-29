from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stockpilot.data import load_panel
from stockpilot.membership import load_membership_history
from stockpilot.prediction.freeze import verify_validation_lock
from stockpilot.prediction_audit import verify_result_lock
from stockpilot.prediction_forward import (
    ForwardPredictionSettings,
    _generate_from_panel,
    _immutable_json,
    _sha256,
    build_latest_pit_feature_panel,
    compare_feature_panel,
    stitch_hfq_market,
)
from stockpilot.prediction.storage import write_immutable_prediction_snapshot, write_latest_metadata


@dataclass(frozen=True)
class ForwardR1Settings(ForwardPredictionSettings):
    version: str = "V30r1-forward-r1"
    artifact_dir: Path = Path("artifacts/prediction_forward/v30r1_r1")
    failed_parent_dir: Path = Path("artifacts/prediction_forward/v30r1")


def attach_optional_ranking(
    current: pd.DataFrame,
    ranking_path: str | Path,
    as_of: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """Attach V6 only as a candidate-score aid; probability heads never consume it."""
    as_of_date = pd.Timestamp(as_of).normalize()
    ranking_path = Path(ranking_path)
    ranking = pd.read_csv(ranking_path, dtype={"symbol": str})
    ranking["symbol"] = ranking["symbol"].astype(str).str.zfill(6)
    ranking["date"] = pd.to_datetime(ranking["date"]).dt.normalize()
    ranking = ranking[ranking["date"].eq(as_of_date)].copy()
    if ranking.empty or "score" not in ranking:
        raise RuntimeError("same-date V6 auxiliary ranking evidence is missing")
    ranking["v6_ranking_component"] = pd.to_numeric(
        ranking["score"], errors="coerce"
    ).rank(pct=True, method="average")
    output = current.drop(columns="ranking_component", errors="ignore").merge(
        ranking[["symbol", "v6_ranking_component"]],
        on="symbol", how="left", validate="one_to_one",
    )
    matched = int(output["v6_ranking_component"].notna().sum())
    output["ranking_component"] = output.pop("v6_ranking_component").fillna(0.5)
    preferred = [
        "date", "symbol", "name", "open", "close", "broad_sector", "regime",
        "volatility_20", "ranking_component",
    ]
    leading = [column for column in preferred if column in output]
    output = output[[*leading, *[column for column in output if column not in leading]]]
    return output, {
        "source": str(ranking_path),
        "source_date": str(as_of_date.date()),
        "current_rows": len(output),
        "matched_rows": matched,
        "coverage": matched / len(output),
        "missing_rows": len(output) - matched,
        "missing_policy": "fixed neutral 0.5 for candidate_score only",
        "used_by_probability_heads": False,
        "used_by_expected_return_heads": False,
    }


def create_r1_plan_lock(settings: ForwardR1Settings | None = None) -> dict:
    settings = settings or ForwardR1Settings()
    target = settings.artifact_dir / "plan.lock.json"
    if target.exists():
        raise RuntimeError(f"forward-r1 plan lock already exists: {target}")
    files = [
        settings.artifact_dir / "protocol.json",
        Path("stockpilot/prediction_forward_r1.py"),
        Path("tests/test_prediction_forward_r1.py"),
        settings.failed_parent_dir / "plan.lock.json",
        settings.parent_root / "validation.lock.json",
        settings.parent_v30_root / "validation.lock.json",
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise RuntimeError("cannot freeze forward-r1 plan: " + ", ".join(missing))
    payload = {
        "version": settings.version,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "inference_only_auxiliary_ranking_neutral_fallback",
        "parent_failure_preserved": True,
        "probability_heads_unchanged": True,
        "production_prediction_ready_may_not_be_promoted": True,
        "execution_authorized": False,
        "files": {path.as_posix(): _sha256(path) for path in files},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload | {"lock_sha256": _sha256(target)}


def verify_r1_plan_lock(settings: ForwardR1Settings | None = None) -> dict:
    settings = settings or ForwardR1Settings()
    target = settings.artifact_dir / "plan.lock.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    mismatches = [
        name for name, expected in payload["files"].items()
        if not Path(name).exists() or _sha256(Path(name)) != expected
    ]
    return {"intact": not mismatches, "mismatches": mismatches, "lock_sha256": _sha256(target)}


def run_forward_r1(
    incremental_market_path: str | Path,
    as_of: str | pd.Timestamp,
    *,
    ranking_path: str | Path,
    settings: ForwardR1Settings | None = None,
) -> dict:
    settings = settings or ForwardR1Settings()
    as_of_date = pd.Timestamp(as_of).normalize()
    locks = {
        "v30": verify_validation_lock(),
        "v30r1": verify_result_lock(settings.parent_root),
        "failed_forward_parent": verify_r1_parent_lock(settings),
        "forward_r1": verify_r1_plan_lock(settings),
    }
    if not all(lock["intact"] for lock in locks.values()):
        raise RuntimeError(f"a frozen parent or forward-r1 input is not intact: {locks}")
    frozen = load_panel(settings.frozen_market_path)
    cutoff = pd.to_datetime(frozen["date"]).max()
    incremental_path = Path(incremental_market_path)
    ranking_path = Path(ranking_path)
    combined, market_audit = stitch_hfq_market(
        frozen, load_panel(incremental_path), load_membership_history(settings.membership_path),
        cutoff=cutoff, as_of=as_of_date, settings=settings,
    )
    parity_panel, parity_pit = build_latest_pit_feature_panel(
        combined[combined["date"] <= cutoff], cutoff, settings=settings,
    )
    parity = compare_feature_panel(
        parity_panel, settings.parent_v30_root / "models" / "latest_feature_panel.csv",
    )
    if not parity["passed"]:
        raise RuntimeError(f"frozen feature parity failed: {parity}")
    current, pit_audit = build_latest_pit_feature_panel(combined, as_of_date, settings=settings)
    current, ranking_audit = attach_optional_ranking(current, ranking_path, as_of_date)
    feature_path = settings.feature_dir / f"{as_of_date.date()}.csv"
    write_immutable_prediction_snapshot(current, feature_path)
    audit_path = settings.audit_dir / f"{as_of_date.date()}.json"
    generated_at = (
        json.loads(audit_path.read_text(encoding="utf-8"))["generated_at_utc"]
        if audit_path.exists() else datetime.now(timezone.utc).isoformat()
    )
    audit = {
        "version": settings.version,
        "generated_at_utc": generated_at,
        "locks": locks,
        "input_hashes": {
            str(incremental_path): _sha256(incremental_path),
            str(ranking_path): _sha256(ranking_path),
            str(settings.frozen_market_path): _sha256(settings.frozen_market_path),
            str(settings.membership_path): _sha256(settings.membership_path),
            str(settings.fundamental_path): _sha256(settings.fundamental_path),
            str(settings.industry_path): _sha256(settings.industry_path),
        },
        "market_stitch": market_audit,
        "frozen_feature_parity": parity,
        "frozen_feature_pit": parity_pit,
        "latest_feature_pit": pit_audit,
        "auxiliary_ranking": ranking_audit,
        "execution_authorized": False,
    }
    _immutable_json(audit_path, audit)
    result = _generate_from_panel(current, combined, as_of_date, settings)
    result.update({
        "market_audit_passed": True,
        "feature_parity_passed": True,
        "pit_audit_passed": True,
        "auxiliary_ranking_coverage": ranking_audit["coverage"],
        "auxiliary_ranking_neutral_fallback_rows": ranking_audit["missing_rows"],
    })
    write_latest_metadata(settings.artifact_dir / "latest.json", result)
    return result


def verify_r1_parent_lock(settings: ForwardR1Settings | None = None) -> dict:
    settings = settings or ForwardR1Settings()
    path = settings.failed_parent_dir / "plan.lock.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mismatches = [
        name for name, expected in payload["files"].items()
        if not Path(name).exists() or _sha256(Path(name)) != expected
    ]
    return {"intact": not mismatches, "mismatches": mismatches, "lock_sha256": _sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen V30r1 forward-r1 inference")
    parser.add_argument("--market", required=True)
    parser.add_argument("--as-of", required=True, dest="as_of")
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    result = run_forward_r1(args.market, args.as_of, ranking_path=args.ranking)
    frame = pd.read_csv(result["snapshot_path"], dtype={"symbol": str}).head(args.limit)
    columns = [
        "rank_5d", "symbol", "name", "p_up_1d", "p_up_5d", "p_up_20d",
        "expected_return_5d", "expected_return_20d", "confidence_level",
        "prediction_ready", "execution_authorized",
    ]
    print(frame[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
