from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from stockpilot.intelligence.adapters import (
    CSI300_UNIVERSE_ID,
    adapt_forward_r2_snapshot,
    adapt_v6_snapshot,
    adapt_v30r1_snapshot,
    sha256_file,
)
from stockpilot.intelligence.schema import (
    PREDICTION_SCHEMA_VERSION,
    canonical_json_bytes,
    prediction_schema_definition,
)
from stockpilot.intelligence.snapshot import (
    build_daily_snapshot,
    canonical_snapshot_bytes,
    derive_top_k,
    write_immutable_daily_snapshot,
)

V6 = Path("artifacts/research_v6/live/predictions/2026-08-28.csv")
V30R1 = Path("artifacts/prediction_v30r1/live/predictions/2026-08-21.csv")
FORWARD_R2 = Path("artifacts/prediction_forward/v30r1_r2/predictions/2026-08-28.csv")
V1R4_LOCK = Path("artifacts/prospective_alpha_v1r4/plan.lock.json")
GEN2_LOCK = Path(
    "artifacts/research_challenger/gen02/experiments/007_human_readjudication/"
    "experiments/009_prospective_runtime_hardening/experiments/"
    "010r3_runtime_self_verification_activation/plan.lock.json"
)


def test_v6_adapter_preserves_raw_values() -> None:
    source = pd.read_csv(V6, dtype={"symbol": str}).iloc[0]
    record = adapt_v6_snapshot(V6)[0]
    assert record.symbol == source["symbol"].zfill(6)
    assert record.raw_rank_score == pytest.approx(source["score"])
    assert record.market_rank == int(source["pred_rank"])
    assert record.model_version == source["model"]


def test_v30r1_adapter_preserves_probabilities() -> None:
    source = pd.read_csv(V30R1, dtype={"symbol": str}).iloc[0]
    record = adapt_v30r1_snapshot(V30R1)[0]
    assert record.up_prob_1d == pytest.approx(source["p_up_1d"])
    assert record.up_prob_5d == pytest.approx(source["p_up_5d"])
    assert record.up_prob_20d == pytest.approx(source["p_up_20d"])
    assert record.return_5d_pred == pytest.approx(source["expected_return_5d"])
    assert record.return_20d_pred == pytest.approx(source["expected_return_20d"])


def test_forward_adapter_preserves_data_snapshot_provenance() -> None:
    record = adapt_forward_r2_snapshot(FORWARD_R2)[0]
    features = Path("artifacts/prediction_forward/v30r1_r2/features/2026-08-28.csv")
    assert record.source_kind == "V30R1_FORWARD_R2"
    assert record.data_snapshot_hash == sha256_file(features)
    assert record.source_snapshot_hash == sha256_file(FORWARD_R2)


def test_missing_field_is_null_not_fabricated() -> None:
    v6 = adapt_v6_snapshot(V6)[0]
    v30r1 = adapt_v30r1_snapshot(V30R1)[0]
    assert v6.stock_name is None
    assert v6.industry is None
    assert v6.up_prob_1d is None
    assert v30r1.return_1d_pred is None
    assert v30r1.industry_rank is None
    assert v30r1.stock_score is None
    assert v30r1.risk_score is None


def test_prediction_schema_deterministic() -> None:
    first = adapt_v30r1_snapshot(V30R1)[0]
    second = adapt_v30r1_snapshot(V30R1)[0]
    assert canonical_json_bytes(first.to_dict()) == canonical_json_bytes(second.to_dict())
    assert prediction_schema_definition() == prediction_schema_definition()


def test_prediction_hash_deterministic() -> None:
    first = adapt_forward_r2_snapshot(FORWARD_R2)[0]
    second = adapt_forward_r2_snapshot(FORWARD_R2)[0]
    assert first.prediction_hash == second.prediction_hash
    assert len(first.prediction_hash) == 64


def test_source_provenance_preserved() -> None:
    record = adapt_v6_snapshot(V6)[0]
    assert record.source_artifact_path.endswith(V6.as_posix())
    assert record.source_artifact_hash == sha256_file(V6)
    assert record.adapter_version
    assert record.schema_version == PREDICTION_SCHEMA_VERSION


@pytest.mark.parametrize(
    "adapter,path",
    [
        (adapt_v6_snapshot, V6),
        (adapt_v30r1_snapshot, V30R1),
        (adapt_forward_r2_snapshot, FORWARD_R2),
    ],
)
def test_adapter_is_read_only(adapter, path: Path) -> None:
    before = sha256_file(path)
    adapter(path)
    assert sha256_file(path) == before


def test_daily_snapshot_immutable(tmp_path: Path) -> None:
    records = adapt_v6_snapshot(V6)
    snapshot = build_daily_snapshot(records)
    target = tmp_path / "2026-08-28.json"
    wrote, digest = write_immutable_daily_snapshot(snapshot, target)
    assert wrote
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert write_immutable_daily_snapshot(snapshot, target) == (False, digest)

    changed_record = replace(records[0], raw_rank_score=-99.0, prediction_hash="")
    changed = build_daily_snapshot((changed_record, *records[1:]))
    with pytest.raises(FileExistsError, match="immutable"):
        write_immutable_daily_snapshot(changed, target)


@pytest.mark.parametrize("k", [10, 20, 50])
def test_topk_is_subset_of_same_snapshot(k: int) -> None:
    snapshot = build_daily_snapshot(adapt_forward_r2_snapshot(FORWARD_R2))
    derived = derive_top_k(snapshot, k)
    assert derived == snapshot.records[:k]
    assert {row.prediction_hash for row in derived}.issubset(
        {row.prediction_hash for row in snapshot.records}
    )
    assert snapshot.snapshot_hash == build_daily_snapshot(snapshot.records).snapshot_hash


def test_universe_not_claimed_all_a_share() -> None:
    records = (
        *adapt_v6_snapshot(V6)[:1],
        *adapt_v30r1_snapshot(V30R1)[:1],
        *adapt_forward_r2_snapshot(FORWARD_R2)[:1],
    )
    assert {record.universe_id for record in records} == {CSI300_UNIVERSE_ID}
    assert all("ALL_A_SHARE" not in record.universe_id for record in records)


@pytest.mark.parametrize("protected", [V6, V1R4_LOCK, GEN2_LOCK])
def test_protected_evidence_unchanged(protected: Path) -> None:
    before = sha256_file(protected)
    adapt_v6_snapshot(V6)
    adapt_v30r1_snapshot(V30R1)
    adapt_forward_r2_snapshot(FORWARD_R2)
    assert sha256_file(protected) == before


def test_execution_authorized_false() -> None:
    records = (
        *adapt_v6_snapshot(V6),
        *adapt_v30r1_snapshot(V30R1),
        *adapt_forward_r2_snapshot(FORWARD_R2),
    )
    assert records
    assert not any(record.execution_authorized for record in records)


def test_schema_artifact_is_product_only() -> None:
    payload = json.loads(Path("artifacts/prediction_v1/schema/v1.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == PREDICTION_SCHEMA_VERSION
    assert payload["scope"] == "PRODUCT_INTERFACE_ONLY_NOT_MODEL_PROMOTION_EVIDENCE"
    assert payload["missing_value_policy"] == "null_not_fabricated"
    assert payload["execution_authorized"] is False


def test_snapshot_bytes_deterministic() -> None:
    records = adapt_forward_r2_snapshot(FORWARD_R2)
    first = build_daily_snapshot(records)
    second = build_daily_snapshot(reversed(records))
    assert canonical_snapshot_bytes(first) == canonical_snapshot_bytes(second)


def test_versioned_api_keeps_legacy_routes() -> None:
    from stockpilot.api import app

    paths = {route.path for route in app.routes}
    assert "/predictions/latest" in paths
    assert "/predictions/{symbol}" in paths
    assert "/api/v1/predictions/latest" in paths
    assert "/api/v1/predictions/top/{k}" in paths
    assert "/api/v1/predictions/symbol/{symbol}" in paths
