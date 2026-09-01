from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from stockpilot.daily_pit import sandbox
from stockpilot.prospective_r2.integrity import (
    read_verified_json,
    sha256_file,
    write_immutable_json,
)

TARGET = "2026-09-02"
REPLAY = Path("tests/fixtures/daily_pit_sandbox_replay_v1")


def _production_snapshot() -> dict:
    return sandbox._production_snapshot()


def _copy_replay(tmp_path: Path) -> Path:
    target = tmp_path / "replay"
    shutil.copytree(REPLAY, target)
    return target


def _rewrite_manifest(root: Path, changed_file: str) -> None:
    path = root / "replay_manifest.json"
    sidecar = path.with_suffix(path.suffix + ".sha256")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["files"][changed_file] = sha256_file(root / changed_file)
    path.unlink()
    sidecar.unlink()
    write_immutable_json(path, value)


def test_sandbox_replay_e2e_has_complete_lineage_and_zero_production_effects(
    tmp_path: Path,
) -> None:
    before = _production_snapshot()
    result = sandbox.run_sandbox_replay(
        TARGET, REPLAY, sandbox_root=tmp_path / "sandbox", run_id="e2e"
    )
    assert result["status"] == "OPERATIONAL_DRY_RUN_PASSED"
    assert all(value in (0, False) for value in result["side_effects"].values())
    manifest = read_verified_json(Path(result["sandbox_run_manifest"]))
    assert manifest["final_status"] == "OPERATIONAL_DRY_RUN_PASSED"
    assert manifest["market"]["provider_requests"] == 0
    assert manifest["features"]["columns"] == 71
    assert manifest["seal"]["seal_sha256"]
    assert manifest["prediction"]["prediction_rows"] == 10
    assert manifest["candidate"]["count"] == 10
    assert manifest["gate"]["state"] == "ACCEPTED_RESEARCH_ONLY"
    assert manifest["decision"]["action"] == "HOLD"
    assert manifest["execution_simulation"]["execution_authorized"] is False
    assert manifest["settlement_simulation"]["eligibility_evaluated"] is True
    assert manifest["settlement_simulation"]["production_settlement_created"] is False
    assert manifest["mode"] == "SIDE_EFFECT_FREE_SANDBOX_REPLAY"
    assert _production_snapshot() == before


def test_repeat_runs_have_identical_business_outputs(tmp_path: Path) -> None:
    first = sandbox.run_sandbox_replay(
        TARGET, REPLAY, sandbox_root=tmp_path / "sandbox", run_id="repeat-a"
    )
    second = sandbox.run_sandbox_replay(
        TARGET, REPLAY, sandbox_root=tmp_path / "sandbox", run_id="repeat-b"
    )
    assert first["run_id"] != second["run_id"]
    assert first["deterministic_outputs"] == second["deterministic_outputs"]


def test_production_root_is_rejected() -> None:
    with pytest.raises(sandbox.SandboxSafetyError, match="PRODUCTION_ROOT"):
        sandbox.validate_sandbox_root("data/prospective_gen2/predictions")


def test_sandbox_root_cannot_contain_production_roots(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(sandbox.SandboxSafetyError, match="MUST_NOT_CONTAIN"):
        sandbox.validate_sandbox_root("data")


def test_real_provider_backend_is_unreachable() -> None:
    evidence = sandbox.load_replay_evidence(REPLAY, TARGET)
    backend = sandbox.ReplayMarketBackend(evidence)
    with pytest.raises(sandbox.SandboxSafetyError, match="REAL_PROVIDER_FORBIDDEN"):
        backend.attempt_real_provider()
    assert backend.provider_requests == 0


def test_invalid_replay_source_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(sandbox.ReplayContractError, match="NOT_DIRECTORY"):
        sandbox.load_replay_evidence(tmp_path / "missing", TARGET)


def test_replay_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    replay = _copy_replay(tmp_path)
    with (replay / "market.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(sandbox.ReplayContractError, match="REPLAY_HASH_MISMATCH"):
        sandbox.load_replay_evidence(replay, TARGET)


def test_future_market_row_fails_closed(tmp_path: Path) -> None:
    replay = _copy_replay(tmp_path)
    market = pd.read_csv(replay / "market.csv", dtype={"symbol": str})
    future = market.iloc[[0]].copy()
    future["date"] = "2026-09-03"
    pd.concat([market, future], ignore_index=True).to_csv(
        replay / "market.csv", index=False, lineterminator="\n"
    )
    _rewrite_manifest(replay, "market.csv")
    with pytest.raises(sandbox.ReplayContractError, match="MARKET_PIT_VIOLATION"):
        sandbox.load_replay_evidence(replay, TARGET)


def test_feature_schema_mismatch_fails_closed(tmp_path: Path) -> None:
    replay = _copy_replay(tmp_path)
    panel = pd.read_parquet(replay / "panel.parquet").drop(columns=["momentum"])
    panel.to_parquet(replay / "panel.parquet", index=False, compression="zstd")
    _rewrite_manifest(replay, "panel.parquet")
    with pytest.raises(sandbox.ReplayContractError, match="REPLAY_FEATURE_INVALID"):
        sandbox.load_replay_evidence(replay, TARGET)


def test_duplicate_sandbox_run_id_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "sandbox"
    (root / "duplicate").mkdir(parents=True)
    with pytest.raises(sandbox.SandboxSafetyError, match="RUN_ALREADY_EXISTS"):
        sandbox.run_sandbox_replay(TARGET, REPLAY, sandbox_root=root, run_id="duplicate")


def test_invalid_run_id_cannot_escape_root(tmp_path: Path) -> None:
    with pytest.raises(sandbox.SandboxSafetyError, match="RUN_ID_INVALID"):
        sandbox.run_sandbox_replay(
            TARGET, REPLAY, sandbox_root=tmp_path / "sandbox", run_id="../escape"
        )


def test_effective_lock_mismatch_fails_before_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sandbox.daily_runtime,
        "verify_effective_daily_runtime_freeze",
        lambda: {"effective_daily_input_lock_intact": False, "failures": ["011"]},
    )
    with pytest.raises(sandbox.SandboxSafetyError, match="EFFECTIVE_LOCK_INVALID"):
        sandbox.run_sandbox_replay(
            TARGET, REPLAY, sandbox_root=tmp_path / "sandbox", run_id="lock-failure"
        )
    assert not (tmp_path / "sandbox").exists()


def test_scorer_failure_stays_inside_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = _production_snapshot()

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("INJECTED_SCORER_FAILURE")

    monkeypatch.setattr(sandbox, "_sandbox_train_and_score", fail)
    root = tmp_path / "sandbox"
    with pytest.raises(RuntimeError, match="INJECTED_SCORER_FAILURE"):
        sandbox.run_sandbox_replay(TARGET, REPLAY, sandbox_root=root, run_id="scorer-failure")
    assert (root / "scorer-failure/attempts/2026-09-02.json").is_file()
    assert _production_snapshot() == before


def test_candidate_gate_rejects_execution_authorization() -> None:
    prediction = pd.DataFrame(
        {
            "symbol": ["000001"],
            "score": [1.0],
            "rank": [1],
            "selected_for_new_portfolio": [True],
            "portfolio_weight": [1.0],
        }
    )
    with pytest.raises(sandbox.SandboxSafetyError, match="CANDIDATE_GATE_REJECTED"):
        sandbox._candidate_gate_decision(
            prediction,
            {
                "research_only": True,
                "execution_authorized": True,
                "portfolio_action": "REBALANCE",
            },
            "prediction",
            {"sandbox_012_lock_sha256": "lock"},
            "2026-09-02T19:00:00+08:00",
        )


def test_broker_adapter_invocation_is_forbidden() -> None:
    broker = sandbox.ForbiddenBrokerAdapter()
    with pytest.raises(sandbox.SandboxSafetyError, match="BROKER_ADAPTER_FORBIDDEN"):
        broker.submit({"symbol": "000001"})
    assert broker.request_count == 0


@pytest.mark.parametrize(
    "field,production",
    [
        ("daily_input_root", Path("data/prospective_gen2/daily_inputs")),
        ("input_seal_root", Path("data/prospective_gen2/input_seals")),
        ("reservation_root", Path("data/prospective_gen2/_prediction_attempts")),
        ("settlement_root", Path("data/prospective_gen2/settlements")),
        ("prediction_root", Path("data/prospective_gen2/predictions")),
    ],
)
def test_runtime_production_writer_roots_are_rejected(
    field: str, production: Path, tmp_path: Path
) -> None:
    evidence = sandbox.load_replay_evidence(REPLAY, TARGET)
    paths = sandbox._paths(sandbox.validate_sandbox_root(tmp_path / "sandbox"), "root-check")
    settings = sandbox._runtime_settings(evidence, paths)
    unsafe = replace(settings, **{field: production})
    with pytest.raises(sandbox.SandboxSafetyError, match="RUNTIME_ROOT"):
        sandbox._assert_runtime_roots(unsafe, paths)


def test_invalid_sandbox_seal_fails_preflight(tmp_path: Path) -> None:
    evidence = sandbox.load_replay_evidence(REPLAY, TARGET)
    root = sandbox.validate_sandbox_root(tmp_path / "sandbox")
    paths = sandbox._paths(root, "invalid-seal")
    paths.run_dir.mkdir(parents=True)
    now = pd.Timestamp(evidence.manifest["as_of_timestamp"]).to_pydatetime()
    market, pit = sandbox._market_stage(evidence, paths, TARGET, now, "invalid-seal")
    sandbox._materialize_replayed_features(evidence, paths, pit, TARGET, "invalid-seal", market)
    settings = sandbox._runtime_settings(evidence, paths)
    seal = sandbox.daily_runtime.seal_inputs(TARGET, now=now, settings=settings)
    Path(seal["seal_path"]).chmod(0o666)
    Path(seal["seal_path"]).write_bytes(b"invalid")
    result = sandbox.daily_runtime.preflight(TARGET, now=now, settings=settings)
    assert result["daily_prediction_allowed"] is False


def test_sandbox_contract_and_parent_011_are_intact() -> None:
    result = sandbox.verify_sandbox_contract()
    assert result["sandbox_012_lock_intact"] is True
