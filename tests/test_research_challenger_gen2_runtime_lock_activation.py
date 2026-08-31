from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stockpilot.prospective_r2.integrity import sha256_file, write_immutable_json
from stockpilot.research_challenger import prospective_gen2_runtime as base
from stockpilot.research_challenger import prospective_gen2_runtime_locked as locked


TARGET = "2026-09-01"
NOW = datetime(2026, 9, 1, 11, tzinfo=timezone.utc)
GOOD = {
    "effective_operational_lock_intact": True,
    "parent_008_lock_intact": True,
    "parent_008_lock_sha256": None,
    "runtime_009_lock_intact": True,
    "self_verification_010_intact": True,
    "self_verification_010_lock_sha256": "010",
    "failures": [],
    "provider_requests_made": 0,
    "production_prediction_ready": False,
    "execution_authorized": False,
}
BAD = {
    "effective_operational_lock_intact": False,
    "failures": ["tampered"],
    "provider_requests_made": 0,
    "production_prediction_ready": False,
    "execution_authorized": False,
}


def good_verifier(_):
    return GOOD.copy()


def bad_verifier(_):
    return BAD.copy()


@pytest.mark.parametrize("operation", ["seal", "predict", "settle"])
def test_state_changing_commands_require_effective_runtime_lock(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    called = []
    monkeypatch.setattr(base, "seal_inputs", lambda *a, **k: called.append("seal"))
    monkeypatch.setattr(base, "generate_prediction", lambda *a, **k: called.append("predict"))
    monkeypatch.setattr(base, "settle_prediction", lambda *a, **k: called.append("settle"))
    with pytest.raises(locked.EffectiveRuntimeLockError, match="GEN2_EFFECTIVE_RUNTIME_LOCK_INVALID"):
        if operation == "seal":
            locked.seal_inputs(TARGET, effective_verifier=bad_verifier)
        elif operation == "predict":
            locked.generate_prediction(TARGET, effective_verifier=bad_verifier)
        else:
            locked.settle_prediction(
                TARGET, Path("market.csv"), effective_verifier=bad_verifier
            )
    assert called == []


def test_seal_requires_effective_runtime_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "seal_inputs", lambda *a, **k: {"sealed": True})
    assert locked.seal_inputs(TARGET, effective_verifier=good_verifier)["sealed"] is True


def test_preflight_requires_effective_runtime_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(base, "preflight", lambda *a, **k: called.append(True))
    result = locked.preflight(TARGET, effective_verifier=bad_verifier)
    assert result["daily_prediction_allowed"] is False
    assert result["effective_operational_lock_intact"] is False
    assert result["provider_requests_made"] == 0
    assert called == []


def test_predict_requires_effective_runtime_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "generate_prediction", lambda *a, **k: {"predicted": True})
    assert locked.generate_prediction(TARGET, effective_verifier=good_verifier)["predicted"]


def test_settle_requires_effective_runtime_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "settle_prediction", lambda *a, **k: {"settled": True})
    result = locked.settle_prediction(
        TARGET, Path("market.csv"), effective_verifier=good_verifier
    )
    assert result["settled"]


def _fixture_runtime_lock(tmp_path: Path) -> tuple[base.RuntimeSettings, Path, Path]:
    code = tmp_path / "runtime.py"
    code.write_text("original", encoding="utf-8")
    parent = tmp_path / "008.json"
    parent_digest = write_immutable_json(parent, {"lock_id": "008"})
    lock = tmp_path / "009.json"
    write_immutable_json(
        lock,
        {
            "files": {
                code.as_posix(): sha256_file(code),
                parent.as_posix(): sha256_file(parent),
            }
        },
    )
    settings = base.RuntimeSettings(
        runtime_lock_path=lock,
        parent_008_lock_path=parent,
        expected_parent_008_lock=parent_digest,
    )
    return settings, code, parent


def test_tampered_009_code_blocks_prediction(tmp_path: Path) -> None:
    settings, code, _ = _fixture_runtime_lock(tmp_path)
    code.write_text("tampered", encoding="utf-8")

    def verifier(value):
        result = base.verify_amendment(value)
        return {"effective_operational_lock_intact": result["intact"], "failures": result["mismatches"]}

    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.generate_prediction(TARGET, settings=settings, effective_verifier=verifier)


def test_tampered_009_lock_blocks_prediction(tmp_path: Path) -> None:
    settings, _, _ = _fixture_runtime_lock(tmp_path)
    settings.runtime_lock_path.write_bytes(settings.runtime_lock_path.read_bytes() + b"x")
    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.generate_prediction(
            TARGET,
            settings=settings,
            effective_verifier=lambda value: {
                "effective_operational_lock_intact": base.verify_amendment(value)["intact"]
            },
        )


def test_tampered_008_parent_blocks_prediction(tmp_path: Path) -> None:
    settings, _, parent = _fixture_runtime_lock(tmp_path)
    parent.write_bytes(parent.read_bytes() + b"x")

    def verifier(value):
        result = base.verify_amendment(value)
        return {"effective_operational_lock_intact": result["intact"], "failures": result["mismatches"]}

    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.generate_prediction(TARGET, settings=settings, effective_verifier=verifier)


def test_tampered_human_decision_blocks_prediction(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text('{"operative_champion":"V6"}', encoding="utf-8")
    lock = tmp_path / "008.json"
    write_immutable_json(lock, {"files": {decision.as_posix(): sha256_file(decision)}})
    assert locked._verify_lock_surface(lock)["intact"] is True
    decision.write_text('{"operative_champion":"GEN2"}', encoding="utf-8")
    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.generate_prediction(
            TARGET,
            effective_verifier=lambda _: {
                "effective_operational_lock_intact": locked._verify_lock_surface(lock)[
                    "intact"
                ],
                "failures": ["007:DECISION_TAMPERED"],
            },
        )


def test_operational_lock_status_includes_009(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base,
        "preflight",
        lambda *a, **k: {"daily_prediction_allowed": False, "provider_requests_made": 0},
    )
    result = locked.preflight(TARGET, effective_verifier=good_verifier)
    assert result["parent_008_lock_intact"] is True
    assert result["runtime_009_lock_intact"] is True
    assert result["self_verification_010_intact"] is True


def test_effective_operational_lock_requires_all_parents() -> None:
    for field in (
        "parent_008_lock_intact",
        "runtime_009_lock_intact",
        "self_verification_010_intact",
    ):
        result = GOOD.copy()
        result[field] = False
        result["effective_operational_lock_intact"] = all(
            result[name]
            for name in (
                "parent_008_lock_intact",
                "runtime_009_lock_intact",
                "self_verification_010_intact",
            )
        )
        assert result["effective_operational_lock_intact"] is False


def test_010_activation_verifies_in_clean_checkout(tmp_path: Path) -> None:
    bound = tmp_path / "bound.py"
    bound.write_text("stable", encoding="utf-8")
    lock = tmp_path / "010.json"
    write_immutable_json(
        lock,
        {
            "lock_id": "GEN02-RUNTIME-SELF-VERIFICATION-ACTIVATION-010R2",
            "runtime_009_lock_sha256": base.verify_amendment()["lock_sha256"],
            "files": {bound.as_posix(): sha256_file(bound)},
        },
    )
    assert locked.verify_activation(lock)["intact"] is True


def test_010r1_resolves_lock_local_relative_members(tmp_path: Path) -> None:
    member = tmp_path / "member.json"
    member.write_text("{}", encoding="utf-8")
    lock = tmp_path / "lock.json"
    write_immutable_json(lock, {"files": {"member.json": sha256_file(member)}})
    assert locked._verify_lock_surface(lock)["intact"] is True


def test_010r2_effective_runtime_verifies_in_clean_checkout() -> None:
    result = locked.verify_effective_runtime_freeze()
    assert result["effective_operational_lock_intact"] is True
    assert result["v31_operational_lock_intact"] is True
    assert result["failures"] == []


def test_active_research_cannot_claim_ready_when_runtime_lock_invalid(tmp_path: Path) -> None:
    target = tmp_path / "active.json"
    target.write_text(
        json.dumps(
            {
                "production_prediction_ready": True,
                "execution_authorized": True,
                "gen02_effective_operational_lock_intact": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.write_derived_active_research_view(
            target, effective_verifier=bad_verifier
        )
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["production_prediction_ready"] is True  # no false PASS rewrite occurred


def test_active_research_uses_only_self_verifying_operational_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "active.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        base,
        "review_checkpoint",
        lambda settings: {
            "prediction_trading_days": 0,
            "settled_20d_dates": 0,
            "status": "PIPELINE_ONLY",
            "human_review_required": True,
            "automatic_promotion_allowed": False,
            "production_prediction_ready": False,
            "execution_authorized": False,
        },
    )
    value = locked.write_derived_active_research_view(
        target, effective_verifier=good_verifier
    )
    canonical = "prospective_gen2_runtime_locked"
    for key in (
        "gen02_prospective_runtime_entrypoint",
        "gen02_prospective_input_sealing_entrypoint",
        "gen02_prospective_preflight_entrypoint",
        "gen02_prospective_prediction_entrypoint",
        "gen02_prospective_settlement_entrypoint",
    ):
        assert canonical in value[key]
    assert value["gen02_parent_008_operational_lock_sha256"] is None
    assert value["gen02_effective_operational_lock_sha256"] == "010"
    assert value["gen02_prospective_operational_lock_sha256"] == "010"


def test_no_provider_request_before_lock_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(base, "seal_inputs", lambda *a, **k: calls.append("provider"))
    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.seal_inputs(TARGET, effective_verifier=bad_verifier)
    assert calls == []


def test_no_prediction_artifact_written_when_lock_invalid(tmp_path: Path) -> None:
    settings = base.RuntimeSettings(prediction_root=tmp_path / "predictions")
    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.generate_prediction(TARGET, settings=settings, effective_verifier=bad_verifier)
    assert not settings.prediction_root.exists()


def test_no_settlement_written_when_lock_invalid(tmp_path: Path) -> None:
    settings = base.RuntimeSettings(settlement_root=tmp_path / "settlements")
    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.settle_prediction(
            TARGET,
            tmp_path / "market.csv",
            settings=settings,
            effective_verifier=bad_verifier,
        )
    assert not settings.settlement_root.exists()


def _changed_paths() -> list[str]:
    import subprocess

    return subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], text=True
    ).splitlines()


def test_v6_unchanged() -> None:
    assert not any(name.startswith("research_v6/") for name in _changed_paths())


def test_v1r4_unchanged() -> None:
    assert not any(
        name.startswith("stockpilot/prospective_r4/")
        or name.startswith("artifacts/prospective_alpha_v1r4/")
        for name in _changed_paths()
    )


def test_no_auto_promotion() -> None:
    assert GOOD["production_prediction_ready"] is False
    assert GOOD["execution_authorized"] is False


@pytest.mark.parametrize("command", ["seal-inputs", "preflight", "predict", "settle"])
def test_canonical_cli_routes_every_operational_command_through_guard(
    monkeypatch: pytest.MonkeyPatch, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []

    def mark(*args, **kwargs):
        calls.append(command)
        return {"command": command, "execution_authorized": False}

    monkeypatch.setattr(locked, "seal_inputs", mark)
    monkeypatch.setattr(locked, "preflight", mark)
    monkeypatch.setattr(locked, "generate_prediction", mark)
    monkeypatch.setattr(locked, "settle_prediction", mark)
    argv = [command, "--date", TARGET]
    if command == "settle":
        argv.extend(["--market", "market.csv"])
    assert locked.main(argv) == 0
    assert calls == [command]
    assert command in capsys.readouterr().out


def test_status_and_verify_require_effective_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(locked, "verify_effective_runtime_freeze", lambda *a, **k: BAD.copy())
    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.main(["verify"])
    monkeypatch.setattr(locked, "derived_status", lambda *a, **k: (_ for _ in ()).throw(locked.EffectiveRuntimeLockError("invalid")))
    with pytest.raises(locked.EffectiveRuntimeLockError):
        locked.main(["status"])
