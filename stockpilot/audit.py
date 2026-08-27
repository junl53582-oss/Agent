from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings

PROTOCOL_VERSION = 1
CORE_SOURCE_FILES = [
    "audit.py",
    "adjudication.py",
    "backtest.py",
    "config.py",
    "data.py",
    "exposure.py",
    "features.py",
    "future_test.py",
    "membership.py",
    "model.py",
    "portfolio.py",
    "shadow.py",
    "shadow_evaluate.py",
    "trading.py",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serializable_settings(settings: Settings) -> dict:
    result = asdict(settings)
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value)
    return result


def create_protocol_addendum(
    manifest_path: str | Path = "artifacts/future_test/manifest.lock.json",
    output_path: str | Path = "artifacts/future_test/protocol.addendum.lock.json",
    settings: Settings | None = None,
    source_root: str | Path | None = None,
    adopted_files: list[str | Path] | None = None,
) -> dict:
    """Complete an existing frozen protocol without modifying its original manifest."""
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"协议补充锁已存在，禁止覆盖: {target}")
    manifest_target = Path(manifest_path)
    manifest = json.loads(manifest_target.read_text(encoding="utf-8"))
    selected = manifest["selected_config"]
    base = settings or Settings.from_env()
    allowed = {"model_name", "top_n", "weighting", "hold_buffer", "industry_cap"}
    locked = replace(base, **{key: selected[key] for key in allowed if key in selected})
    package_root = Path(source_root).resolve() if source_root else Path(__file__).parent.resolve()
    source_hashes = {
        name: sha256_file(package_root / name)
        for name in CORE_SOURCE_FILES
        if (package_root / name).exists()
    }
    dependency_versions = {}
    for distribution in ["numpy", "pandas", "lightgbm", "akshare"]:
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = None
    adopted = {
        str(Path(path).resolve()): sha256_file(path)
        for path in (adopted_files or [])
        if Path(path).exists()
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "amends_manifest": str(manifest_target.resolve()),
        "manifest_sha256": sha256_file(manifest_target),
        "locked_settings": _serializable_settings(locked),
        "source_root": str(package_root),
        "core_source_sha256": source_hashes,
        "runtime": {
            "python": platform.python_version(),
            "dependencies": dependency_versions,
            "market_data_provider": "tencent",
            "price_adjustment": "qfq",
        },
        "adopted_artifacts_sha256": adopted,
        "prediction_snapshot_schema": [
            "date",
            "symbol",
            "score",
            "pred_rank",
            "eligible",
            "selected",
            "planned_weight",
        ],
        "adjudication": {
            "window": "first minimum_trading_days observations",
            "label_maturity_buffer": locked.horizon + 1,
            "rank_ic": "daily Spearman(score, neutral_or_market_relative_label)",
            "benchmark": "equal-weight eligible point-in-time universe, gross return",
            "decision_never_authorizes_execution": True,
        },
        "execution_authorized": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def verify_protocol_addendum(
    addendum_path: str | Path,
    *,
    verify_source: bool = True,
    raise_on_error: bool = True,
) -> dict:
    target = Path(addendum_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    checks = {"manifest": sha256_file(payload["amends_manifest"]) == payload["manifest_sha256"]}
    if verify_source:
        root = Path(payload["source_root"])
        for name, expected in payload["core_source_sha256"].items():
            path = root / name
            checks[f"source:{name}"] = path.exists() and sha256_file(path) == expected
        runtime = payload.get("runtime", {})
        checks["runtime:python"] = platform.python_version() == runtime.get("python")
        for distribution, expected in runtime.get("dependencies", {}).items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            checks[f"runtime:{distribution}"] = actual == expected
    for path, expected in payload.get("adopted_artifacts_sha256", {}).items():
        target = Path(path)
        checks[f"adopted:{path}"] = target.exists() and sha256_file(target) == expected
    if raise_on_error and not all(checks.values()):
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise RuntimeError(f"协议补充锁完整性校验失败: {failed}")
    return checks


def settings_from_addendum(addendum_path: str | Path) -> Settings:
    payload = json.loads(Path(addendum_path).read_text(encoding="utf-8"))
    values = dict(payload["locked_settings"])
    values["data_dir"] = Path(values["data_dir"])
    values["artifact_dir"] = Path(values["artifact_dir"])
    return Settings(**values)


def _record_hash(record: dict) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_audit_chain(chain_path: str | Path) -> list[dict]:
    target = Path(chain_path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]


def verify_audit_chain(chain_path: str | Path, raise_on_error: bool = True) -> dict:
    records = read_audit_chain(chain_path)
    previous = "GENESIS"
    errors: list[str] = []
    for expected_sequence, record in enumerate(records, 1):
        stored_hash = record.get("record_hash")
        unsigned = {key: value for key, value in record.items() if key != "record_hash"}
        if record.get("sequence") != expected_sequence:
            errors.append(f"sequence:{expected_sequence}")
        if record.get("previous_hash") != previous:
            errors.append(f"previous_hash:{expected_sequence}")
        if _record_hash(unsigned) != stored_hash:
            errors.append(f"record_hash:{expected_sequence}")
        path = Path(record["path"])
        if not path.exists():
            errors.append(f"missing:{path}")
        elif sha256_file(path) != record["file_sha256"]:
            errors.append(f"file_sha256:{path}")
        previous = stored_hash or "INVALID"
    result = {"records": len(records), "head": previous, "intact": not errors, "errors": errors}
    if raise_on_error and errors:
        raise RuntimeError("影子审计哈希链失败: " + ", ".join(errors[:5]))
    return result


def append_audit_record(chain_path: str | Path, file_path: str | Path, category: str) -> dict:
    chain_target = Path(chain_path)
    existing = read_audit_chain(chain_target)
    normalized = str(Path(file_path))
    for record in existing:
        if record["path"] == normalized:
            if sha256_file(file_path) != record["file_sha256"]:
                raise RuntimeError(f"已登记文件发生变化: {file_path}")
            return record
    verify_audit_chain(chain_target)
    unsigned = {
        "sequence": len(existing) + 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "path": normalized,
        "file_sha256": sha256_file(file_path),
        "previous_hash": existing[-1]["record_hash"] if existing else "GENESIS",
    }
    record = {**unsigned, "record_hash": _record_hash(unsigned)}
    chain_target.parent.mkdir(parents=True, exist_ok=True)
    with chain_target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return record


def bootstrap_audit_chain(
    chain_path: str | Path,
    files: list[tuple[str | Path, str]],
) -> dict:
    for path, category in sorted(files, key=lambda item: str(item[0])):
        if Path(path).exists():
            append_audit_record(chain_path, path, category)
    return verify_audit_chain(chain_path)
