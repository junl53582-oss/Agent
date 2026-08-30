from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from stockpilot.prospective_r2.integrity import (
    canonical_frame_bytes,
    read_verified_json,
    sha256_bytes,
    sha256_file,
    verify_immutable,
    write_immutable_json,
)
from stockpilot.prospective_r2.sources import load_pit_context

from .config import OperationalSettings


HORIZONS = (1, 5, 20)


@dataclass(frozen=True)
class SettlementBundle:
    manifest_path: str
    manifest_sha256: str
    market_status: str
    market_path: str | None
    market_sha256: str | None
    benchmark_status: str
    benchmark_path: str | None
    benchmark_sha256: str | None
    price_adjustment_mode: str
    corporate_action_dataset_path: str
    corporate_action_dataset_hash: str
    corporate_action_manifest_hash: str
    corporate_action_lock_hash: str
    corporate_action_lock_verified: bool
    corporate_action_dataset_verified: bool
    corporate_action_verified: bool
    trading_calendar_path: str
    trading_calendar_hash: str

    @property
    def ready(self) -> bool:
        return (
            self.market_status == "APPROVED"
            and self.benchmark_status == "APPROVED"
            and self.corporate_action_verified
        )


def _mapping(payload: dict) -> dict[str, str]:
    values = payload.get("sha256") or payload.get("files") or {}
    return {str(name).replace("\\", "/"): str(value).lower() for name, value in values.items()}


def verify_mapping_lock(lock_path: str | Path, expected_lock_sha256: str) -> dict:
    path = Path(lock_path)
    actual_lock = sha256_file(path)
    if actual_lock.lower() != expected_lock_sha256.lower():
        raise RuntimeError("trusted lock hash mismatch")
    sidecar_candidates = [
        path.with_suffix(path.suffix + ".sha256"),
        path.with_name("plan.lock.sha256"),
    ]
    sidecar = next((item for item in sidecar_candidates if item.exists()), None)
    if sidecar is None or sidecar.read_text(encoding="ascii").strip().lower() != actual_lock.lower():
        raise RuntimeError("trusted lock sidecar mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    bindings = _mapping(payload)
    if not bindings:
        raise RuntimeError("trusted lock contains no file bindings")
    mismatches = [
        name
        for name, expected in bindings.items()
        if not Path(name).exists() or sha256_file(name).lower() != expected
    ]
    if mismatches:
        raise RuntimeError(f"trusted lock file binding mismatch: {mismatches}")
    return {
        "lock_sha256": actual_lock,
        "lock_sidecar_verified": True,
        "internal_file_bindings_verified": True,
        "bindings": bindings,
        "payload": payload,
    }


def verify_corporate_action_trust_root(
    dataset_path: str | Path,
    lock_path: str | Path,
    expected_lock_sha256: str,
    settlement_manifest_hash: str,
) -> dict:
    evidence = verify_mapping_lock(lock_path, expected_lock_sha256)
    dataset = Path(dataset_path)
    key = dataset.as_posix()
    expected = evidence["bindings"].get(key)
    actual = sha256_file(dataset)
    if expected is None or expected.lower() != actual.lower():
        raise RuntimeError("corporate action dataset is outside the trusted lock chain")
    return {
        "price_adjustment_mode": "HFQ_PIT_GOVERNED",
        "corporate_action_dataset_hash": actual,
        "corporate_action_manifest_hash": settlement_manifest_hash,
        "corporate_action_lock_hash": evidence["lock_sha256"],
        "corporate_action_lock_verified": True,
        "corporate_action_dataset_verified": True,
        "corporate_action_verified": True,
    }


def _verify_market_chain(manifest: dict, lock_evidence: dict) -> tuple[str, str]:
    market = manifest["market"]
    path = Path(market["path"])
    audit_path = Path(market["data_audit_path"])
    audit_key = audit_path.as_posix()
    if lock_evidence["bindings"].get(audit_key) != sha256_file(audit_path).lower():
        raise RuntimeError("market data audit is outside the trusted V20r2 lock")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    inputs = {str(name).replace("\\", "/"): str(value).lower() for name, value in audit["input_sha256"].items()}
    actual = sha256_file(path)
    if inputs.get(path.as_posix()) != actual.lower():
        raise RuntimeError("market source is not bound by the trusted data audit")
    return path.as_posix(), actual


def _verify_benchmark(manifest: dict) -> tuple[str, str]:
    benchmark = manifest["benchmark"]
    path = Path(benchmark["path"])
    evidence_path = Path(benchmark["evidence_manifest_path"])
    evidence = read_verified_json(evidence_path)
    mapping = _mapping(evidence)
    actual = sha256_file(path)
    if mapping.get(path.as_posix()) != actual.lower():
        raise RuntimeError("benchmark source is not bound by its approved manifest")
    return path.as_posix(), actual


def load_approved_settlement_bundle(
    settings: OperationalSettings,
    *,
    lock_verifier: Callable | None = None,
) -> SettlementBundle:
    manifest = read_verified_json(settings.settlement_manifest_path)
    manifest_hash = verify_immutable(settings.settlement_manifest_path)
    if lock_verifier is None:
        from .freeze import verify_lock

        lock_verifier = verify_lock
    official = lock_verifier(settings)
    if official.get("frozen_inputs_intact") is not True:
        raise RuntimeError("settlement manifest has no intact V1r3 trust root")
    expected_lock = manifest["corporate_actions"]["trusted_lock_sha256"]
    lock_evidence = verify_mapping_lock(
        manifest["corporate_actions"]["trusted_lock_path"], expected_lock
    )
    action = verify_corporate_action_trust_root(
        manifest["corporate_actions"]["dataset_path"],
        manifest["corporate_actions"]["trusted_lock_path"],
        expected_lock,
        manifest_hash,
    )

    market_status = manifest["market"]["status"]
    market_path = market_hash = None
    if market_status == "APPROVED":
        market_path, market_hash = _verify_market_chain(manifest, lock_evidence)

    benchmark_status = manifest["benchmark"]["status"]
    benchmark_path = benchmark_hash = None
    if benchmark_status == "APPROVED":
        benchmark_path, benchmark_hash = _verify_benchmark(manifest)

    calendar_path = Path(manifest["trading_calendar"]["path"])
    calendar_hash = sha256_file(calendar_path)
    if calendar_hash != manifest["trading_calendar"]["sha256"]:
        raise RuntimeError("approved settlement calendar hash mismatch")
    return SettlementBundle(
        manifest_path=settings.settlement_manifest_path.as_posix(),
        manifest_sha256=manifest_hash,
        market_status=market_status,
        market_path=market_path,
        market_sha256=market_hash,
        benchmark_status=benchmark_status,
        benchmark_path=benchmark_path,
        benchmark_sha256=benchmark_hash,
        corporate_action_dataset_path=manifest["corporate_actions"]["dataset_path"],
        trading_calendar_path=calendar_path.as_posix(),
        trading_calendar_hash=calendar_hash,
        **action,
    )


def _normalize_market(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "open"}
    if required - set(frame.columns):
        raise ValueError("market settlement source requires date/symbol/open")
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    output["symbol"] = output["symbol"].astype(str).str.zfill(6)
    if output.duplicated(["date", "symbol"]).any():
        raise ValueError("market settlement source has duplicate date/symbol")
    return output


def _normalize_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    if {"date", "open"} - set(frame.columns):
        raise ValueError("benchmark settlement source requires date/open")
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    if output["date"].duplicated().any():
        raise ValueError("benchmark settlement source has duplicate dates")
    return output


def _frame_hash(frame: pd.DataFrame, keys: list[str]) -> str:
    return sha256_bytes(canonical_frame_bytes(frame, keys))


def _read_csv(path: str | Path, kind: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str})
    return _normalize_market(frame) if kind == "market" else _normalize_benchmark(frame)


def _verify_supplied(
    supplied: pd.DataFrame | None,
    parsed: pd.DataFrame,
    keys: list[str],
    kind: str,
) -> pd.DataFrame:
    if supplied is None:
        return parsed
    normalized = _normalize_market(supplied) if kind == "market" else _normalize_benchmark(supplied)
    if _frame_hash(normalized, keys) != _frame_hash(parsed, keys):
        raise RuntimeError(f"{kind} DataFrame provenance mismatch")
    return normalized


def settle_certified_labels(
    predictions: pd.DataFrame,
    *,
    bundle: SettlementBundle,
    settings: OperationalSettings,
    as_of: str | pd.Timestamp,
    expected_universe_by_date: dict[str, set[str]],
    market: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
) -> list[dict]:
    if not bundle.ready or not bundle.market_path or not bundle.benchmark_path:
        raise RuntimeError("approved settlement source bundle is incomplete")
    parsed_market = _read_csv(bundle.market_path, "market")
    parsed_benchmark = _read_csv(bundle.benchmark_path, "benchmark")
    prices = _verify_supplied(market, parsed_market, ["date", "symbol"], "market")
    bench = _verify_supplied(benchmark, parsed_benchmark, ["date"], "benchmark")
    calendar = pd.DatetimeIndex(prices["date"].drop_duplicates().sort_values())
    as_of_date = pd.Timestamp(as_of).normalize()
    if calendar.empty or calendar.max() > as_of_date or bench["date"].max() > as_of_date:
        raise ValueError("settlement source contains data after as_of")
    benchmark_open = bench.set_index("date")["open"]
    results: list[dict] = []
    for prediction in predictions.itertuples(index=False):
        prediction_date = pd.Timestamp(prediction.date).normalize()
        prediction_key = str(prediction_date.date())
        expected = {
            str(value).zfill(6)
            for value in expected_universe_by_date.get(prediction_key, set())
        }
        if not expected:
            raise ValueError(f"prediction universe provenance missing for {prediction_key}")
        position = calendar.get_indexer([prediction_date])[0]
        if position < 0:
            continue
        symbol = str(prediction.symbol).zfill(6)
        if symbol not in expected:
            raise ValueError("prediction symbol is outside its PIT universe proof")
        _, proof = load_pit_context(prediction_key, settings)
        for horizon in HORIZONS:
            entry_pos, exit_pos = position + 1, position + horizon + 1
            if exit_pos >= len(calendar) or calendar[exit_pos] > as_of_date:
                continue
            entry_date, maturity_date = calendar[entry_pos], calendar[exit_pos]
            entry = prices[prices["date"].eq(entry_date) & prices["symbol"].eq(symbol)]
            exit_ = prices[prices["date"].eq(maturity_date) & prices["symbol"].eq(symbol)]
            entry_open = pd.to_numeric(entry["open"].iloc[0], errors="coerce") if len(entry) else np.nan
            exit_open = pd.to_numeric(exit_["open"].iloc[0], errors="coerce") if len(exit_) else np.nan
            benchmark_entry = pd.to_numeric(benchmark_open.get(entry_date, np.nan), errors="coerce")
            benchmark_exit = pd.to_numeric(benchmark_open.get(maturity_date, np.nan), errors="coerce")
            status = "SETTLED"
            if not np.isfinite(entry_open):
                status = "MISSING_ENTRY_PRICE"
            elif not np.isfinite(exit_open):
                status = "MISSING_EXIT_PRICE"
            elif len(exit_) and bool(exit_.iloc[0].get("is_delisted", False)):
                status = "DELISTED_AT_EXIT"
            elif (len(entry) and bool(entry.iloc[0].get("is_suspended", False))) or (
                len(exit_) and bool(exit_.iloc[0].get("is_suspended", False))
            ):
                status = "SUSPENDED_PRICE_UNAVAILABLE"
            elif not np.isfinite(benchmark_entry) or not np.isfinite(benchmark_exit):
                status = "MISSING_BENCHMARK_PRICE"
            forward = float(exit_open / entry_open - 1) if status == "SETTLED" else None
            benchmark_return = (
                float(benchmark_exit / benchmark_entry - 1) if status == "SETTLED" else None
            )
            record = {
                "prediction_date": prediction_key,
                "maturity_date": str(maturity_date.date()),
                "symbol": symbol,
                "horizon": horizon,
                "entry_date": str(entry_date.date()),
                "entry_price": float(entry_open) if np.isfinite(entry_open) else None,
                "exit_price": float(exit_open) if np.isfinite(exit_open) else None,
                "forward_return": forward,
                "benchmark_return": benchmark_return,
                "excess_return": forward - benchmark_return if status == "SETTLED" else None,
                "status": status,
                "expected_universe_size": len(expected),
                "membership_snapshot_hash": proof["membership_snapshot_sha256"],
                "price_source_path": bundle.market_path,
                "price_source_sha256": bundle.market_sha256,
                "benchmark_source_path": bundle.benchmark_path,
                "benchmark_source_sha256": bundle.benchmark_sha256,
                "settlement_manifest_path": bundle.manifest_path,
                "settlement_manifest_sha256": bundle.manifest_sha256,
                "price_provenance_verified": True,
                "benchmark_provenance_verified": True,
                "price_adjustment_mode": "HFQ_PIT_GOVERNED",
                "corporate_action_dataset_path": bundle.corporate_action_dataset_path,
                "corporate_action_dataset_hash": bundle.corporate_action_dataset_hash,
                "corporate_action_manifest_hash": bundle.corporate_action_manifest_hash,
                "corporate_action_lock_hash": bundle.corporate_action_lock_hash,
                "corporate_action_lock_verified": bundle.corporate_action_lock_verified,
                "corporate_action_dataset_verified": bundle.corporate_action_dataset_verified,
                "corporate_action_verified": bundle.corporate_action_verified,
                "label_fully_verified": status == "SETTLED" and bundle.corporate_action_verified,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "execution_authorized": False,
            }
            target = settings.labels_root / prediction_key / f"{symbol}_{horizon}d.json"
            if target.exists():
                existing = read_verified_json(target)
                stable = lambda item: {key: value for key, value in item.items() if key != "recorded_at_utc"}
                if stable(existing) != stable(record):
                    raise RuntimeError(f"mature label is immutable: {target}")
                results.append(existing)
            else:
                write_immutable_json(target, record)
                results.append(record)
    return results


def load_verified_label_records(root: str | Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(Path(root).glob("*/*.json")):
        item = read_verified_json(path)
        item["label_record_path"] = path.as_posix()
        item["label_record_sha256"] = verify_immutable(path)
        records.append(item)
    return records


def certify_label_record(
    record: dict,
    settings: OperationalSettings,
    *,
    bundle_loader: Callable = load_approved_settlement_bundle,
    bundle: SettlementBundle | None = None,
    source_cache: dict | None = None,
) -> dict:
    failures: list[str] = []
    try:
        path = Path(record["label_record_path"])
        if verify_immutable(path) != record["label_record_sha256"]:
            raise RuntimeError("label record hash mismatch")
        stored = read_verified_json(path)
        supplied = {
            key: value
            for key, value in record.items()
            if key not in {"label_record_path", "label_record_sha256"}
        }
        if stored != supplied:
            raise RuntimeError("label record differs from immutable evidence")
        bundle = bundle or bundle_loader(settings)
        if not bundle.ready:
            raise RuntimeError("approved settlement source bundle is incomplete")
        if sha256_file(record["price_source_path"]) != record["price_source_sha256"]:
            raise RuntimeError("price source hash mismatch")
        if sha256_file(record["benchmark_source_path"]) != record["benchmark_source_sha256"]:
            raise RuntimeError("benchmark source hash mismatch")
        if record["settlement_manifest_sha256"] != bundle.manifest_sha256:
            raise RuntimeError("settlement manifest hash mismatch")
        if record["corporate_action_lock_hash"] != bundle.corporate_action_lock_hash:
            raise RuntimeError("corporate action lock trust root mismatch")
        if record["corporate_action_dataset_hash"] != bundle.corporate_action_dataset_hash:
            raise RuntimeError("corporate action dataset trust root mismatch")
        _, proof = load_pit_context(record["prediction_date"], settings)
        if record["membership_snapshot_hash"] != proof["membership_snapshot_sha256"]:
            raise RuntimeError("label PIT universe proof mismatch")
        if int(record["expected_universe_size"]) != int(proof["universe_size"]):
            raise RuntimeError("label expected universe size mismatch")

        cache = source_cache if source_cache is not None else {}
        market_key = ("market", record["price_source_path"], record["price_source_sha256"])
        benchmark_key = (
            "benchmark", record["benchmark_source_path"], record["benchmark_source_sha256"]
        )
        if market_key not in cache:
            cache[market_key] = _read_csv(record["price_source_path"], "market")
        if benchmark_key not in cache:
            cache[benchmark_key] = _read_csv(record["benchmark_source_path"], "benchmark")
        market = cache[market_key]
        benchmark = cache[benchmark_key]
        symbol = str(record["symbol"]).zfill(6)
        entry_date = pd.Timestamp(record["entry_date"])
        exit_date = pd.Timestamp(record["maturity_date"])
        entry = market[market["date"].eq(entry_date) & market["symbol"].eq(symbol)]
        exit_ = market[market["date"].eq(exit_date) & market["symbol"].eq(symbol)]
        bmap = benchmark.set_index("date")["open"]
        entry_open = float(entry["open"].iloc[0])
        exit_open = float(exit_["open"].iloc[0])
        b_entry, b_exit = float(bmap.loc[entry_date]), float(bmap.loc[exit_date])
        expected_forward = exit_open / entry_open - 1
        expected_benchmark = b_exit / b_entry - 1
        if not np.isclose(record["forward_return"], expected_forward):
            raise RuntimeError("label forward return does not recompute")
        if not np.isclose(record["benchmark_return"], expected_benchmark):
            raise RuntimeError("label benchmark return does not recompute")
        if record.get("status") != "SETTLED":
            raise RuntimeError("only SETTLED labels qualify")
        verified = True
    except Exception as error:
        failures.append(f"{type(error).__name__}:{error}")
        verified = False
    return {
        "prediction_date": record.get("prediction_date"),
        "symbol": record.get("symbol"),
        "horizon": record.get("horizon"),
        "label_evidence_verified": verified,
        "failures": failures,
    }


def load_predictions(root: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(root).glob("????-??-??.csv")):
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists() or sidecar.read_text(encoding="ascii").strip() != sha256_file(path):
            raise RuntimeError(f"prediction snapshot is not intact: {path}")
        frames.append(pd.read_csv(path, dtype={"symbol": str}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "symbol"])


def run_approved_settlement(target_date: str, settings: OperationalSettings) -> dict:
    bundle = load_approved_settlement_bundle(settings)
    if bundle.market_status != "APPROVED":
        return {"status": "SETTLEMENT_BLOCKED_MARKET_UNAPPROVED", "mature_records_written": 0}
    if bundle.benchmark_status != "APPROVED":
        return {"status": "SETTLEMENT_BLOCKED_BENCHMARK_UNAPPROVED", "mature_records_written": 0}
    predictions = load_predictions(settings.prediction_root)
    expected: dict[str, set[str]] = {}
    for value in predictions.get("date", pd.Series(dtype=str)).astype(str).unique():
        panel, _ = load_pit_context(value, settings)
        expected[value] = set(panel["symbol"].astype(str).str.zfill(6))
    records = settle_certified_labels(
        predictions,
        bundle=bundle,
        settings=settings,
        as_of=target_date,
        expected_universe_by_date=expected,
    )
    newly_mature = len(records)
    return {
        "status": "SETTLED" if newly_mature else "NO_MATURE_LABELS",
        "mature_records_written": newly_mature,
        "bundle_manifest_sha256": bundle.manifest_sha256,
    }
