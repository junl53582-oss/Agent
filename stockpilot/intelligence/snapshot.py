from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import (
    PREDICTION_SCHEMA_VERSION,
    CanonicalPrediction,
    canonical_json_bytes,
    sha256_bytes,
)


def _rank_key(record: CanonicalPrediction) -> tuple[float, float, str]:
    rank = float(record.market_rank) if record.market_rank is not None else float("inf")
    score = -record.raw_rank_score if record.raw_rank_score is not None else float("inf")
    return rank, score, record.symbol


@dataclass(frozen=True)
class CanonicalDailySnapshot:
    prediction_date: str
    universe_id: str
    records: tuple[CanonicalPrediction, ...]
    schema_version: str = PREDICTION_SCHEMA_VERSION
    snapshot_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("canonical daily snapshot cannot be empty")
        expected = sha256_bytes(canonical_json_bytes(self.hash_payload()))
        if self.snapshot_hash and self.snapshot_hash != expected:
            raise ValueError("snapshot_hash does not match canonical snapshot bytes")
        object.__setattr__(self, "snapshot_hash", expected)

    def hash_payload(self) -> dict[str, Any]:
        return {
            "prediction_date": self.prediction_date,
            "universe_id": self.universe_id,
            "schema_version": self.schema_version,
            "records": [record.to_dict() for record in self.records],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.hash_payload() | {"snapshot_hash": self.snapshot_hash}


def build_daily_snapshot(records: Iterable[CanonicalPrediction]) -> CanonicalDailySnapshot:
    ordered = tuple(sorted(records, key=_rank_key))
    if not ordered:
        raise ValueError("canonical daily snapshot cannot be empty")
    dates = {record.prediction_date for record in ordered}
    universes = {record.universe_id for record in ordered}
    schemas = {record.schema_version for record in ordered}
    symbols = [record.symbol for record in ordered]
    if len(dates) != 1:
        raise ValueError("daily snapshot records must share one prediction_date")
    if len(universes) != 1:
        raise ValueError("daily snapshot records must share one universe_id")
    if len(schemas) != 1:
        raise ValueError("daily snapshot records must share one schema_version")
    if len(symbols) != len(set(symbols)):
        raise ValueError("daily snapshot contains duplicate symbols")
    return CanonicalDailySnapshot(
        prediction_date=next(iter(dates)),
        universe_id=next(iter(universes)),
        schema_version=next(iter(schemas)),
        records=ordered,
    )


def derive_top_k(snapshot: CanonicalDailySnapshot, k: int) -> tuple[CanonicalPrediction, ...]:
    if k not in {10, 20, 50}:
        raise ValueError("Prediction V1 Phase 1 supports Top10, Top20, or Top50")
    return snapshot.records[: min(k, len(snapshot.records))]


def canonical_snapshot_bytes(snapshot: CanonicalDailySnapshot) -> bytes:
    return canonical_json_bytes(snapshot.to_dict())


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def write_immutable_daily_snapshot(
    snapshot: CanonicalDailySnapshot, path: str | Path
) -> tuple[bool, str]:
    """Write once; an identical retry is accepted and a changed retry fails closed."""
    target = Path(path)
    payload = canonical_snapshot_bytes(snapshot)
    digest = sha256_bytes(payload)
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"canonical daily snapshot is immutable: {target}")
        return False, digest
    _write_new(target, payload)
    _write_new(Path(str(target) + ".sha256"), (digest + "\n").encode("ascii"))
    return True, digest
