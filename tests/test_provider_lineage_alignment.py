from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research_v10.features import V10_FEATURES
from stockpilot.daily_pit.pipeline import DAILY_FEATURE_COLUMNS, META_COLUMNS
from stockpilot.prospective_r2.integrity import sha256_file
from stockpilot.provider_lineage_alignment import (
    ProviderLineageAlignmentSettings,
    _assemble_candidate_panel,
    acquire_tencent_candidate,
    verify_candidate,
)


def _settings(tmp_path: Path) -> ProviderLineageAlignmentSettings:
    production = tmp_path / "production/2026-09-03"
    production.mkdir(parents=True)
    market = pd.DataFrame(
        [{"date": "2026-09-03", "symbol": "000001", "open": 1, "high": 1, "low": 1,
          "close": 1, "volume": 1, "amount": 1}]
    )
    market.to_csv(production / "market.csv", index=False)
    (production / "market_manifest.json").write_text("{}", encoding="utf-8")
    (production / "source_receipt.json").write_text("{}", encoding="utf-8")
    membership = tmp_path / "membership.csv"
    pd.DataFrame(
        {"snapshot_date": ["2026-06-30"], "index_code": ["000300"],
         "symbol": ["000001"], "weight": [1.0], "source": ["test"]}
    ).to_csv(membership, index=False)
    return ProviderLineageAlignmentSettings(
        candidate_root=tmp_path / "candidate",
        cache_root=tmp_path / "cache",
        production_root=tmp_path / "production",
        membership_path=membership,
        workers=1,
    )


def test_tencent_candidate_is_isolated_complete_and_idempotent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    production = settings.production_dir("2026-09-03")
    before = {path.name: sha256_file(path) for path in production.iterdir()}

    def provider(**kwargs):
        assert kwargs["symbol"] == "sz000001"
        assert kwargs["adjust"] == "hfq"
        return pd.DataFrame(
            {
                "date": ["2026-07-20", "2026-09-03"],
                "open": [10.0, 11.0],
                "close": [10.1, 11.1],
                "high": [10.2, 11.2],
                "low": [9.9, 10.9],
                "amount": [1000.0, 1100.0],
            }
        )

    result = acquire_tencent_candidate(
        "2026-09-03",
        now=datetime(2026, 9, 4, tzinfo=timezone.utc),
        settings=settings,
        provider=provider,
    )
    assert result["provider"] == "akshare-tencent"
    assert result["target_symbols"] == 1
    receipt = json.loads(
        (settings.candidate_dir("2026-09-03") / "source_receipt.json").read_text()
    )
    assert receipt["mixed_provider"] is False
    assert receipt["production_partition_modified"] is False
    assert list(
        pd.read_csv(settings.candidate_dir("2026-09-03") / "market.csv", nrows=0).columns
    ) == ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    assert {path.name: sha256_file(path) for path in production.iterdir()} == before
    assert verify_candidate("2026-09-03", settings)["idempotent"] is True
    assert acquire_tencent_candidate("2026-09-03", settings=settings)["idempotent"] is True


def test_candidate_panel_carries_builder_sector_without_changing_features() -> None:
    keys = {"date": pd.Timestamp("2026-09-03"), "symbol": "000001"}
    metadata = pd.DataFrame(
        [{
            **keys,
            "eligible": True,
            "in_universe": True,
            "membership_snapshot_date": pd.Timestamp("2026-06-30"),
            "available_date": pd.Timestamp("2026-08-31"),
            "industry_effective_date": pd.Timestamp("2026-01-01"),
            "industry": "test-industry",
            "benchmark_weight": 1.0,
        }]
    )
    feature_values = {name: float(index) for index, name in enumerate(V10_FEATURES)}
    reduced = pd.DataFrame([{**keys, "broad_sector": "builder-sector", **feature_values}])

    panel = _assemble_candidate_panel(metadata, reduced)

    assert list(panel.columns) == DAILY_FEATURE_COLUMNS
    assert len(set(panel.columns)) == 71
    assert panel.loc[0, "broad_sector"] == "builder-sector"
    assert panel.loc[0, V10_FEATURES].to_dict() == feature_values
    assert set(META_COLUMNS).issubset(panel.columns)
