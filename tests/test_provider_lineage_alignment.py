from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from research_v10.features import V10_FEATURES
from stockpilot.daily_pit.pipeline import (
    DAILY_FEATURE_COLUMNS,
    META_COLUMNS,
    DailyPitError,
    DailyPitSettings,
)
from stockpilot.daily_prediction.product import predict_daily
from stockpilot.prospective_r2.integrity import sha256_file, write_immutable_json
from stockpilot.provider_lineage_alignment import (
    ProviderLineageAlignmentSettings,
    ProviderPrioritySettings,
    _assemble_candidate_panel,
    acquire_lineage_aligned_market,
    acquire_tencent_candidate,
    fetch_tencent_first_hfq,
    validate_routed_hfq_lineage,
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


def _provider_raw(dates: pd.DatetimeIndex, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "close": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "amount": [1000.0] * len(dates),
        }
    )


def test_tencent_is_primary_when_historical_lineage_is_tencent(tmp_path: Path) -> None:
    dates = pd.bdate_range("2026-07-20", "2026-09-03")
    eastmoney_calls = 0

    def tencent_provider(**kwargs):
        assert kwargs["symbol"] == "sz000001"
        return _provider_raw(dates, [10.0 + index * 0.01 for index in range(len(dates))])

    def eastmoney_provider(**kwargs):
        nonlocal eastmoney_calls
        eastmoney_calls += 1
        raise AssertionError(kwargs)

    market, failures, sources, requests = fetch_tencent_first_hfq(
        ["000001"],
        "2026-07-20",
        "2026-09-03",
        cache_root=tmp_path / "cache",
        workers=1,
        tencent_provider=tencent_provider,
        eastmoney_provider=eastmoney_provider,
    )

    assert failures == []
    assert sources == {"tencent": 1}
    assert requests == 1
    assert eastmoney_calls == 0
    assert market["symbol"].unique().tolist() == ["000001"]


def test_daily_prediction_default_uses_lineage_aligned_acquisition() -> None:
    default = inspect.signature(predict_daily).parameters["acquisition_runner"].default
    assert default is acquire_lineage_aligned_market


def test_eastmoney_fallback_enters_unchanged_lineage_block(tmp_path: Path) -> None:
    del tmp_path
    frozen_dates = pd.bdate_range("2026-07-20", "2026-08-21")
    incremental_dates = pd.bdate_range("2026-07-20", "2026-09-03")
    frozen_prices = [10.0 + index * 0.01 for index in range(len(frozen_dates))]
    incremental_prices = [10.0 + index * 0.01 for index in range(len(incremental_dates))]
    incremental_prices[10] *= 1.02
    frozen = _provider_raw(frozen_dates, frozen_prices)
    frozen["symbol"] = "000001"
    frozen["volume"] = frozen.pop("amount")
    frozen["amount"] = frozen["volume"] * frozen["close"]
    incremental = _provider_raw(incremental_dates, incremental_prices)
    incremental["symbol"] = "000001"
    incremental["volume"] = incremental.pop("amount")
    incremental["amount"] = incremental["volume"] * incremental["close"]
    membership = pd.DataFrame(
        {
            "snapshot_date": ["2026-08-31"],
            "index_code": ["000300"],
            "symbol": ["000001"],
            "weight": [1.0],
            "source": ["test"],
        }
    )

    with pytest.raises(DailyPitError, match="HFQ_LINEAGE_FALLBACK_BLOCKED"):
        validate_routed_hfq_lineage(
            frozen,
            incremental,
            membership,
            target_date="2026-09-03",
            settings=DailyPitSettings(minimum_universe_coverage=1.0),
        )


def test_eastmoney_fallback_is_blocked_before_immutable_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen_dates = pd.bdate_range("2026-07-20", "2026-08-21")
    incremental_dates = pd.bdate_range("2026-07-20", "2026-09-03")

    def panel(dates: pd.DatetimeIndex, *, mismatch: bool) -> pd.DataFrame:
        prices = [10.0 + index * 0.01 for index in range(len(dates))]
        if mismatch:
            prices[10] *= 1.02
        result = _provider_raw(dates, prices)
        result["symbol"] = "000001"
        result["volume"] = result.pop("amount")
        result["amount"] = result["volume"] * result["close"]
        return result

    frozen_path = tmp_path / "frozen.csv"
    membership_path = tmp_path / "membership.csv"
    panel(frozen_dates, mismatch=False).to_csv(frozen_path, index=False)
    pd.DataFrame(
        {
            "snapshot_date": ["2026-08-31"],
            "index_code": ["000300"],
            "symbol": ["000001"],
            "weight": [1.0],
            "source": ["test"],
        }
    ).to_csv(membership_path, index=False)
    evidence_path = tmp_path / "lineage.json"
    write_immutable_json(
        evidence_path,
        {"HFQ_MISMATCH_ROOT_CAUSE": {"historical_canonical_provider": "tencent"}},
    )
    settings = DailyPitSettings(
        root=tmp_path / "daily",
        frozen_market_path=frozen_path,
        membership_path=membership_path,
        minimum_universe_coverage=1.0,
    )
    priority = ProviderPrioritySettings(
        lineage_evidence_path=evidence_path,
        cache_root=tmp_path / "cache",
        workers=1,
    )
    monkeypatch.setattr(
        "stockpilot.provider_lineage_alignment.daily_pit_pipeline._session_guard",
        lambda *args, **kwargs: None,
    )

    def fallback_fetcher(*args, **kwargs):
        del args, kwargs
        return panel(incremental_dates, mismatch=True), [], {"eastmoney": 1}, 2

    with pytest.raises(DailyPitError, match="HFQ_LINEAGE_FALLBACK_BLOCKED"):
        acquire_lineage_aligned_market(
            "2026-09-03",
            [],
            now=datetime(2026, 9, 3, 11, tzinfo=timezone.utc),
            settings=settings,
            priority_settings=priority,
            fetcher=fallback_fetcher,
        )
    assert not settings.date_dir("2026-09-03").exists()


def test_tencent_daily_passes_unchanged_overlap_validator() -> None:
    frozen_dates = pd.bdate_range("2026-07-20", "2026-08-21")
    incremental_dates = pd.bdate_range("2026-07-20", "2026-09-03")
    prices = [10.0 + index * 0.01 for index in range(len(incremental_dates))]

    def panel(dates: pd.DatetimeIndex, values: list[float]) -> pd.DataFrame:
        result = _provider_raw(dates, values)
        result["symbol"] = "000001"
        result["volume"] = result.pop("amount")
        result["amount"] = result["volume"] * result["close"]
        return result

    frozen = panel(frozen_dates, prices[: len(frozen_dates)])
    incremental = panel(incremental_dates, prices)
    membership = pd.DataFrame(
        {
            "snapshot_date": ["2026-08-31"],
            "symbol": ["000001"],
        }
    )
    audit = validate_routed_hfq_lineage(
        frozen,
        incremental,
        membership,
        target_date="2026-09-03",
        settings=DailyPitSettings(minimum_universe_coverage=1.0),
    )

    assert audit["passed"] is True
    assert audit["anchored_symbols"] == 1
    assert audit["maximum_observed_ratio_relative_deviation"] == 0.0
    assert audit["maximum_observed_return_absolute_difference"] == 0.0
