from __future__ import annotations

import numpy as np
import pandas as pd


def assess_panel(panel: pd.DataFrame, backtest_start: str | pd.Timestamp) -> dict:
    ordered = panel.sort_values(["symbol", "date"])
    grouped = ordered.groupby("symbol", group_keys=False)
    first = grouped["close"].first()
    last = grouped["close"].last()
    buy_hold = last / first - 1
    daily_return = grouped["close"].pct_change()
    max_move_by_symbol = daily_return.abs().groupby(ordered["symbol"]).max()
    quarantined_symbols = sorted(max_move_by_symbol[max_move_by_symbol > 0.35].index.astype(str))
    warnings: list[str] = []

    snapshot_date = None
    membership_mode = "unrestricted"
    membership_coverage = 0.0
    membership_symbol_coverage = 0.0
    membership_snapshots = 0
    survivorship_bias = False
    if "in_universe" in panel and "membership_known" in panel:
        membership_mode = "point_in_time"
        by_date = panel.groupby("date")["membership_known"].max()
        test_dates = by_date.index >= pd.Timestamp(backtest_start)
        membership_coverage = float(by_date.loc[test_dates].mean()) if test_dates.any() else 0.0
        if "membership_snapshot_date" in panel:
            membership_snapshots = int(
                pd.to_datetime(panel["membership_snapshot_date"], errors="coerce").nunique()
            )
        if {"membership_universe_size", "membership_available_size"}.issubset(panel.columns):
            sizes = panel.groupby("date")[
                ["membership_universe_size", "membership_available_size"]
            ].max()
            valid_sizes = sizes["membership_universe_size"] > 0
            if valid_sizes.any():
                membership_symbol_coverage = float(
                    (
                        sizes.loc[valid_sizes, "membership_available_size"]
                        / sizes.loc[valid_sizes, "membership_universe_size"]
                    ).mean()
                )
        if membership_coverage < 1:
            warnings.append("历史成分覆盖不足100%，缺失区间不会参与训练或回测。")
        if membership_symbol_coverage < 0.9:
            survivorship_bias = True
            warnings.append("行情未覆盖至少90%的当期成分股，仍存在严重的股票选择偏差。")
    elif "snapshot_date" in panel:
        membership_mode = "latest_snapshot_backfill"
        snapshots = pd.to_datetime(panel["snapshot_date"], errors="coerce").dropna()
        if not snapshots.empty:
            snapshot_date = str(snapshots.max().date())
            survivorship_bias = snapshots.max() > pd.Timestamp(backtest_start)
            if survivorship_bias:
                warnings.append("股票池快照晚于回测起点，存在显著幸存者偏差。")
    if buy_hold.max() > 5 and (
        membership_mode != "point_in_time" or membership_symbol_coverage < 0.9
    ):
        warnings.append("股票池包含历史涨幅超过500%的事后赢家，组合收益可能高度集中。")
    if quarantined_symbols:
        warnings.append(
            f"{len(quarantined_symbols)}只股票出现单日绝对收益超过35%，已从训练和交易中隔离。"
        )
    if len(buy_hold) < 30:
        warnings.append("股票池少于30只，横截面预测与Rank IC稳定性不足。")

    exposure_scope = (
        panel["in_universe"].fillna(False)
        if "in_universe" in panel
        else pd.Series(True, index=panel.index)
    )
    float_market_cap_coverage = (
        float(panel.loc[exposure_scope, "float_market_cap"].notna().mean())
        if "float_market_cap" in panel and exposure_scope.any()
        else 0.0
    )
    industry_coverage = (
        float(panel.loc[exposure_scope, "industry"].notna().mean())
        if "industry" in panel and exposure_scope.any()
        else 0.0
    )
    if "float_market_cap" in panel and float_market_cap_coverage < 0.9:
        warnings.append("历史流通市值覆盖不足90%，缺失样本不会用于中性标签训练。")
    if "industry" in panel and industry_coverage < 0.8:
        warnings.append("历史行业覆盖不足80%，缺失处将回退到上市板块分组。")

    return {
        "symbols": int(panel["symbol"].nunique()),
        "rows": len(panel),
        "data_start": str(pd.to_datetime(panel["date"]).min().date()),
        "data_end": str(pd.to_datetime(panel["date"]).max().date()),
        "snapshot_date": snapshot_date,
        "membership_mode": membership_mode,
        "membership_coverage": membership_coverage,
        "membership_symbol_coverage": membership_symbol_coverage,
        "membership_snapshots": membership_snapshots,
        "survivorship_bias": survivorship_bias,
        "best_constituent_return": float(buy_hold.max()),
        "median_constituent_return": float(buy_hold.median()),
        "max_abs_daily_return": float(np.nanmax(np.abs(daily_return))),
        "quarantined_symbols": quarantined_symbols,
        "warnings": warnings,
        "float_market_cap_coverage": float_market_cap_coverage,
        "industry_coverage": industry_coverage,
        "neutralization_size_source": (
            "float_market_cap" if "float_market_cap" in panel else "amount_proxy"
        ),
        "neutralization_group_source": "pit_industry+board_fallback"
        if "industry" in panel
        else "board_proxy",
    }
