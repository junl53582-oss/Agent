from __future__ import annotations

from pathlib import Path

import pandas as pd

from research_v3.fundamentals import FUNDAMENTAL_COLUMNS, RAW_COLUMNS


EXTRA_RAW_COLUMNS = {
    "KCFJCXSYJLRTZ": "deducted_profit_growth",
    "XSMLL_TB": "gross_margin_yoy",
    "YSZKZZL": "receivables_turnover",
    "CHZZL": "inventory_turnover",
    "FIXED_ASSET_TR": "fixed_asset_turnover",
    "CASH_RATIO": "cash_ratio",
    "INTEREST_DEBT_RATIO": "interest_debt_ratio",
    "OPERATE_CYCLE": "operating_cycle",
    "AVG_TOI": "staff_average_revenue",
    "AVG_NET_PROFIT": "staff_average_profit",
    "FCFF_BACK": "fcff_back",
}
EXTRA_COLUMNS = list(EXTRA_RAW_COLUMNS.values())
V10_FUNDAMENTAL_COLUMNS = [*FUNDAMENTAL_COLUMNS, *EXTRA_COLUMNS]


def build_extended_fundamentals(
    symbols: list[str] | pd.Series,
    cache_dir: str | Path = "data/fundamental_cache",
    output_path: str | Path = "data/fundamentals_pit_v10_extended.csv",
) -> tuple[pd.DataFrame, list[str]]:
    cache = Path(cache_dir)
    pieces = []
    failures = []
    mapping = {**RAW_COLUMNS, **EXTRA_RAW_COLUMNS}
    for symbol in sorted({str(value).zfill(6) for value in symbols}):
        path = cache / f"{symbol}.csv"
        if not path.exists():
            failures.append(symbol)
            continue
        raw = pd.read_csv(path, dtype={"SECURITY_CODE": str}).rename(columns=mapping)
        raw["symbol"] = symbol
        for column in V10_FUNDAMENTAL_COLUMNS:
            if column not in raw:
                raw[column] = pd.NA
        for column in ["report_date", "available_date", "update_date"]:
            raw[column] = pd.to_datetime(raw[column], errors="coerce").dt.normalize()
        numeric = [
            column
            for column in V10_FUNDAMENTAL_COLUMNS
            if column not in {"symbol", "report_date", "available_date", "update_date"}
        ]
        raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
        raw = raw.dropna(subset=["report_date", "available_date"])
        raw = raw[raw["available_date"] >= raw["report_date"]]
        pieces.append(raw[V10_FUNDAMENTAL_COLUMNS])
    if not pieces:
        raise RuntimeError("没有取得任何V10扩展PIT财务数据")
    result = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["symbol", "available_date", "report_date", "update_date"])
        .drop_duplicates(["symbol", "available_date", "report_date"], keep="last")
        .reset_index(drop=True)
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    pd.DataFrame({"symbol": failures}).to_csv(
        output.with_suffix(".failures.csv"), index=False, encoding="utf-8-sig"
    )
    return result, failures


def load_extended_fundamentals(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"symbol": str})
    missing = sorted(set(V10_FUNDAMENTAL_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"V10扩展财务数据缺少字段: {', '.join(missing)}")
    data["symbol"] = data["symbol"].str.zfill(6)
    for column in ["report_date", "available_date", "update_date"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    violation = data["available_date"].notna() & (data["available_date"] < data["report_date"])
    if violation.any():
        raise RuntimeError("V10扩展财务数据包含公告日前记录")
    return data.sort_values(["symbol", "available_date", "report_date"]).reset_index(drop=True)


def attach_extended_fundamentals_asof(
    panel: pd.DataFrame, fundamentals: pd.DataFrame
) -> pd.DataFrame:
    value_columns = [column for column in V10_FUNDAMENTAL_COLUMNS if column != "symbol"]
    left = panel.drop(columns=value_columns, errors="ignore").copy()
    left["date"] = pd.to_datetime(left["date"])
    left["symbol"] = left["symbol"].astype(str).str.zfill(6)
    pieces = []
    for symbol, group in left.groupby("symbol", sort=False):
        filings = fundamentals[fundamentals["symbol"] == symbol].drop(columns="symbol")
        if filings.empty:
            missing = group.copy()
            for column in value_columns:
                missing[column] = pd.NA
            pieces.append(missing)
            continue
        pieces.append(
            pd.merge_asof(
                group.sort_values("date"),
                filings.sort_values("available_date"),
                left_on="date",
                right_on="available_date",
                direction="backward",
            )
        )
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    result["fundamental_age_days"] = (
        result["date"] - pd.to_datetime(result["available_date"])
    ).dt.days
    violation = result["available_date"].notna() & (result["available_date"] > result["date"])
    if violation.any():
        raise RuntimeError("V10扩展财务拼接出现未来数据")
    return result.reset_index(drop=True)

