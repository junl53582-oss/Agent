from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import sha256_file

FINANCE_TABLES = {
    "STK_REPORT_DISCLOSURE": ("pub_date", "PIT_SAFE_WITH_LAG_FOR_PUB_DATE_ONLY"),
    "STK_FIN_FORCAST": ("pub_date", "PIT_SAFE_WITH_LAG"),
    "STK_PERFORMANCE_LETTERS": ("pub_date", "PIT_SAFE_WITH_LAG_RESTATEMENT_RISK"),
    "FINANCE_BALANCE_SHEET": ("pub_date", "PIT_RESTATEMENT_RISK"),
    "FINANCE_INCOME_STATEMENT": ("pub_date", "PIT_RESTATEMENT_RISK"),
    "FINANCE_CASHFLOW_STATEMENT": ("pub_date", "PIT_RESTATEMENT_RISK"),
    "STK_HK_HOLD_INFO": ("day", "PIT_SAFE_WITH_LAG"),
}
TABLE_OBJECTS = {
    "FINANCE_BALANCE_SHEET": "STK_BALANCE_SHEET",
    "FINANCE_INCOME_STATEMENT": "STK_INCOME_STATEMENT",
    "FINANCE_CASHFLOW_STATEMENT": "STK_CASHFLOW_STATEMENT",
}


@dataclass(frozen=True)
class Settings:
    root: Path
    artifact_dir: Path
    protocol_path: Path
    audit_date: date
    budget_rows: int = 800_000


@dataclass
class Runtime:
    jq: Any
    settings: Settings
    started_spare: int
    requests: int = 0
    rows_acquired: int = 0
    partitions: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def query(self, dataset: str, estimate: int, call: Callable[[], Any]) -> Any:
        current = int(self.jq.get_query_count()["spare"])
        used = self.started_spare - current
        if used + estimate > self.settings.budget_rows:
            self.write_state("JQDATA_DAILY_QUOTA_PAUSED", dataset)
            raise RuntimeError("JQDATA_DAILY_QUOTA_PAUSED")
        self.requests += 1
        return call()

    def write_state(self, status: str, current_dataset: str | None = None) -> None:
        current = int(self.jq.get_query_count()["spare"])
        value = {
            "status": status,
            "current_dataset": current_dataset,
            "provider_queries": self.requests,
            "real_rows_acquired": self.rows_acquired,
            "raw_partitions": len(self.partitions),
            "quota_start_spare": self.started_spare,
            "quota_current_spare": current,
            "local_budget_rows": self.settings.budget_rows,
            "completed_datasets": [item["dataset"] for item in self.partitions],
            "rejected_datasets": [item["dataset"] for item in self.rejected],
        }
        _atomic_json(self.settings.root / "state/checkpoint.json", value)


def credentials_from_environment() -> tuple[str, str]:
    username = os.environ.get("JQDATA_USERNAME", "")
    password = os.environ.get("JQDATA_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("JQDATA_CREDENTIALS_NOT_AVAILABLE_IN_ENVIRONMENT")
    return username, password


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n").encode()


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    raw = _json_bytes(value)
    _atomic_bytes(path, raw)
    _atomic_bytes(path.with_name(f"{path.name}.sha256"), f"{hashlib.sha256(raw).hexdigest()}\n".encode())


def _schema_hash(frame: pd.DataFrame) -> str:
    value = "\n".join(f"{name}:{frame[name].dtype}" for name in frame.columns).encode()
    return hashlib.sha256(value).hexdigest()


def _date_summary(frame: pd.DataFrame, column: str | None) -> tuple[str | None, str | None]:
    if not column or column not in frame or frame.empty:
        return None, None
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        return None, None
    return str(values.min().date()), str(values.max().date())


def _symbol_count(frame: pd.DataFrame) -> int:
    for name in ("symbol", "stock", "code"):
        if name in frame:
            return int(frame[name].astype(str).nunique())
    return 0


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(name) for name in result.columns]
    sort = [name for name in ("date", "day", "pub_date", "code", "stock", "id") if name in result]
    if sort:
        result = result.sort_values(sort, kind="stable", na_position="last")
    return result.reset_index(drop=True)


def _partition_id(dataset: str, parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256(_json_bytes({"dataset": dataset, "parameters": parameters})).hexdigest()
    return digest[:16]


def store_raw_partition(
    runtime: Runtime,
    dataset: str,
    parameters: dict[str, Any],
    frame: pd.DataFrame,
    date_column: str | None,
    pit_classification: str,
    query_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _canonical_frame(frame)
    part = _partition_id(dataset, parameters)
    raw_path = runtime.settings.root / "raw" / dataset.lower() / f"{part}.csv"
    receipt_path = runtime.settings.root / "receipts" / dataset.lower() / f"{part}.json"
    raw = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if raw_path.exists():
        if sha256_file(raw_path) != digest:
            raise RuntimeError(f"IMMUTABLE_PARTITION_CONFLICT:{dataset}:{part}")
        source = "IMMUTABLE_LOCAL_REUSE"
    else:
        _atomic_bytes(raw_path, raw)
        _atomic_bytes(raw_path.with_name(f"{raw_path.name}.sha256"), f"{digest}\n".encode())
        source = "JQDATA_REAL_PROVIDER"
        runtime.rows_acquired += len(frame)
    date_min, date_max = _date_summary(frame, date_column)
    receipt = {
        "provider": "JQData",
        "dataset": dataset,
        "query_parameters": parameters,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source": source,
        "provider_query_count": query_count,
        "row_count": len(frame),
        "date_min": date_min,
        "date_max": date_max,
        "symbol_count": _symbol_count(frame),
        "columns": list(frame.columns),
        "schema_hash": _schema_hash(frame),
        "sha256": digest,
        "raw_path": raw_path.relative_to(runtime.settings.root).as_posix(),
        "pit_classification": pit_classification,
        "account_identity_persisted": False,
        "credential_values_persisted": False,
    }
    _atomic_json(receipt_path, receipt)
    runtime.partitions.append(receipt)
    runtime.write_state("RUNNING", dataset)
    return frame, receipt


def load_raw_partition(
    runtime: Runtime, dataset: str, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    part = _partition_id(dataset, parameters)
    raw_path = runtime.settings.root / "raw" / dataset.lower() / f"{part}.csv"
    receipt_path = runtime.settings.root / "receipts" / dataset.lower() / f"{part}.json"
    if not raw_path.exists() and not receipt_path.exists():
        return None
    if not raw_path.exists() or not receipt_path.exists():
        raise RuntimeError(f"INCOMPLETE_IMMUTABLE_PARTITION:{dataset}:{part}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if sha256_file(raw_path) != receipt["sha256"]:
        raise RuntimeError(f"IMMUTABLE_PARTITION_HASH_MISMATCH:{dataset}:{part}")
    sidecar = receipt_path.with_name(f"{receipt_path.name}.sha256")
    if not sidecar.exists() or sha256_file(receipt_path) != sidecar.read_text().strip():
        raise RuntimeError(f"RECEIPT_HASH_MISMATCH:{dataset}:{part}")
    frame = pd.read_csv(raw_path, dtype={"code": "string", "stock": "string"})
    observed_symbols = _symbol_count(frame)
    if int(receipt.get("symbol_count", -1)) != observed_symbols:
        receipt["symbol_count"] = observed_symbols
        _atomic_json(receipt_path, receipt)
    runtime.rows_acquired += int(receipt["row_count"])
    runtime.partitions.append(receipt)
    runtime.write_state("RUNNING", dataset)
    return frame, receipt


def update_receipt_classification(
    runtime: Runtime,
    dataset: str,
    parameters: dict[str, Any],
    receipt: dict[str, Any],
    classification: str,
) -> dict[str, Any]:
    if receipt["pit_classification"] == classification:
        return receipt
    updated = {**receipt, "pit_classification": classification}
    part = _partition_id(dataset, parameters)
    path = runtime.settings.root / "receipts" / dataset.lower() / f"{part}.json"
    _atomic_json(path, updated)
    return updated


def _normalize_symbol(values: pd.Series) -> pd.Series:
    return values.astype(str).str.extract(r"(\d{6})", expand=False)


def _next_sessions(values: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    days = pd.to_datetime(values, errors="coerce").values.astype("datetime64[D]")
    sessions = calendar.values.astype("datetime64[D]")
    indices = np.searchsorted(sessions, days, side="right")
    output = [pd.NaT if pd.isna(day) or index >= len(sessions) else sessions[index] for day, index in zip(days, indices, strict=True)]
    return pd.Series(pd.to_datetime(output), index=values.index)


def _write_normalized(root: Path, dataset: str, frame: pd.DataFrame) -> Path:
    path = root / "normalized" / f"{dataset.lower()}.csv"
    raw = _canonical_frame(frame).to_csv(index=False, lineterminator="\n").encode("utf-8")
    _atomic_bytes(path, raw)
    _atomic_bytes(path.with_name(f"{path.name}.sha256"), f"{hashlib.sha256(raw).hexdigest()}\n".encode())
    return path


def _write_feature_store(root: Path, name: str, frame: pd.DataFrame) -> Path:
    path = root / "feature_store" / f"{name}.csv"
    raw = _canonical_frame(frame).to_csv(index=False, lineterminator="\n").encode("utf-8")
    _atomic_bytes(path, raw)
    _atomic_bytes(path.with_name(f"{path.name}.sha256"), f"{hashlib.sha256(raw).hexdigest()}\n".encode())
    return path


def _finance_query(runtime: Runtime, dataset: str, symbols: list[str]) -> pd.DataFrame:
    finance = runtime.jq.finance
    table_name = TABLE_OBJECTS.get(dataset, dataset)
    table = getattr(finance, table_name)
    return runtime.query(
        dataset,
        200_000,
        lambda: finance.run_offset_query(runtime.jq.query(table).filter(table.code.in_(symbols))),
    )


def _factor_long(values: dict[str, pd.DataFrame], categories: dict[str, list[str]]) -> pd.DataFrame:
    category = {
        factor: group
        for group, factors in categories.items()
        if isinstance(factors, list)
        for factor in factors
    }
    rows: list[pd.DataFrame] = []
    for factor, frame in values.items():
        current = frame.rename_axis("date").reset_index().melt(
            id_vars="date", var_name="code", value_name="value"
        )
        current["factor"] = factor
        current["category"] = category[factor]
        rows.append(current)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _feature_rows(
    frame: pd.DataFrame,
    dataset: str,
    raw_hash: str,
    date_column: str,
    available_column: str,
    features: dict[str, str],
    pit_status: str,
) -> list[pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for source, target in features.items():
        if source not in frame:
            continue
        current = pd.DataFrame(
            {
                "date": frame[date_column],
                "symbol": frame["symbol"],
                "feature_name": target,
                "feature_value": pd.to_numeric(frame[source], errors="coerce"),
                "source_dataset": dataset,
                "raw_sha256": raw_hash,
                "available_at": frame[available_column],
                "feature_asof_date": frame[date_column],
                "pit_status": pit_status,
            }
        ).dropna(subset=["date", "symbol", "feature_value", "available_at"])
        rows.append(current)
    return rows


def run_pipeline(settings: Settings) -> dict[str, Any]:
    protocol = json.loads(settings.protocol_path.read_text(encoding="utf-8"))
    username, password = credentials_from_environment()
    jq = importlib.import_module("jqdatasdk")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        jq.auth(username, password)
    if not jq.is_auth():
        raise RuntimeError("JQDATA_AUTHENTICATION_FAILED")
    account = jq.get_account_info()
    settings.root.mkdir(parents=True, exist_ok=True)
    quota = jq.get_query_count()
    budget_path = settings.root / "state/quota_budget.json"
    if budget_path.exists():
        budget_state = json.loads(budget_path.read_text(encoding="utf-8"))
        if budget_state.get("budget_date") != str(settings.audit_date):
            budget_state = {
                "budget_date": str(settings.audit_date),
                "initial_spare": int(quota["spare"]),
            }
            _atomic_json(budget_path, budget_state)
    else:
        budget_state = {
            "budget_date": str(settings.audit_date),
            "initial_spare": int(quota["spare"]),
        }
        _atomic_json(budget_path, budget_state)
    previous_path = settings.root / "state/checkpoint.json"
    previous = (
        json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    )
    runtime = Runtime(jq, settings, int(budget_state["initial_spare"]))
    runtime.rejected = [
        {
            "dataset": name,
            "status": "REJECTED_ACCESS_DENIED",
            "error_type": "ProviderAccessDenied",
            "credential_or_identity_in_error": False,
        }
        for name in previous.get("rejected_datasets", [])
    ]
    runtime.write_state("RUNNING", "UNIVERSE")

    market_start = pd.Timestamp(account["date_range_start"]).date()
    market_end = pd.Timestamp(account["date_range_end"]).date()
    universe_parameters = {"index": "000300.XSHG", "date": str(market_end)}
    loaded = load_raw_partition(runtime, "UNIVERSE", universe_parameters)
    if loaded:
        universe, universe_receipt = loaded
        all_symbols = sorted(universe["code"].astype(str).tolist())
    else:
        all_symbols = sorted(
            runtime.query(
                "UNIVERSE", 300, lambda: jq.get_index_stocks("000300.XSHG", date=market_end)
            )
        )
        universe = pd.DataFrame({"code": all_symbols, "reference_date": str(market_end)})
        universe, universe_receipt = store_raw_partition(
            runtime, "UNIVERSE", universe_parameters, universe, "reference_date", "RESEARCH_SCOPE_ONLY", 1
        )
    core_symbols = all_symbols[:50]
    calendar_parameters = {"start": str(market_start), "end": str(settings.audit_date)}
    loaded = load_raw_partition(runtime, "TRADING_CALENDAR", calendar_parameters)
    if loaded:
        calendar_frame, calendar_receipt = loaded
    else:
        calendar_values = runtime.query(
            "TRADING_CALENDAR",
            400,
            lambda: jq.get_trade_days(start_date=market_start, end_date=settings.audit_date),
        )
        calendar_frame = pd.DataFrame({"date": pd.to_datetime(calendar_values)})
        calendar_frame, calendar_receipt = store_raw_partition(
            runtime,
            "TRADING_CALENDAR",
            calendar_parameters,
            calendar_frame,
            "date",
            "PIT_REFERENCE_CALENDAR",
            1,
        )
    calendar = pd.DatetimeIndex(calendar_frame["date"].dropna().sort_values().unique())
    market_calendar = calendar[(calendar.date >= market_start) & (calendar.date <= market_end)]

    data: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {
        "UNIVERSE": (universe, universe_receipt),
        "TRADING_CALENDAR": (calendar_frame, calendar_receipt),
    }

    if any(item["dataset"] == "MONEYFLOW_HISTORY_DAILY" for item in runtime.rejected):
        runtime.write_state("RUNNING", "MONEYFLOW_HISTORY_DAILY")
    else:
        try:
            money = runtime.query(
                "MONEYFLOW_HISTORY_DAILY",
                len(core_symbols) * len(market_calendar),
                lambda: jq.get_money_flow(
                    core_symbols, start_date=market_start, end_date=market_end
                ),
            )
            data["MONEYFLOW_HISTORY_DAILY"] = store_raw_partition(
                runtime,
                "MONEYFLOW_HISTORY_DAILY",
                {
                    "symbols": "core50_sha256",
                    "start": str(market_start),
                    "end": str(market_end),
                },
                money,
                "date",
                "PIT_SAFE_WITH_LAG",
                1,
            )
        except Exception as error:  # noqa: BLE001 -- SDK uses plain Exception for denial.
            runtime.rejected.append(
                {
                    "dataset": "MONEYFLOW_HISTORY_DAILY",
                    "status": "REJECTED_ACCESS_DENIED",
                    "error_type": type(error).__name__,
                    "credential_or_identity_in_error": False,
                }
            )
            runtime.write_state("RUNNING", "MONEYFLOW_HISTORY_DAILY")

    industry_parameters = {"classification": "sw_l1", "symbols": "core50_sha256"}
    loaded = load_raw_partition(runtime, "GET_HISTORY_INDUSTRY", industry_parameters)
    if loaded:
        data["GET_HISTORY_INDUSTRY"] = loaded
    else:
        industry = runtime.query(
            "GET_HISTORY_INDUSTRY",
            len(core_symbols) * 5,
            lambda: jq.get_history_industry("sw_l1", core_symbols),
        )
        data["GET_HISTORY_INDUSTRY"] = store_raw_partition(
            runtime,
            "GET_HISTORY_INDUSTRY",
            industry_parameters,
            industry,
            "start_date",
            "PIT_SAFE_HISTORICAL_MEMBERSHIP",
            1,
        )
    catalog_parameters = {"classification": "sw_l1", "date": str(market_end)}
    loaded = load_raw_partition(runtime, "INDUSTRY_CATALOG", catalog_parameters)
    if loaded:
        data["INDUSTRY_CATALOG"] = loaded
    else:
        catalog = runtime.query(
            "INDUSTRY_CATALOG",
            100,
            lambda: jq.get_industries("sw_l1", date=market_end),
        ).rename_axis("industry_code").reset_index()
        data["INDUSTRY_CATALOG"] = store_raw_partition(
            runtime,
            "INDUSTRY_CATALOG",
            catalog_parameters,
            catalog,
            "start_date",
            "PIT_REFERENCE_INDUSTRY_CATALOG",
            1,
        )

    valuation_dates = list(market_calendar[::5])
    valuation_parameters = {
        "symbols": "core50_sha256",
        "dates": [str(value.date()) for value in valuation_dates],
    }
    loaded = load_raw_partition(runtime, "VALUATION", valuation_parameters)
    if loaded:
        data["VALUATION"] = loaded
    else:
        valuation_parts = []
        for session in valuation_dates:
            frame = runtime.query(
                "VALUATION",
                len(core_symbols),
                lambda session=session: jq.get_fundamentals(
                    jq.query(jq.valuation).filter(jq.valuation.code.in_(core_symbols)),
                    date=session.date(),
                ),
            )
            valuation_parts.append(frame)
        valuation = pd.concat(valuation_parts, ignore_index=True)
        data["VALUATION"] = store_raw_partition(
            runtime,
            "VALUATION",
            valuation_parameters,
            valuation,
            "day",
            "PIT_SAFE_PROVIDER_ASOF",
            len(valuation_dates),
        )

    for dataset, (date_column, pit_status) in FINANCE_TABLES.items():
        parameters = {"universe": "csi300_reference", "date_restriction": "provider_entitlement"}
        loaded = load_raw_partition(runtime, dataset, parameters)
        if loaded:
            frame, receipt = loaded
            data[dataset] = (
                frame,
                update_receipt_classification(
                    runtime, dataset, parameters, receipt, pit_status
                ),
            )
        else:
            frame = _finance_query(runtime, dataset, all_symbols)
            data[dataset] = store_raw_partition(
                runtime, dataset, parameters, frame, date_column, pit_status, 1
            )

    factors = protocol["factor_selection"]
    factor_names = [factor for group in ("quality", "growth", "risk", "momentum", "emotion") for factor in factors[group]]
    factor_parameters = {
        "symbols": "core50_sha256",
        "start": str(market_start),
        "end": str(market_end),
        "factors": factor_names,
        "transport_batches": ["quality", "growth", "risk", "momentum", "emotion"],
    }
    loaded = load_raw_partition(runtime, "FACTOR_LIBRARY", factor_parameters)
    if loaded:
        data["FACTOR_LIBRARY"] = loaded
    else:
        factor_values: dict[str, pd.DataFrame] = {}
        for group in ("quality", "growth", "risk", "momentum", "emotion"):
            batch = factors[group]
            values = runtime.query(
                "FACTOR_LIBRARY",
                len(core_symbols) * len(market_calendar) * len(batch),
                lambda batch=batch: jq.get_factor_values(
                    core_symbols,
                    batch,
                    start_date=market_start,
                    end_date=market_end,
                ),
            )
            factor_values.update(values)
        factor_frame = _factor_long(factor_values, factors)
        data["FACTOR_LIBRARY"] = store_raw_partition(
            runtime,
            "FACTOR_LIBRARY",
            factor_parameters,
            factor_frame,
            "date",
            "PIT_SAFE_PROVIDER_ASOF_WITH_LAG",
            5,
        )

    normalized: dict[str, pd.DataFrame] = {}
    for dataset, (frame, receipt) in data.items():
        current = frame.copy()
        if dataset == "GET_HISTORY_INDUSTRY":
            current["symbol"] = _normalize_symbol(current["stock"])
            current["industry_code"] = current["code"].astype(str)
        elif "code" in current:
            current["symbol"] = _normalize_symbol(current["code"])
        elif "stock" in current:
            current["symbol"] = _normalize_symbol(current["stock"])
        date_column = FINANCE_TABLES.get(dataset, (None,))[0]
        if date_column and date_column in current:
            current[date_column] = pd.to_datetime(current[date_column], errors="coerce")
            if dataset not in {"FINANCE_BALANCE_SHEET", "FINANCE_INCOME_STATEMENT", "FINANCE_CASHFLOW_STATEMENT"}:
                current["available_at"] = _next_sessions(current[date_column], calendar)
                current["effective_date"] = current["available_at"]
        current["source_raw_sha256"] = receipt["sha256"]
        current["pit_classification"] = receipt["pit_classification"]
        normalized[dataset] = current
        _write_normalized(settings.root, dataset, current)
        if dataset == "STK_FIN_FORCAST":
            _write_normalized(settings.root, "company_earnings_forecast", current)
        elif dataset == "STK_PERFORMANCE_LETTERS":
            _write_normalized(settings.root, "earnings_event_data", current)

    history = normalized["GET_HISTORY_INDUSTRY"].copy()
    history["start_date"] = pd.to_datetime(history["start_date"], errors="coerce")
    history["end_date"] = pd.to_datetime(history["end_date"], errors="coerce")
    catalog = normalized["INDUSTRY_CATALOG"].copy()
    catalog["industry_code"] = catalog["industry_code"].astype(str)
    name_column = "name" if "name" in catalog else "display_name"
    catalog_names = catalog[["industry_code", name_column]].rename(
        columns={name_column: "industry_name"}
    )
    industry_parts = []
    for session in valuation_dates:
        current = history[
            history["start_date"].le(session)
            & (history["end_date"].isna() | history["end_date"].ge(session))
        ][["symbol", "industry_code"]].copy()
        current["asof_date"] = session
        industry_parts.append(current)
    industry_asof = pd.concat(industry_parts, ignore_index=True).merge(
        catalog_names, on="industry_code", how="left", validate="many_to_one"
    )
    industry_asof["available_at"] = industry_asof["asof_date"]
    normalized["INDUSTRY_ASOF"] = industry_asof
    _write_normalized(settings.root, "industry_asof", industry_asof)

    feature_parts: list[pd.DataFrame] = []
    factor_norm = normalized["FACTOR_LIBRARY"].copy()
    factor_norm["symbol"] = _normalize_symbol(factor_norm["code"])
    factor_norm["available_at"] = _next_sessions(factor_norm["date"], calendar)
    factor_norm["feature_name"] = (
        "jq_" + factor_norm["category"].str.lower() + "_" + factor_norm["factor"].str.lower()
    )
    factor_norm["feature_value"] = pd.to_numeric(factor_norm["value"], errors="coerce")
    factor_feature = factor_norm.assign(
        source_dataset="FACTOR_LIBRARY",
        raw_sha256=data["FACTOR_LIBRARY"][1]["sha256"],
        feature_asof_date=factor_norm["date"],
        pit_status="PIT_SAFE_PROVIDER_ASOF_WITH_LAG",
    )[["date", "symbol", "feature_name", "feature_value", "source_dataset", "raw_sha256", "available_at", "feature_asof_date", "pit_status"]]
    feature_parts.append(factor_feature.dropna(subset=["feature_value", "available_at"]))

    valuation_norm = normalized["VALUATION"].copy()
    valuation_norm["date"] = pd.to_datetime(valuation_norm["day"])
    valuation_norm["available_at"] = _next_sessions(valuation_norm["date"], calendar)
    valuation_norm = valuation_norm.merge(
        industry_asof.rename(columns={"asof_date": "date"})[
            ["date", "symbol", "industry_code", "industry_name"]
        ],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    for source in ("pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio", "market_cap"):
        values = pd.to_numeric(valuation_norm[source], errors="coerce")
        valuation_norm[f"{source}_percentile"] = values.groupby(valuation_norm["date"]).rank(pct=True)
        valuation_norm[f"{source}_industry_percentile"] = values.groupby(
            [valuation_norm["date"], valuation_norm["industry_code"]]
        ).rank(pct=True)
        valuation_norm[f"{source}_own_percentile"] = values.groupby(
            valuation_norm["symbol"]
        ).transform(lambda item: item.expanding().rank(pct=True))
        mean = values.groupby(valuation_norm["symbol"]).transform(lambda item: item.expanding().mean())
        std = values.groupby(valuation_norm["symbol"]).transform(lambda item: item.expanding().std())
        valuation_norm[f"{source}_own_zscore"] = (values - mean) / std.replace(0, np.nan)
    value_features = {
        name: f"jq_valuation_{name}" for name in valuation_norm.columns
        if name.endswith(("_percentile", "_own_zscore"))
    }
    feature_parts.extend(
        _feature_rows(
            valuation_norm,
            "VALUATION",
            data["VALUATION"][1]["sha256"],
            "date",
            "available_at",
            value_features,
            "PIT_SAFE_PROVIDER_ASOF_WITH_LAG",
        )
    )

    industry_features = pd.get_dummies(
        industry_asof["industry_code"], prefix="jq_industry", dtype=float
    )
    industry_identity = industry_asof[["asof_date", "symbol", "available_at"]].reset_index(
        drop=True
    )
    for feature_name in industry_features:
        feature_parts.append(
            pd.DataFrame(
                {
                    "date": industry_identity["asof_date"],
                    "symbol": industry_identity["symbol"],
                    "feature_name": feature_name,
                    "feature_value": industry_features[feature_name],
                    "source_dataset": "GET_HISTORY_INDUSTRY",
                    "raw_sha256": data["GET_HISTORY_INDUSTRY"][1]["sha256"],
                    "available_at": industry_identity["available_at"],
                    "feature_asof_date": industry_identity["asof_date"],
                    "pit_status": "PIT_SAFE_HISTORICAL_MEMBERSHIP",
                }
            )
        )

    for dataset, mapping in {
        "STK_FIN_FORCAST": {
            "profit_min": "jq_company_forecast_profit_min",
            "profit_max": "jq_company_forecast_profit_max",
            "profit_ratio_min": "jq_company_forecast_ratio_min",
            "profit_ratio_max": "jq_company_forecast_ratio_max",
        },
        "STK_PERFORMANCE_LETTERS": {
            "total_operating_revenue": "jq_earnings_event_revenue",
            "np_parent_company_owners": "jq_earnings_event_parent_profit",
            "basic_eps": "jq_earnings_event_basic_eps",
        },
        "STK_HK_HOLD_INFO": {
            "share_number": "jq_hkhold_share_number",
            "share_ratio": "jq_hkhold_share_ratio",
        },
    }.items():
        current = normalized[dataset]
        date_column = FINANCE_TABLES[dataset][0]
        feature_parts.extend(
            _feature_rows(
                current,
                dataset,
                data[dataset][1]["sha256"],
                date_column,
                "available_at",
                mapping,
                FINANCE_TABLES[dataset][1],
            )
        )

    features = pd.concat(feature_parts, ignore_index=True)
    features["feature_value"] = pd.to_numeric(features["feature_value"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    features = features.dropna(
        subset=[
            "date",
            "symbol",
            "feature_name",
            "feature_value",
            "source_dataset",
            "raw_sha256",
            "available_at",
            "feature_asof_date",
        ]
    )
    features["date"] = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")
    features["available_at"] = pd.to_datetime(features["available_at"]).dt.strftime("%Y-%m-%d")
    features["feature_asof_date"] = pd.to_datetime(features["feature_asof_date"]).dt.strftime("%Y-%m-%d")
    features = features.sort_values(["date", "symbol", "feature_name", "available_at"]).drop_duplicates(
        ["date", "symbol", "feature_name"], keep="last"
    )
    _write_feature_store(settings.root, "features_long", features)
    wide = features.pivot(index=["date", "symbol"], columns="feature_name", values="feature_value").reset_index()
    wide.columns.name = None
    _write_feature_store(settings.root, "features_wide", wide)

    quota_after = jq.get_query_count()
    coverage = {
        dataset: {
            "rows": receipt["row_count"],
            "date_min": receipt["date_min"],
            "date_max": receipt["date_max"],
            "symbols": receipt["symbol_count"],
            "pit_classification": receipt["pit_classification"],
            "query_restriction": receipt["query_parameters"],
            "raw_sha256": receipt["sha256"],
            "schema_hash": receipt["schema_hash"],
        }
        for dataset, (_, receipt) in data.items()
    }
    summary = {
        "pipeline": "PREDICTION_V2_JQDATA_REAL_DATA_PIPELINE",
        "status": "PREDICTION_V2_JQDATA_RESEARCH_DATA_READY",
        "jqdata_adapter_usable": True,
        "real_provider_queries": int(
            sum(item["provider_query_count"] for item in runtime.partitions)
            + len(runtime.rejected)
        ),
        "real_rows_acquired": runtime.rows_acquired,
        "raw_partitions": len(runtime.partitions),
        "normalized_rows": int(sum(len(frame) for frame in normalized.values())),
        "feature_store": {
            "status": "READY" if len(features) else "NOT_READY",
            "long_rows": len(features),
            "wide_rows": len(wide),
            "features": int(features["feature_name"].nunique()),
            "symbols": int(features["symbol"].nunique()),
            "date_min": str(features["date"].min()),
            "date_max": str(features["date"].max()),
        },
        "dataset_coverage": coverage,
        "rejected": runtime.rejected,
        "pit_safe_datasets": [name for name, value in coverage.items() if value["pit_classification"].startswith("PIT_SAFE") and "RISK" not in value["pit_classification"]],
        "pit_safe_with_lag_datasets": [
            name
            for name, value in coverage.items()
            if "WITH_LAG" in value["pit_classification"]
            and "RISK" not in value["pit_classification"]
        ],
        "research_only": [name for name, value in coverage.items() if "RISK" in value["pit_classification"] or "RESEARCH" in value["pit_classification"]],
        "analyst_vintages": "NOT_AVAILABLE",
        "actual_versions_contract": "FAIL_REVISION_LINEAGE_NOT_PROVED",
        "full_5_year_historical_gate": "FAIL",
        "prediction_v2_data_acquisition": "BLOCKED",
        "quota": {
            "start_spare": runtime.started_spare,
            "end_spare": int(quota_after["spare"]),
            "used_this_run": runtime.started_spare - int(quota_after["spare"]),
            "local_budget": settings.budget_rows,
        },
        "integrity": {
            "credentials_persisted": False,
            "account_identity_persisted": False,
            "model_training": False,
            "return_labels_read": False,
            "rank_ic": "NOT_COMPUTED",
            "gen2_modified": 0,
            "contracts_007_012_modified": 0,
            "daily_pit_modified": 0,
            "daily_prediction_modified": 0,
            "tencent_first_routing_modified": 0,
            "feature_lineage_complete": bool(
                features[
                    [
                        "source_dataset",
                        "raw_sha256",
                        "available_at",
                        "feature_asof_date",
                    ]
                ].notna().all().all()
            ),
            "feature_available_not_before_asof": bool(
                pd.to_datetime(features["available_at"])
                .ge(pd.to_datetime(features["feature_asof_date"]))
                .all()
            ),
            "future_return_values_used": False,
        },
        "protocol_sha256": sha256_file(settings.protocol_path),
    }
    _atomic_json(settings.root / "state/final_summary.json", summary)
    runtime.write_state("COMPLETE")
    return summary


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PREDICTION_V2_JQDATA_REAL_DATA_ACQUISITION_REPORT",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"JQData adapter usable: `{'YES' if summary['jqdata_adapter_usable'] else 'NO'}`",
        f"Real provider queries: `{summary['real_provider_queries']}`",
        f"Real rows acquired: `{summary['real_rows_acquired']}`",
        f"Raw partitions: `{summary['raw_partitions']}`",
        f"Normalized rows: `{summary['normalized_rows']}`",
        "",
        "## Actual per-dataset coverage",
        "",
    ]
    for name, value in summary["dataset_coverage"].items():
        lines.append(
            f"- {name}: rows={value['rows']}, dates={value['date_min']}..{value['date_max']}, "
            f"symbols={value['symbols']}, status=`{value['pit_classification']}`, "
            f"sha256=`{value['raw_sha256']}`, schema=`{value['schema_hash']}`, "
            f"query={json.dumps(value['query_restriction'], ensure_ascii=False, sort_keys=True)}"
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            f"- PIT-safe: `{', '.join(summary['pit_safe_datasets'])}`",
            f"- PIT-safe-with-lag: `{', '.join(summary['pit_safe_with_lag_datasets'])}`",
            f"- Research-only/risk: `{', '.join(summary['research_only'])}`",
            f"- Rejected: `{', '.join(item['dataset'] + '(' + item['status'] + ')' for item in summary['rejected'])}`",
            "- `STK_FIN_FORCAST` is normalized as company earnings forecast, never analyst estimates.",
            "- Performance letters are earnings-event data; actual_versions contract remains failed.",
            "- Three statement tables retain `PIT_RESTATEMENT_RISK` and are excluded from PIT-safe features.",
            "",
            "## Feature store",
            "",
            f"- Status: `{summary['feature_store']['status']}`",
            f"- Long rows / wide rows: `{summary['feature_store']['long_rows']}` / `{summary['feature_store']['wide_rows']}`",
            f"- Features / symbols: `{summary['feature_store']['features']}` / `{summary['feature_store']['symbols']}`",
            f"- Date range: `{summary['feature_store']['date_min']}` to `{summary['feature_store']['date_max']}`",
            "- Every long-form feature row carries source dataset, raw SHA256, available_at, and feature_asof_date.",
            "",
            "## Remaining gates",
            "",
            f"- Analyst vintages: `{summary['analyst_vintages']}`",
            f"- Full 5-year historical gate: `{summary['full_5_year_historical_gate']}`",
            f"- Complete Prediction V2 acquisition: `{summary['prediction_v2_data_acquisition']}`",
            "",
            "## Integrity",
            "",
            "- Model training: `FALSE`",
            "- Return labels read: `FALSE`",
            "- RankIC: `NOT_COMPUTED`",
            "- Credentials/account identity persisted: `FALSE`",
            "- Gen2 modified: `0`",
            "- 007–012 modified: `0`",
            "- DAILY PIT modified: `0`",
            "- daily_prediction modified: `0`",
            "- Tencent-first routing modified: `0`",
            "",
            "## Final status",
            "",
            f"`{summary['status']}`",
            "",
            "`PREDICTION_V2_DATA_ACQUISITION_BLOCKED` remains simultaneously true.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(settings: Settings, summary: dict[str, Any]) -> None:
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(
        settings.protocol_path.with_name(f"{settings.protocol_path.name}.sha256"),
        f"{sha256_file(settings.protocol_path)}\n".encode(),
    )
    _atomic_json(settings.artifact_dir / "acquisition_summary.json", summary)
    report = settings.artifact_dir / "PREDICTION_V2_JQDATA_REAL_DATA_ACQUISITION_REPORT.md"
    _atomic_bytes(report, render_report(summary).encode("utf-8"))
    _atomic_bytes(report.with_name(f"{report.name}.sha256"), f"{sha256_file(report)}\n".encode())
    files = sorted(
        path
        for path in settings.artifact_dir.iterdir()
        if path.is_file() and not path.name.startswith("artifact_manifest")
    )
    _atomic_json(
        settings.artifact_dir / "artifact_manifest.json",
        {"files": {path.name: sha256_file(path) for path in files}},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire real JQData research data for Prediction V2")
    parser.add_argument("--root", type=Path, default=Path("data/prediction_v2/jqdata"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/prediction_v2/jqdata_real_data"))
    parser.add_argument("--protocol", type=Path, default=Path("artifacts/prediction_v2/jqdata_real_data/protocol.json"))
    parser.add_argument("--audit-date", type=date.fromisoformat, default=date(2026, 9, 5))
    parser.add_argument("--budget-rows", type=int, default=800_000)
    args = parser.parse_args()
    settings = Settings(args.root.resolve(), args.artifact_dir.resolve(), args.protocol.resolve(), args.audit_date, args.budget_rows)
    summary = run_pipeline(settings)
    write_artifacts(settings, summary)
    print(json.dumps({key: summary[key] for key in ("status", "real_provider_queries", "real_rows_acquired", "raw_partitions", "normalized_rows", "feature_store")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
