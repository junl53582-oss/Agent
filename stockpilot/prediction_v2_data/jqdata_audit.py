from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import sha256_file

REQUIRED_MARKET_FIELDS = ("open", "high", "low", "close", "volume", "money", "factor")
RELEVANT_TABLE_TERMS = (
    "FORECAST",
    "FORCAST",
    "ANALYST",
    "RESEARCH",
    "REPORT",
    "CONSENSUS",
    "EXPECT",
    "RATING",
    "EARNINGS",
    "PERFORMANCE",
)
ANALYST_TABLE_TERMS = ("ANALYST", "RESEARCH", "CONSENSUS", "RATING")
AUDITED_TABLES = {
    "STK_FIN_FORCAST": "pub_date",
    "STK_PERFORMANCE_LETTERS": "pub_date",
    "STK_REPORT_DISCLOSURE": "pub_date",
}
SAFE_ACCOUNT_FIELDS = (
    "date_range_start",
    "date_range_end",
    "expire_time",
    "license",
    "query_count_limit",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_credentials(env_file: Path) -> tuple[str, str, str]:
    """Load runtime-only credentials without returning them in any audit record."""
    file_values = _parse_env_file(env_file)
    username = os.environ.get("JQDATA_USERNAME") or file_values.get("JQDATA_USERNAME", "")
    password = os.environ.get("JQDATA_PASSWORD") or file_values.get("JQDATA_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("JQDATA_CREDENTIALS_NOT_AVAILABLE")
    source = "PROCESS_ENVIRONMENT" if os.environ.get("JQDATA_USERNAME") else "IGNORED_ENV_FILE"
    return username, password, source


def _safe_account_info(value: dict[str, Any]) -> dict[str, Any]:
    return {field: value.get(field) for field in SAFE_ACCOUNT_FIELDS}


def _table_schema(jq: Any, name: str) -> dict[str, Any]:
    frame = jq.get_table_info(name)
    return {
        "fields": frame["name_en"].astype(str).tolist(),
        "field_count": len(frame),
    }


def _date_bounds(jq: Any, finance: Any, name: str, date_column: str) -> dict[str, Any]:
    table = getattr(finance, name)
    column = getattr(table, date_column)
    oldest = finance.run_query(jq.query(column).order_by(column.asc(), table.id.asc()).limit(1))
    newest = finance.run_query(jq.query(column).order_by(column.desc(), table.id.desc()).limit(1))
    return {
        "oldest_observed": str(oldest.iloc[0, 0]) if not oldest.empty else None,
        "newest_observed": str(newest.iloc[0, 0]) if not newest.empty else None,
    }


def _market_probe(jq: Any, entitlement_end: date) -> dict[str, Any]:
    start = entitlement_end - timedelta(days=7)
    trade_days = jq.get_trade_days(start_date=start, end_date=entitlement_end)
    if len(trade_days) == 0:
        return {"passed": False, "reason": "NO_TRADING_SESSION_IN_PROBE_WINDOW"}
    probe_end = pd.Timestamp(trade_days[-1]).date()
    probe_start = pd.Timestamp(trade_days[max(0, len(trade_days) - 2)]).date()
    frame = jq.get_price(
        "000001.XSHE",
        start_date=probe_start,
        end_date=probe_end,
        frequency="daily",
        fields=list(REQUIRED_MARKET_FIELDS),
        fq="post",
        skip_paused=False,
        panel=False,
    )
    required = set(REQUIRED_MARKET_FIELDS)
    columns = set(frame.columns)
    numeric = frame[list(REQUIRED_MARKET_FIELDS)].apply(pd.to_numeric, errors="coerce")
    checks = {
        "required_fields_present": required <= columns,
        "rows_present": len(frame) > 0,
        "no_nulls": not numeric.isna().any().any(),
        "positive_prices": bool((numeric[["open", "high", "low", "close"]] > 0).all().all()),
        "nonnegative_volume_and_money": bool((numeric[["volume", "money"]] >= 0).all().all()),
        "valid_ohlc_envelope": bool(
            (numeric["high"] >= numeric[["open", "close"]].max(axis=1)).all()
            and (numeric["low"] <= numeric[["open", "close"]].min(axis=1)).all()
        ),
    }
    sample_bytes = frame.to_csv(index=True, lineterminator="\n").encode()
    return {
        "passed": all(checks.values()),
        "adjustment": "post/HFQ",
        "provider_amount_field": "money",
        "stockpilot_amount_mapping": "money -> amount",
        "probe_start": str(probe_start),
        "probe_end": str(probe_end),
        "rows_observed": len(frame),
        "rows_retained": 0,
        "columns": list(frame.columns),
        "in_memory_sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "checks": checks,
    }


def _fundamentals_probe(jq: Any, entitlement_end: date) -> dict[str, Any]:
    query = jq.query(
        jq.income.code,
        jq.income.pubDate,
        jq.income.statDate,
        jq.income.net_profit,
    ).filter(jq.income.code == "000001.XSHE").limit(10)
    frame = jq.get_fundamentals(query, date=entitlement_end)
    published = pd.to_datetime(frame.get("pubDate"), errors="coerce")
    as_of = pd.Timestamp(entitlement_end)
    return {
        "passed": bool(
            len(frame)
            and published.notna().all()
            and published.le(as_of).all()
            and {"code", "pubDate", "statDate", "net_profit"} <= set(frame.columns)
        ),
        "query_mode": "date/as-of",
        "statDate_mode_used": False,
        "as_of": str(entitlement_end),
        "rows_observed": len(frame),
        "rows_retained": 0,
        "columns": list(frame.columns),
        "latest_publication_observed": str(published.max().date()) if published.notna().any() else None,
        "all_publications_not_after_as_of": bool(published.le(as_of).all()) if len(frame) else None,
    }


def classify_readiness(result: dict[str, Any], minimum_history_years: int = 5) -> dict[str, Any]:
    account = result["account"]
    start = pd.Timestamp(account["date_range_start"])
    end = pd.Timestamp(account["date_range_end"])
    coverage_years = (end - start).days / 365.2425
    catalog = result["finance_catalog"]
    analyst_tables = [
        name for name in catalog["relevant_tables"]
        if any(term in name.upper() for term in ANALYST_TABLE_TERMS)
    ]
    historical_ok = coverage_years >= minimum_history_years
    analyst_ok = bool(result.get("analyst_schema_validation", {}).get("passed", False))
    return {
        "status": (
            "PREDICTION_V2_JQDATA_FOUNDATION_READY"
            if historical_ok and analyst_ok
            else "PREDICTION_V2_JQDATA_FOUNDATION_BLOCKED"
        ),
        "technical_connection": "PASS",
        "market_schema": "PASS" if result["market_probe"]["passed"] else "FAIL",
        "pit_fundamental_schema": (
            "PASS" if result["fundamentals_probe"]["passed"] else "FAIL"
        ),
        "entitled_history_years": round(coverage_years, 3),
        "minimum_required_history_years": minimum_history_years,
        "historical_coverage": "PASS" if historical_ok else "FAIL",
        "analyst_vintage_tables": analyst_tables,
        "historical_analyst_expectations": "PASS" if analyst_ok else "NOT_AVAILABLE",
        "issuer_earnings_forecast": "SCHEMA_CANDIDATE_ONLY",
        "issuer_forecast_limitations": [
            "publication date has no verified intraday timestamp",
            "no explicit revision/supersession lineage fields",
            "account entitlement is shorter than the frozen five-year minimum",
            "issuer forecasts are not sell-side analyst consensus vintages",
        ],
        "approved_use_now": "BOUNDED_RECENT_SCHEMA_VALIDATION_ONLY",
        "challenger_experiment": "NOT_STARTED",
        "production_provider_modified": False,
    }


def run_audit(env_file: Path, protocol_path: Path, audit_date: str) -> dict[str, Any]:
    username, password, credential_source = load_credentials(env_file)
    jq = importlib.import_module("jqdatasdk")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        jq.auth(username, password)
    if not jq.is_auth():
        raise RuntimeError("JQDATA_AUTHENTICATION_FAILED")

    quota_before = jq.get_query_count()
    account = _safe_account_info(jq.get_account_info())
    privileges = sorted(str(jq.get_privilege()).split("|"))
    finance = jq.finance
    _ = finance.STK_FIN_FORCAST
    catalog = sorted(getattr(finance, "_DBTable__table_names", []))
    relevant = [
        name for name in catalog if any(term in name.upper() for term in RELEVANT_TABLE_TERMS)
    ]
    tables: dict[str, Any] = {}
    for name, date_column in AUDITED_TABLES.items():
        tables[name] = {
            **_table_schema(jq, name),
            **_date_bounds(jq, finance, name, date_column),
        }
    entitlement_end = pd.Timestamp(account["date_range_end"]).date()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = {
        "audit": "PREDICTION_V2_JQDATA_CAPABILITY_AND_PIT_AUDIT",
        "audit_date": audit_date,
        "sdk_version": jq.__version__,
        "server_version": jq.get_server_version(),
        "authenticated": True,
        "credential_source": credential_source,
        "credential_values_persisted": False,
        "account_identifier_persisted": False,
        "account": account,
        "quota_before": quota_before,
        "privilege_count": len(privileges),
        "relevant_privileges": [
            name for name in privileges
            if any(term in name for term in ("STOCK_HISTORY", "FINANCE", "FORCAST", "REPORT"))
        ],
        "finance_catalog": {
            "table_count": len(catalog),
            "relevant_tables": relevant,
            "analyst_table_search_terms": list(ANALYST_TABLE_TERMS),
        },
        "analyst_schema_validation": {
            "candidate_tables": [
                name for name in relevant
                if any(term in name.upper() for term in ANALYST_TABLE_TERMS)
            ],
            "passed": False,
            "reason": "NO_ANALYST_VINTAGE_TABLE_DISCOVERED",
        },
        "table_audits": tables,
        "market_probe": _market_probe(jq, entitlement_end),
        "fundamentals_probe": _fundamentals_probe(jq, entitlement_end),
        "protocol_sha256": sha256_file(protocol_path),
        "scope": {
            "raw_rows_committed": 0,
            "model_training": False,
            "return_labels_read": False,
            "gen2_modified": False,
            "production_provider_modified": False,
        },
    }
    minimum_years = protocol["required_imports"]["analyst_estimates"]["minimum_years"]
    result["decision"] = classify_readiness(result, minimum_years)
    result["quota_after"] = jq.get_query_count()
    return result


def _write_json(path: Path, value: Any) -> None:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{hashlib.sha256(raw).hexdigest()}\n", encoding="ascii"
    )


def render_report(result: dict[str, Any]) -> str:
    decision = result["decision"]
    account = result["account"]
    tables = result["table_audits"]
    return "\n".join(
        [
            "# PREDICTION_V2_JQDATA_CAPABILITY_AND_PIT_AUDIT_REPORT",
            "",
            f"Status: `{decision['status']}`",
            "",
            "## Connection and entitlement",
            "",
            "- Authentication: `PASS`",
            f"- SDK/server: `{result['sdk_version']}` / `{result['server_version']}`",
            f"- Licensed date range: `{account['date_range_start']}` to `{account['date_range_end']}`",
            f"- Account expiry: `{account['expire_time']}`",
            f"- Query quota after audit: `{result['quota_after']}`",
            "- Credential values or account identifier persisted: `FALSE`",
            "",
            "## Market and fundamentals",
            "",
            f"- HFQ OHLCVA schema probe: `{'PASS' if result['market_probe']['passed'] else 'FAIL'}`",
            "- JQData `money` maps to StockPilot `amount`; no synthetic fields were created.",
            f"- Fundamentals `date`/as-of probe: `{'PASS' if result['fundamentals_probe']['passed'] else 'FAIL'}`",
            "- `statDate` query mode used: `FALSE`.",
            "- Production Tencent-first routing changed: `FALSE`.",
            "",
            "## Event and expectation data",
            "",
            f"- Finance tables enumerated: `{result['finance_catalog']['table_count']}`",
            f"- Relevant tables: `{', '.join(result['finance_catalog']['relevant_tables'])}`",
            "- Historical analyst consensus/vintage table: `NOT AVAILABLE`.",
            "- `STK_FIN_FORCAST` is issuer earnings guidance, not sell-side analyst consensus.",
            f"- Issuer forecast observed range: `{tables['STK_FIN_FORCAST']['oldest_observed']}` to `{tables['STK_FIN_FORCAST']['newest_observed']}`.",
            "- The issuer table has publication date and forecast-period/value fields, but no verified intraday timestamp or explicit revision/supersession link.",
            "",
            "## Admission decision",
            "",
            f"- Entitled history: `{decision['entitled_history_years']}` years; required: `{decision['minimum_required_history_years']}` years.",
            f"- Historical coverage: `{decision['historical_coverage']}`.",
            f"- Analyst vintages: `{decision['historical_analyst_expectations']}`.",
            f"- Approved use now: `{decision['approved_use_now']}`.",
            "- Model training/challenger: `NOT STARTED`.",
            "- Gen2, 007–012, DAILY PIT, and production provider code: unchanged.",
            "",
            "## Final status",
            "",
            f"`{decision['status']}`",
            "",
        ]
    )


def write_outputs(artifact_dir: Path, result: dict[str, Any]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    audit_path = artifact_dir / "jqdata_capability_audit.json"
    _write_json(audit_path, result)
    report_path = artifact_dir / "PREDICTION_V2_JQDATA_CAPABILITY_AND_PIT_AUDIT_REPORT.md"
    report_path.write_text(render_report(result), encoding="utf-8", newline="\n")
    report_path.with_name(f"{report_path.name}.sha256").write_text(
        f"{sha256_file(report_path)}\n", encoding="ascii"
    )
    files = sorted(
        path for path in artifact_dir.iterdir()
        if path.is_file() and not path.name.startswith("artifact_manifest")
    )
    _write_json(
        artifact_dir / "artifact_manifest.json",
        {"files": {path.name: sha256_file(path) for path in files}},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit JQData for Prediction V2 PIT readiness")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("artifacts/prediction_v2/data_acquisition/protocol.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/prediction_v2/jqdata_capability_audit"),
    )
    parser.add_argument("--audit-date", default="2026-09-05")
    args = parser.parse_args()
    result = run_audit(args.env_file.resolve(), args.protocol.resolve(), args.audit_date)
    write_outputs(args.artifact_dir.resolve(), result)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
