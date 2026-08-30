from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .immutable import sha256_file, write_new_frame, write_new_json, verify_sidecar


FEATURE_KEYS = ["date", "symbol"]


def build_feature_panel(
    universe: pd.DataFrame,
    expectations: pd.DataFrame | None,
    *,
    date: str,
    observation_id: str,
    observation_hash: str,
    announcement_counts: pd.DataFrame | None = None,
    fund_flows: pd.DataFrame | None = None,
    revision: pd.DataFrame | None = None,
    qualifying_trading_observation: bool,
) -> pd.DataFrame:
    required = {"symbol", "industry", "universe_member"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"feature universe missing {sorted(missing)}")
    panel = universe[["symbol", "industry", "universe_member"]].copy()
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel.insert(0, "date", date)
    panel["observation_id"] = observation_id
    panel["qualifying_trading_observation"] = bool(qualifying_trading_observation)
    if expectations is not None:
        expectation = expectations.copy()
        expectation["symbol"] = expectation["symbol"].astype(str).str.zfill(6)
        expectation["expectation_level"] = pd.to_numeric(expectation["forecast_eps_1"], errors="coerce")
        panel = panel.merge(
            expectation[["symbol", "expectation_level"]], on="symbol", how="left", validate="one_to_one"
        )
        panel["expectation_available"] = panel["expectation_level"].notna()
    else:
        panel["expectation_level"] = pd.NA
        panel["expectation_available"] = False
    revision_columns = [
        "symbol", "expectation_revision_pct", "expectation_revision_rank",
        "relative_revision_vs_industry", "revision_available",
    ]
    if revision is not None:
        panel = panel.merge(revision[revision_columns], on="symbol", how="left", validate="one_to_one")
        panel["revision_available"] = panel["revision_available"].fillna(False).astype(bool)
    else:
        panel["expectation_revision_pct"] = pd.NA
        panel["expectation_revision_rank"] = pd.NA
        panel["relative_revision_vs_industry"] = pd.NA
        panel["revision_available"] = False
    if announcement_counts is not None:
        values = announcement_counts[["symbol", "announcement_event_count"]].copy()
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        panel = panel.merge(values, on="symbol", how="left", validate="one_to_one")
        panel["announcement_event_count"] = panel["announcement_event_count"].fillna(0).astype(int)
        panel["announcement_available"] = True
    else:
        panel["announcement_event_count"] = pd.NA
        panel["announcement_available"] = False
    if fund_flows is not None:
        values = fund_flows.copy()
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        columns = [name for name in ("symbol", "main_net_inflow", "main_net_inflow_ratio") if name in values]
        panel = panel.merge(values[columns], on="symbol", how="left", validate="one_to_one")
        panel["fund_flow_available"] = panel[columns[1:]].notna().any(axis=1)
    else:
        panel["main_net_inflow"] = pd.NA
        panel["main_net_inflow_ratio"] = pd.NA
        panel["fund_flow_available"] = False
    panel["source_freshness"] = "CURRENT_OBSERVATION"
    panel["observation_sha256"] = observation_hash
    panel["missing_is_not_zero"] = True
    return panel.sort_values(FEATURE_KEYS).reset_index(drop=True)


def write_feature_panel(
    panel: pd.DataFrame,
    root: str | Path,
    *,
    source_provenance: dict,
) -> dict:
    dates = panel["date"].astype(str).unique()
    if len(dates) != 1:
        raise ValueError("feature panel must contain exactly one date")
    date = dates[0]
    target = Path(root) / "panels" / f"{date}.csv"
    digest = write_new_frame(target, panel, FEATURE_KEYS)
    manifest = {
        "date": date,
        "panel_path": target.as_posix(),
        "panel_sha256": digest,
        "rows": len(panel),
        "symbols": panel["symbol"].nunique(),
        "feature_availability": {
            name: int(panel[name].fillna(False).sum())
            for name in panel.columns if name.endswith("_available")
        },
        "source_provenance": source_provenance,
        "append_only": True,
        "model_training_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    write_new_json(Path(root) / "manifests" / f"{date}.json", manifest)
    return manifest


def verify_feature_panel(manifest_path: str | Path) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual = verify_sidecar(manifest["panel_path"])
    if actual != manifest["panel_sha256"]:
        raise RuntimeError("feature manifest does not match panel")
    return {"intact": True, "panel_sha256": actual}
