from __future__ import annotations

from pathlib import Path

import pandas as pd

from .integrity import read_verified_json, verify_immutable, write_immutable_frame, write_immutable_json


FEATURE_KEYS = ["date", "symbol"]


def build_feature_panel(
    universe: pd.DataFrame,
    *,
    date: str,
    observation_id: str,
    observation_hash: str,
    expectations: pd.DataFrame | None = None,
    announcements: pd.DataFrame | None = None,
    fund_flows: pd.DataFrame | None = None,
    revision: pd.DataFrame | None = None,
    source_provenance: dict | None = None,
) -> pd.DataFrame:
    required = {"symbol", "industry", "universe_member"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"feature universe missing {sorted(missing)}")
    panel = universe[["symbol", "industry", "universe_member"]].copy()
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel.insert(0, "date", date)
    panel["observation_id"] = observation_id
    panel["observation_sha256"] = observation_hash

    if expectations is not None:
        values = expectations.copy()
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        values["expectation_level"] = pd.to_numeric(values["forecast_eps_1"], errors="coerce")
        panel = panel.merge(values[["symbol", "expectation_level"]], on="symbol", how="left", validate="one_to_one")
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

    if announcements is not None:
        required_announcement = {"symbol", "announcement_event_count", "announcement_available"}
        missing = required_announcement - set(announcements.columns)
        if missing:
            raise ValueError(f"announcement features missing {sorted(missing)}")
        values = announcements[list(required_announcement)].copy()
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        values["announcement_event_count"] = pd.to_numeric(values["announcement_event_count"], errors="coerce")
        confirmed = values["announcement_available"].fillna(False).astype(bool)
        if values.loc[confirmed, "announcement_event_count"].isna().any():
            raise ValueError("confirmed announcement queries require a real count, including zero")
        if values.loc[~confirmed, "announcement_event_count"].notna().any():
            raise ValueError("unconfirmed announcement queries cannot carry a count")
        panel = panel.merge(values, on="symbol", how="left", validate="one_to_one")
        panel["announcement_available"] = panel["announcement_available"].map(
            lambda value: bool(value) if pd.notna(value) else False
        ).astype(bool)
        panel.loc[~panel["announcement_available"], "announcement_event_count"] = pd.NA
    else:
        panel["announcement_event_count"] = pd.NA
        panel["announcement_available"] = False

    if fund_flows is not None:
        values = fund_flows.copy()
        values["symbol"] = values["symbol"].astype(str).str.zfill(6)
        flow_columns = ["main_net_inflow", "main_net_inflow_ratio"]
        for name in flow_columns:
            if name not in values:
                values[name] = pd.NA
            values[name] = pd.to_numeric(values[name], errors="coerce")
        panel = panel.merge(values[["symbol", *flow_columns]], on="symbol", how="left", validate="one_to_one")
        panel["fund_flow_available"] = panel[flow_columns].notna().any(axis=1)
    else:
        panel["main_net_inflow"] = pd.NA
        panel["main_net_inflow_ratio"] = pd.NA
        panel["fund_flow_available"] = False

    panel["source_provenance"] = str(source_provenance or {})
    panel["missing_is_not_zero"] = True
    return panel.sort_values(FEATURE_KEYS).reset_index(drop=True)


def write_feature_panel(panel: pd.DataFrame, root: str | Path, *, source_provenance: dict) -> dict:
    dates = panel["date"].astype(str).unique()
    if len(dates) != 1:
        raise ValueError("feature panel must contain exactly one date")
    date = dates[0]
    target = Path(root) / "panels" / f"{date}.csv"
    digest = write_immutable_frame(target, panel, FEATURE_KEYS)
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
        "replacement_evaluation_ready": False,
        "production_prediction_ready": False,
        "execution_authorized": False,
    }
    manifest_path = Path(root) / "manifests" / f"{date}.json"
    write_immutable_json(manifest_path, manifest)
    return manifest | {"manifest_path": manifest_path.as_posix(), "manifest_sha256": verify_immutable(manifest_path)}


def verify_feature_panel(manifest_path: str | Path) -> dict:
    manifest = read_verified_json(manifest_path)
    actual = verify_immutable(manifest["panel_path"])
    if actual != manifest["panel_sha256"]:
        raise RuntimeError("feature manifest does not match panel")
    return {
        "intact": True,
        "manifest_sha256": verify_immutable(manifest_path),
        "panel_sha256": actual,
    }
