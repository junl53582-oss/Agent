import json
import os
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research_v15.features import load_event_documents
from research_v16.data import load_v16_dataset
from research_v16.model import fit_v16_models, score_v16
from research_v16.text_model import EnsembleTextCorpus
from research_v20.freeze import digest, write_new
from research_v20.timing import historical_market_state, weights_for_momentum
from research_v20r2.config import V20R2Settings
from research_v20r2.ledger import evaluation_schedule
from .diagnostics import COMPONENTS, TARGETS, check_legacy_reproduction, component_diagnostics, summary_table
from .freeze import DIRECTORY, PARENT, verify


def progress(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    target = DIRECTORY / "runtime_status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_diagnosis(dataset, corpus, reference, settings, callback=progress, output_dir=DIRECTORY):
    schedule = evaluation_schedule(dataset, settings)
    dates = [row[0] for row in schedule]
    if set(dates) != set(reference.date):
        raise ValueError("diagnostic dates differ from parent evaluation")
    market = historical_market_state(dataset, settings)
    scope = dataset[dataset.in_universe.eq(True) & dataset.date.isin(dates)]
    cache, metric_parts, reproduction, annual_files = {}, [], [], {}
    for year in settings.test_years:
        callback("fitting", test_year=int(year))
        models = fit_v16_models(dataset, corpus, year, settings, cache)
        _, v5, v4 = cache[year]
        traces, year_reproduction = [], []
        for date in [value for value in dates if value.year == year]:
            callback("component_diagnosis", test_year=int(year), date=str(date.date()))
            current = score_v16(scope[scope.date.eq(date)].copy(), models, v5, v4, settings)
            base_weight, regime = weights_for_momentum(float(market.loc[date, "market_momentum"]), settings)
            current["adaptive_score"] = base_weight * current.v13_comparable_score + (1 - base_weight) * current.text_event_score
            proof = check_legacy_reproduction(current, reference[reference.date.eq(date)])
            year_reproduction.append({"date": str(date.date()), "checks": proof})
            metric_parts.append(component_diagnostics(current))
            columns = ["date", "symbol", "eligible", "broad_sector", "benchmark_weight", "recent_text_events",
                       "label_end_date_5", "label_end_date_20", *COMPONENTS, *TARGETS]
            trace = current[columns].copy()
            trace["test_year"], trace["market_regime"], trace["baseline_weight"] = year, regime, base_weight
            trace["global_validation_gate"] = models.global_gate
            trace["technology_validation_gate"] = models.technology_gate
            probability, magnitude = models.baseline_model.predict_components(current)
            trace["baseline_probability"], trace["baseline_magnitude"] = probability, magnitude
            traces.append(trace)
        trace_path = output_dir / f"scores_{year}.csv"
        pd.concat(traces, ignore_index=True).to_csv(trace_path, index=False, mode="x")
        annual_files[trace_path.name] = digest(trace_path)
        gate_path = output_dir / f"diagnostics_{year}.json"
        write_new(gate_path, json_safe({"year": year, "global_validation_gate": models.global_gate,
                  "technology_validation_gate": models.technology_gate,
                  "validation_diagnostics": models.validation_diagnostics,
                  "training_events": models.training_events, "raw_event_years": models.raw_event_years,
                  "payoff_lower_bound": models.payoff_lower_bound, "incremental_lower_bound": models.incremental_lower_bound,
                  "technology_lower_bound": models.technology_lower_bound,
                  "technology_incremental_lower_bound": models.technology_incremental_lower_bound,
                  "legacy_reproduction": year_reproduction, "score_sha256": annual_files[trace_path.name],
                  "partial_result_only": True, "execution_authorized": False}))
        annual_files[gate_path.name] = digest(gate_path)
        reproduction.extend(year_reproduction)
        callback("year_complete", test_year=int(year), completed_years=sum(y <= year for y in settings.test_years), total_years=len(settings.test_years))
        for old_year in list(cache):
            if old_year < year - settings.validation_years + 1:
                del cache[old_year]
    return pd.concat(metric_parts, ignore_index=True), reproduction, annual_files


def run():
    lock = verify()
    write_new(DIRECTORY / "run.started.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                                               "lock_sha256": lock["lock_sha256"]})
    try:
        settings = V20R2Settings()  # exactly the parent predictors; no diagnostic hyperparameters
        with pd.option_context("mode.copy_on_write", True):
            progress("loading_dataset", purpose="component_diagnosis_only")
            dataset = load_v16_dataset()
            progress("building_corpus", rows=len(dataset), symbols=int(dataset.symbol.nunique()))
            corpus = EnsembleTextCorpus.build(load_event_documents(), settings)
            reference = pd.read_csv(PARENT / "equity.csv", parse_dates=["date"])
            diagnostics, reproduction, files = run_diagnosis(dataset, corpus, reference, settings)
            summary = summary_table(diagnostics)
        progress("verifying_outputs")
        verify()
        for name, frame in (("component_metrics.csv", diagnostics), ("component_summary.csv", summary)):
            frame.to_csv(DIRECTORY / name, index=False, mode="x")
            files[name] = digest(DIRECTORY / name)
        report = {"status": "retrospective_component_diagnosis_complete", "lock_sha256": lock["lock_sha256"],
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "parent_ic_reproduction_passed": True,
                  "reproduced_dates": len(reproduction), "output_sha256": files, "frozen_inputs_intact": True,
                  "purpose": "component_diagnosis_only", "decision": "keep_v6_review_diagnostics_before_new_hypothesis",
                  "replacement_approved": False, "execution_authorized": False, "promotion_gates_evaluated": False,
                  "summary": json_safe(summary.astype(object).where(summary.notna(), None).to_dict("records")),
                  "limitations": ["Known retrospective window; diagnostics do not select a replacement or prove causal attribution.",
                                  "No parameter changes, sign reversals, portfolio optimization or new returns backtest.",
                                  "Target-specific pairwise masks can differ from the legacy label5-filtered IC20.",
                                  "Validation gates are observed only; parent research modes forced both gates on.",
                                  "Existing lowvol_top20 future shadow does not validate this candidate."]}
        write_new(DIRECTORY / "report.json", report)
        progress("complete", purpose="component_diagnosis_only", decision="keep_v6", execution_authorized=False)
        return {key: report[key] for key in ("status", "reproduced_dates", "parent_ic_reproduction_passed", "execution_authorized")}
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc(), execution_authorized=False)
        raise
