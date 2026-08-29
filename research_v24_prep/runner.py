import json
import os
from datetime import datetime, timezone

from research_v20.freeze import digest, write_new
from .contract import build_document_features, training_gate
from .freeze import DIRECTORY, verify


def run():
    lock = verify()
    write_new(DIRECTORY / "run.started.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                                                "lock_sha256": lock["lock_sha256"]})
    facts = json.loads(open("artifacts/announcement_body_v3r1/facts.json", encoding="utf-8").read())
    observation = json.loads(open("artifacts/announcement_body_v5r2/observations/20260829T070833554473Z.report.json", encoding="utf-8").read())
    features = build_document_features(facts)
    prospective = int(observation["latest_observation"]["prospective_new_eligible"])
    gate = training_gate(prospective_events=prospective, approved_bodies=0, mature_5d=0, mature_20d=0)
    write_new(DIRECTORY / "feature_schema_preview.json", {"documents": features})
    report = {"status": "v24_preparation_contract_complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "lock_sha256": lock["lock_sha256"], "frozen_inputs_intact": True,
              "gold_documents": len(features), "gold_facts": sum(item["fact_count"] for item in features),
              "prospective_events": prospective, "training_gate": gate,
              "output_sha256": {"feature_schema_preview.json": digest(DIRECTORY / "feature_schema_preview.json")},
              "model_training_ready": False, "replacement_approved": False, "execution_authorized": False,
              "next_step": "append post-freeze bodies, then mature fixed 5d/20d labels before any supervised fit"}
    write_new(DIRECTORY / "report.json", report)
    return report

