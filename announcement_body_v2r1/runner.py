import json
import os
import traceback
from datetime import datetime, timezone

from announcement_body.core import sha_file, write_json_new
from announcement_body_v2.binder import evaluate_gold
from announcement_body_v2.runner import PARENT_DATA, PARENT_SELECTION
from .binder import extract_document
from .freeze import DIRECTORY, ROOT, verify


def progress(stage, **values):
    record = {"stage": stage, "pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    temp = (DIRECTORY / "runtime_status.json").with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, DIRECTORY / "runtime_status.json")


def run():
    lock = verify()
    write_json_new(DIRECTORY / "run.started.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"]})
    try:
        selection = json.loads(PARENT_SELECTION.read_text(encoding="utf-8"))
        gold_record = json.loads((ROOT / "artifacts/announcement_body_v2/gold.json").read_text(encoding="utf-8"))
        gold_documents = set(gold_record["documents"])
        predicted, documents = [], []
        for record in selection["records"]:
            document_id = record["symbol"] + "_" + record["announcement_id"]
            parsed = json.loads((PARENT_DATA / document_id / "parsed.json").read_text(encoding="utf-8"))
            if document_id in gold_documents and parsed["body_extraction_passed"]:
                facts, status = extract_document(document_id, parsed, record["selection_category"]), "gold_evaluated"
                predicted.extend(facts)
            elif not parsed["body_extraction_passed"]:
                facts, status = [], "quarantined_scan"
            else:
                facts, status = [], "unreviewed_not_approved"
            documents.append({"document_id": document_id, "status": status, "approved_facts": len(facts),
                              "pdf_sha256": sha_file(PARENT_DATA / document_id / "body.pdf")})
        evaluation = evaluate_gold(predicted, gold_record["facts"])
        passed = evaluation["precision"] == 1 and evaluation["recall"] == 1
        write_json_new(DIRECTORY / "facts.json", {"facts": predicted, "gold_binding_passed": passed,
                                                   "historical_pit_verified": False, "model_training_ready": False, "execution_authorized": False})
        write_json_new(DIRECTORY / "evaluation.json", evaluation)
        report = {"status": "gold_binding_passed" if passed else "gold_binding_failed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "lock_sha256": lock["lock_sha256"], "documents": documents, "evaluation": evaluation,
                  "output_sha256": {name: sha_file(DIRECTORY / name) for name in ("facts.json", "evaluation.json")},
                  "frozen_inputs_intact": True, "historical_pit_verified": False, "model_training_ready": False,
                  "replacement_approved": False, "execution_authorized": False,
                  "limitations": ["Only four of twelve fixed documents are gold-reviewed.", "Seven text documents remain unreviewed and one scan remains quarantined.",
                                  "Historical PIT availability is unverified; facts cannot enter model training.", "No market, return or model data were read."]}
        write_json_new(DIRECTORY / "report.json", report)
        progress("complete", status=report["status"], matched=evaluation["matched_facts"], gold=evaluation["gold_facts"], model_training_ready=False)
        return report
    except BaseException as error:
        progress("failed", error=str(error), traceback=traceback.format_exc(), model_training_ready=False, execution_authorized=False)
        raise
