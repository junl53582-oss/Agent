import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from .core import AccessLimited, CATEGORIES, YEARS, fetch_document, select_pilot, sha_file, verify_cached, write_json_new


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/announcement_body_v1"
DATA = ROOT / "data/announcement_body_v1"
SOURCE = ROOT / "data/announcements_pit_v14.csv"


def progress(stage, **values):
    record = {"stage": stage, "at_utc": datetime.now(timezone.utc).isoformat(), **values}
    with (ARTIFACTS / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def selection():
    path = ARTIFACTS / "selection.json"
    if sha_file(path) != (ARTIFACTS / "selection.sha256").read_text().strip():
        raise ValueError("pilot selection changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["source_sha256"] != sha_file(SOURCE):
        raise ValueError("frozen announcement metadata changed")
    return result


def prepare():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if (ARTIFACTS / "selection.json").exists():
        raise ValueError("pilot already selected; do not overwrite")
    before = sha_file(SOURCE)
    records, count = select_pilot(SOURCE)
    if before != sha_file(SOURCE):
        raise ValueError("metadata changed while selecting")
    write_json_new(ARTIFACTS / "selection.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": before,
        "source_rows": count, "source_path": SOURCE.relative_to(ROOT).as_posix(),
        "rule": "minimum SHA256(symbol:id) per year/category; no market data or outcomes read",
        "years": YEARS, "categories": CATEGORIES, "records": records,
        "purpose": "12-document parsing/provenance pilot, not a training corpus", "model_training_ready": False})
    with (ARTIFACTS / "selection.sha256").open("x", encoding="utf-8") as stream:
        stream.write(sha_file(ARTIFACTS / "selection.json") + "\n")
    progress("selected", documents=len(records), source_rows=count)


def fetch(limit=None):
    records = selection()["records"]
    chosen = records if limit is None else records[:limit]
    for index, record in enumerate(chosen, start=1):
        progress("fetching", current=index, requested=len(chosen), symbol=record["symbol"],
                 announcement_id=record["announcement_id"], category=record["selection_category"])
        try:
            receipt = fetch_document(record, DATA)
            progress("document_complete", symbol=record["symbol"], announcement_id=record["announcement_id"],
                     body_extraction_passed=receipt["body_extraction_passed"], model_training_ready=False)
        except AccessLimited as error:
            progress("access_limited", error=str(error), automatic_retry=False)
            raise
        except Exception as error:
            progress("document_failed", symbol=record["symbol"], announcement_id=record["announcement_id"], error=str(error), automatic_retry=False)
    result = audit()
    progress("pilot_progress", **{key: result[key] for key in ("selected", "completed", "body_extraction_passed", "failed", "pending")})
    return result


def audit():
    records = selection()["records"]
    result = {"selected": len(records), "completed": 0, "body_extraction_passed": 0, "failed": 0, "pending": 0,
              "pages": 0, "numeric_mentions": 0, "tables": 0, "documents": [],
              "historical_pit_verified": False, "model_training_ready": False, "execution_authorized": False}
    for record in records:
        folder = DATA / (record["symbol"] + "_" + record["announcement_id"])
        status = {"symbol": record["symbol"], "announcement_id": record["announcement_id"],
                  "year": record["selection_year"], "category": record["selection_category"]}
        if (folder / "receipt.json").exists():
            receipt = verify_cached(folder)
            parsed = json.loads((folder / "parsed.json").read_text(encoding="utf-8"))
            result["completed"] += 1
            result["body_extraction_passed"] += int(receipt["body_extraction_passed"])
            result["pages"] += len(parsed["pages"])
            result["numeric_mentions"] += len(parsed["numeric_mentions"])
            result["tables"] += sum(len(page["tables"]) for page in parsed["pages"])
            status.update(status="extracted" if receipt["body_extraction_passed"] else "quarantined_quality", quality=parsed["quality"])
        elif (folder / "failure.json").exists():
            result["failed"] += 1
            status.update(status="failed", failure=json.loads((folder / "failure.json").read_text(encoding="utf-8")))
        else:
            result["pending"] += 1
            status["status"] = "pending"
        result["documents"].append(status)
    return result


def freeze():
    result = audit()
    if result["pending"]:
        raise ValueError("pilot still pending; do not freeze incomplete execution")
    qa = ARTIFACTS / "visual_qa.json"
    if not qa.exists():
        raise ValueError("representative PDF/table visual QA required before data freeze")
    qa_record = json.loads(qa.read_text(encoding="utf-8"))
    if qa_record.get("inspection_complete") is not True or len(qa_record.get("documents", [])) < 4:
        raise ValueError("visual QA record is incomplete")
    receipt = json.loads((ARTIFACTS / "test_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("passed") is not True or receipt.get("tests_passed", 0) < 10:
        raise ValueError("body-pipeline tests must pass before freeze")
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["scope"] = "data preparation pilot only; no alpha or performance test"
    result["software"] = {"python": platform.python_version(),
                          **{name: importlib.metadata.version(name) for name in ("pdfplumber", "pypdf", "pypdfium2")}}
    write_json_new(ARTIFACTS / "report.json", result)
    files = list((ROOT / "announcement_body").glob("*.py")) + [ROOT / "tests/test_announcement_body.py"]
    files += [path for path in ARTIFACTS.iterdir() if path.is_file()]
    files += [path for path in DATA.rglob("*") if path.is_file()]
    lock = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "data_pilot_not_model",
            "sha256": {path.relative_to(ROOT).as_posix(): sha_file(path) for path in sorted(files)},
            "source_path": SOURCE.relative_to(ROOT).as_posix(), "source_sha256": sha_file(SOURCE),
            "historical_pit_verified": False, "model_training_ready": False, "execution_authorized": False}
    write_json_new(ARTIFACTS / "data.lock.json", lock)
    with (ARTIFACTS / "data.lock.sha256").open("x", encoding="utf-8") as stream:
        stream.write(sha_file(ARTIFACTS / "data.lock.json") + "\n")
    return verify()


def verify():
    path = ARTIFACTS / "data.lock.json"
    if sha_file(path) != (ARTIFACTS / "data.lock.sha256").read_text().strip():
        raise ValueError("body data lock changed")
    lock = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in lock["sha256"].items():
        if sha_file(ROOT / name) != expected:
            raise ValueError(f"body pilot frozen file changed: {name}")
    if sha_file(ROOT / lock["source_path"]) != lock["source_sha256"]:
        raise ValueError("parent announcement metadata changed")
    return {"frozen_inputs_intact": True, "lock_sha256": sha_file(path), "model_training_ready": False, "execution_authorized": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fetch", "audit", "freeze", "verify"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if (ARTIFACTS / "data.lock.json").exists() and args.command not in {"audit", "verify"}:
        raise ValueError("body pilot frozen; preserve code/data and use a new independent revision")
    functions = {"prepare": prepare, "fetch": lambda: fetch(args.limit), "audit": audit, "freeze": freeze, "verify": verify}
    result = functions[args.command]()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
