"""Read-only, fail-closed integration of frozen experiments and live snapshots."""
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


def read_json(path):
    path = Path(path)
    try:
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"read_error": "JSON must be an object"}
    except (ValueError, OSError) as error:
        return {"read_error": str(error)}


def snapshot_status(snapshot, today=None):
    today = today or datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    prediction = snapshot.get("latest_prediction_date")
    timing = snapshot.get("market_timing", {}).get("timing_date")
    reasons = []
    if not prediction or not timing:
        reasons.append("缺少选股或择时日期")
    elif prediction != timing:
        reasons.append("选股与择时日期不一致，禁止合成当日建议")
    if prediction and prediction < today:
        reasons.append("历史快照；未核验为当前交易日的完整预测")
    if prediction and prediction > today or timing and timing > today:
        reasons.append("快照包含未来日期")
    reasons.append("V17旧回测存在未来数据泄漏，94%胜率不可用于决策")
    return {
        "prediction_date": prediction, "timing_date": timing,
        "prediction_count": snapshot.get("prediction_count", 0),
        "candidate_count": snapshot.get("candidate_count", 0),
        "status": "historical_research_only", "reasons": reasons,
        "recommendation_enabled": False, "execution_authorized": False,
    }


def inspect_process(pid):
    """Read process identity without signals, termination, or optional packages."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return {"state": "unknown", "reason": "invalid PID"}
    if os.name != "nt":
        return {"state": "unknown", "reason": "process inspection currently supports Windows only"}
    script = (
        f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' "
        "-ErrorAction Stop | Select-Object ProcessId,CommandLine,"
        "@{Name='created_at';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}} "
        "| ConvertTo-Json -Compress"
    )
    try:
        output = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, check=True, creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout.strip()
        if not output:
            return {"state": "exited"}
        process = json.loads(output)
        return {"state": "alive", "pid": process["ProcessId"],
                "command_line": process.get("CommandLine", ""), "created_at": process.get("created_at")}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        return {"state": "unknown", "reason": str(error)}


def runtime_status(runtime, started, report, package, process_reader=None):
    if any(value.get("read_error") for value in (runtime, started, report)):
        return "invalid_status", {"state": "unknown", "reason": "状态文件无法读取"}
    stage = runtime.get("stage")
    if stage == "failed":
        return "failed", {"state": "not_checked"}
    if report:
        if report.get("lock_sha256") != started.get("lock_sha256") or not report.get("lock_sha256"):
            return "invalid_status", {"state": "unknown", "reason": "报告与运行锁不一致"}
        return "report_recorded", {"state": "not_checked"}
    if stage == "complete":
        return "incomplete_report", {"state": "unknown", "reason": "完成状态缺少报告"}
    if not runtime and not started:
        return "not_started", {"state": "not_checked"}
    pid = runtime.get("pid", started.get("pid"))
    if started.get("pid") != pid:
        return "invalid_status", {"state": "unknown", "reason": "运行记录PID不一致"}
    process = (process_reader or inspect_process)(pid)
    if process.get("state") == "exited":
        return "interrupted", process
    if process.get("state") != "alive":
        return "process_unverified", process
    try:
        created = datetime.fromisoformat(process["created_at"].replace("Z", "+00:00"))
        recorded = datetime.fromisoformat(started["started_at_utc"].replace("Z", "+00:00"))
        age = (recorded - created).total_seconds()
        command = process.get("command_line") or ""
        exact_module = re.search(r"(?:^|\s)-m\s+" + re.escape(package + ".cli") + r"\s+run(?:\s|$)", command)
        if process.get("pid") != pid or not exact_module or not 0 <= age <= 300:
            return "process_identity_mismatch", {"state": "mismatch", "reason": "PID或启动时间/命令行与本轮研究不符"}
    except (KeyError, ValueError, TypeError):
        return "process_unverified", {"state": "unknown", "reason": "缺少可核验的进程身份"}
    # Do not expose command lines in the browser payload.
    return stage or "starting", {"state": "alive", "pid": pid, "identity_verified": True}


def build_status(root=".", today=None, process_reader=None):
    root = Path(root)
    registry = read_json(root / "artifacts/active_research.json")
    package = registry.get("package", "research_v20")
    valid_registry = not registry.get("read_error") and isinstance(package, str) and re.fullmatch(r"research_v[0-9]+(?:r[0-9]+)?", package)
    if not valid_registry:
        package = "research_v20"
    folder = root / "artifacts" / package
    report = read_json(folder / "report.json")
    runtime = read_json(folder / "runtime_status.json")
    started = read_json(folder / "run.started.json")
    stage, process = runtime_status(runtime, started, report, package, process_reader)
    if not valid_registry:
        stage = "invalid_registry"
    elif registry and not (folder / "plan.lock.json").is_file():
        stage = "candidate_not_frozen"
    return {
        "active_model": "V6", "candidate_model": package.removeprefix("research_").replace("v", "V", 1),
        "candidate_stage": stage, "candidate_process": process,
        "candidate_runtime": runtime, "candidate_report": report,
        "historical_snapshot": snapshot_status(read_json(root / "artifacts/research_v16/live/latest.json"), today),
        "replacement_approved": False, "execution_authorized": False,
        "warnings": ["V17回测结果已隔离，不作为模型升级依据", "研究状态展示不是完整性或性能验收，正式模型仍为V6"],
    }


if __name__ == "__main__":
    print(json.dumps(build_status(), ensure_ascii=False, indent=2))
