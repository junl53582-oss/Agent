$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv311\Scripts\python.exe"
$Artifacts = Join-Path $ProjectRoot "artifacts\research_v20r1"
if (!(Test-Path -LiteralPath (Join-Path $Artifacts "plan.lock.json"))) {
    throw "Freeze V20r1 after tests before starting."
}
foreach ($Name in @("run.started.json", "report.json", "run_stdout.log", "run_stderr.log")) {
    if (Test-Path -LiteralPath (Join-Path $Artifacts $Name)) {
        throw "V20r1 already has a run record. Inspect it; do not overwrite or restart."
    }
}
$Process = Start-Process -FilePath $Python -ArgumentList @("-B", "-u", "-m", "research_v20r1.cli", "run") -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Artifacts "run_stdout.log") -RedirectStandardError (Join-Path $Artifacts "run_stderr.log") -PassThru
Write-Output "V20r1 started with launcher PID $($Process.Id). See artifacts/research_v20r1/runtime_status.json for the worker PID."
