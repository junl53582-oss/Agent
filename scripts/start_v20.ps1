$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv311\Scripts\python.exe"
$Artifacts = Join-Path $ProjectRoot "artifacts\research_v20"
if (!(Test-Path -LiteralPath (Join-Path $Artifacts "plan.lock.json"))) {
    throw "Freeze V20 after tests before starting."
}
foreach ($Name in @("run.started.json", "report.json", "run_stdout.log", "run_stderr.log")) {
    if (Test-Path -LiteralPath (Join-Path $Artifacts $Name)) {
        throw "V20 already has a run record. Inspect it; do not overwrite or restart."
    }
}
$Process = Start-Process -FilePath $Python -ArgumentList @("-B", "-u", "-m", "research_v20.cli", "run") -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Artifacts "run_stdout.log") -RedirectStandardError (Join-Path $Artifacts "run_stderr.log") -PassThru
Write-Output "V20 started with PID $($Process.Id). See artifacts/research_v20/runtime_status.json and run_stdout.log."
