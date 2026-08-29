$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv311\Scripts\python.exe"
$Artifacts = Join-Path $ProjectRoot "artifacts\research_v20r2"
if (!(Test-Path -LiteralPath (Join-Path $Artifacts "plan.lock.json"))) {
    throw "Freeze V20r2 after tests before starting."
}
foreach ($Name in @("run.started.json", "report.json", "run_stdout.log", "run_stderr.log")) {
    if (Test-Path -LiteralPath (Join-Path $Artifacts $Name)) {
        throw "V20r2 already has a run record. Do not overwrite/restart."
    }
}
$Running = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '-m research_v\d+(r\d+)?\.cli run' }
if ($Running) { throw "Another research run is active; inspect it before proceeding." }
$Process = Start-Process -FilePath $Python -ArgumentList @("-B", "-u", "-m", "research_v20r2.cli", "run") -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Artifacts "run_stdout.log") -RedirectStandardError (Join-Path $Artifacts "run_stderr.log") -PassThru
Write-Output "V20r2 launcher PID $($Process.Id); see runtime_status.json for actual worker PID."
