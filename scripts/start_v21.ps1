$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResearchPython = Join-Path $ProjectRoot ".venv311\Scripts\python.exe"
$ResearchArtifacts = Join-Path $ProjectRoot "artifacts\research_v21"
if (!(Test-Path -LiteralPath (Join-Path $ResearchArtifacts "plan.lock.json"))) {
    throw "Freeze V21 diagnosis after tests before starting."
}
foreach ($RecordName in @("run.started.json", "report.json", "run_stdout.log", "run_stderr.log")) {
    if (Test-Path -LiteralPath (Join-Path $ResearchArtifacts $RecordName)) {
        throw "V21 already has a run record; preserve it."
    }
}
$RunningResearch = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '-m research_v\d+(r\d+)?\.cli run' }
if ($RunningResearch) { throw "Another research run is active; inspect before proceeding." }
$ResearchProcess = Start-Process -FilePath $ResearchPython -ArgumentList @("-B", "-u", "-m", "research_v21.cli", "run") -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $ResearchArtifacts "run_stdout.log") -RedirectStandardError (Join-Path $ResearchArtifacts "run_stderr.log") -PassThru
Write-Output "V21 diagnostic launcher PID $($ResearchProcess.Id); actual worker is in runtime_status.json."
