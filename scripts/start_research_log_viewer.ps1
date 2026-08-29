$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv311\Scripts\python.exe"
$MonitorLogs = Join-Path $ProjectRoot "artifacts\autopilot"
$Existing = $null
try { $Existing = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 2 } catch { }
if ($Existing) {
    if ($Existing.app -ne "stockpilot-readonly-live-logs-v1") { throw "Port 8765 belongs to another service." }
    Write-Output "Live log viewer already running: http://127.0.0.1:8765/"
    exit 0
}
New-Item -ItemType Directory -Path $MonitorLogs -Force | Out-Null
$Viewer = Start-Process -FilePath $Python -ArgumentList @("-B", "-u", "-m", "research_log_viewer", "--port", "8765") -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $MonitorLogs "log_viewer_stdout.log") -RedirectStandardError (Join-Path $MonitorLogs "log_viewer_stderr.log") -PassThru
Write-Output "Read-only log viewer started, launcher PID $($Viewer.Id): http://127.0.0.1:8765/"
