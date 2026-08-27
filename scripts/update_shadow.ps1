param(
    [string]$EndDate = (Get-Date -Format "yyyy-MM-dd")
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Push-Location $ProjectRoot
try {
    $ObservationDate = [DateTime]::ParseExact($EndDate, "yyyy-MM-dd", $null)
    if ($ObservationDate.DayOfWeek -eq [DayOfWeek]::Monday) {
        & (Join-Path $PSScriptRoot "update_shadow_exposure.ps1") -EndDate $EndDate
        if ($LASTEXITCODE -ne 0) {
            throw "周一影子暴露刷新失败，停止生成信号"
        }
        & $Python -m research_v4.cli names-fetch
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "股票名称缓存刷新失败，继续使用现有缓存"
        }
    }
    & $Python -m stockpilot.cli shadow-update `
        --end $EndDate `
        --provider tencent `
        --workers 4
    if ($LASTEXITCODE -ne 0) {
        throw "影子观察更新失败，退出码 $LASTEXITCODE"
    }
    & $Python -m stockpilot.cli shadow-evaluate
    if ($LASTEXITCODE -ne 0) {
        throw "影子信号结算失败，退出码 $LASTEXITCODE"
    }
    & $Python -m stockpilot.cli future-adjudicate
    if ($LASTEXITCODE -ne 0) {
        throw "影子协议裁决状态更新失败，退出码 $LASTEXITCODE"
    }
    & $Python -m research_v6.cli predict
    if ($LASTEXITCODE -ne 0) {
        throw "V6最新研究预测生成失败，退出码 $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
