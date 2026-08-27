param(
    [string]$EndDate = (Get-Date -Format "yyyy-MM-dd")
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CompactDate = $EndDate.Replace("-", "")
$Output = Join-Path $ProjectRoot "data\shadow\exposures\$EndDate.csv"

Push-Location $ProjectRoot
try {
    if (Test-Path -LiteralPath $Output) {
        Write-Output "影子暴露快照已存在，不覆盖：$Output"
        exit 0
    }
    & $Python -m stockpilot.cli exposure-fetch `
        --membership data/universes/000300/history.csv `
        --start $CompactDate `
        --end $CompactDate `
        --output $Output `
        --workers 1 `
        --active-only
    if ($LASTEXITCODE -ne 0) {
        throw "影子暴露更新失败，退出码 $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
