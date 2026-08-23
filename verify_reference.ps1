param(
    [string]$Python = "python",
    [string]$ObservedSummary = "",
    [double]$AbsoluteTolerance = 0.03
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ObservedSummary)) {
    $ObservedSummary = Join-Path $Root "results\reproduced\summary.csv"
}
$Reference = Join-Path $Root "reference_results\full\summary.csv"

if (-not (Test-Path -LiteralPath $ObservedSummary)) {
    throw "Observed summary not found: $ObservedSummary"
}
if (-not (Test-Path -LiteralPath $Reference)) {
    throw "Reference summary not found: $Reference"
}

& $Python (Join-Path $Root "scripts\check_reproduction.py") `
    --observed $ObservedSummary `
    --reference $Reference `
    --absolute-tolerance $AbsoluteTolerance

if ($LASTEXITCODE -ne 0) {
    throw "One or more endpoint means differ from the reference by more than $AbsoluteTolerance."
}

