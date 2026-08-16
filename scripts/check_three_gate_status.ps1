$Root = "D:\code\duomotai\chem5090\DGMF_repro_20260808"
$VariantRoot = Join-Path $Root "results\mechanism_three_gate_20260809\direction_id_gate"
$Log = Join-Path $Root "logs\direction_id_gate_20260809.combined.log"
$Predictions = @(Get-ChildItem $VariantRoot -Recurse -Filter test_predictions.csv -ErrorAction SilentlyContinue)
$SeedPath = Join-Path $VariantRoot "seed_metrics.csv"
$SummaryPath = Join-Path $VariantRoot "summary.csv"
$SeedRows = if (Test-Path $SeedPath) { @(Import-Csv $SeedPath).Count } else { 0 }
$SummaryRows = if (Test-Path $SummaryPath) { @(Import-Csv $SummaryPath).Count } else { 0 }
$Errors = if (Test-Path $Log) {
    @(Select-String $Log -Pattern "Traceback|CUDA out of memory|RuntimeError:" -ErrorAction SilentlyContinue).Count
} else { 0 }
$LastLogWrite = if (Test-Path $Log) { (Get-Item $Log).LastWriteTime.ToString("o") } else { "missing" }
$AuditPath = Join-Path $Root "results\mechanism_three_gate_20260809\parameter_match_audit.csv"
$AuditExact = if (Test-Path $AuditPath) {
    $Rows = @(Import-Csv $AuditPath)
    ($Rows.Count -eq 6 -and @($Rows | Where-Object { $_.exact_match -ne "True" }).Count -eq 0)
} else { $false }
[pscustomobject]@{
    prediction_files = $Predictions.Count
    seed_rows = $SeedRows
    summary_rows = $SummaryRows
    error_matches = $Errors
    last_log_write = $LastLogWrite
    parameter_audit_exact = $AuditExact
} | ConvertTo-Json -Compress
if (Test-Path $Log) { Get-Content $Log -Tail 12 }
