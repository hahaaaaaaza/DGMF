param(
    [string]$Python = "C:\Users\Lenovo\miniconda3\envs\chemprop-5090d\python.exe",
    [string]$DataRoot = "D:\code\duomotai\chem5090\chemprop\data\newdata\results_selected_12_backbone_baselines_consistent\consistent_data",
    [string]$MolformerModel = "D:\code\duomotai\chem5090\hf_models\MoLFormer-XL-both-10pct",
    [string]$MolformerCache = "D:\code\duomotai\chem5090\chemprop\data\newdata\molformer_1d_cache",
    [string]$GeometryCache = "D:\code\duomotai\chem5090\chemprop\data\newdata\gotennet_3d_cache",
    [int]$Epochs = 80,
    [int]$Patience = 15,
    [int]$MaxAttempts = 4
)

$ErrorActionPreference = "Continue"
$ReleaseRoot = Split-Path -Parent $PSScriptRoot
$OutputRoot = Join-Path $ReleaseRoot "results\mechanism_matched_shared_gate_20260809"
$VariantRoot = Join-Path $OutputRoot "matched_shared_gate"
$LogRoot = Join-Path $ReleaseRoot "logs"
$LogPath = Join-Path $LogRoot "matched_shared_gate_20260809.combined.log"
$StatusPath = Join-Path $OutputRoot "worker_status.txt"
$FullSummary = Join-Path $ReleaseRoot "results\canonical\full\summary.csv"

Set-Location -LiteralPath $ReleaseRoot
New-Item -ItemType Directory -Path $OutputRoot, $LogRoot -Force | Out-Null
$env:PYTHONPATH = $ReleaseRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONHASHSEED = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:NVIDIA_TF32_OVERRIDE = "0"

& $Python (Join-Path $PSScriptRoot "audit_matched_shared_gate.py") `
    --output (Join-Path $VariantRoot "parameter_match_audit.csv") 2>&1 |
    Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    Set-Content -LiteralPath $StatusPath -Value "FAILED parameter_audit exit_code=$LASTEXITCODE time=$(Get-Date -Format o)" -Encoding ascii
    exit $LASTEXITCODE
}

for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
    Set-Content -LiteralPath $StatusPath -Value "RUNNING attempt=$Attempt pid=$PID started=$(Get-Date -Format o)" -Encoding ascii
    & $Python -u (Join-Path $PSScriptRoot "run_dgmf.py") `
        --variant matched_shared_gate `
        --tasks all `
        --seeds 1 2 3 4 5 `
        --epochs $Epochs `
        --patience $Patience `
        --output-root $OutputRoot `
        --data-root $DataRoot `
        --molformer-model $MolformerModel `
        --molformer-cache-dir $MolformerCache `
        --geometry-cache-dir $GeometryCache `
        --num-workers 0 `
        --accelerator gpu `
        --devices 1 2>&1 | Tee-Object -FilePath $LogPath -Append
    $RunExitCode = $LASTEXITCODE
    if ($RunExitCode -eq 0) {
        break
    }
    Set-Content -LiteralPath $StatusPath -Value "RETRY_PENDING attempt=$Attempt exit_code=$RunExitCode time=$(Get-Date -Format o)" -Encoding ascii
    Start-Sleep -Seconds 60
}

if ($RunExitCode -ne 0) {
    Set-Content -LiteralPath $StatusPath -Value "FAILED attempts=$MaxAttempts exit_code=$RunExitCode time=$(Get-Date -Format o)" -Encoding ascii
    exit $RunExitCode
}

& $Python (Join-Path $PSScriptRoot "verify_matched_shared_gate_results.py") `
    --variant-root $VariantRoot `
    --full-summary $FullSummary 2>&1 | Tee-Object -FilePath $LogPath -Append
$VerifyExitCode = $LASTEXITCODE
if ($VerifyExitCode -ne 0) {
    Set-Content -LiteralPath $StatusPath -Value "FAILED verification exit_code=$VerifyExitCode time=$(Get-Date -Format o)" -Encoding ascii
    exit $VerifyExitCode
}

Set-Content -LiteralPath $StatusPath -Value "COMPLETE runs=60 endpoints=12 finished=$(Get-Date -Format o)" -Encoding ascii
exit 0
