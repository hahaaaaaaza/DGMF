$ErrorActionPreference = "Continue"
$ReleaseRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\Lenovo\miniconda3\envs\chemprop-5090d\python.exe"
$OutputRoot = Join-Path $ReleaseRoot "results\mechanism_target_conditioning_factorial_20260814"
$StatusPath = Join-Path $OutputRoot "worker_status.txt"
$QueueLog = Join-Path $ReleaseRoot "logs\matched_shared_target_agnostic_20260814.queue.log"
$DataRoot = "D:\code\duomotai\chem5090\chemprop\data\newdata\results_selected_12_backbone_baselines_consistent\consistent_data"
$MolformerModel = "D:\code\duomotai\chem5090\hf_models\MoLFormer-XL-both-10pct"
$MolformerCache = "D:\code\duomotai\chem5090\chemprop\data\newdata\molformer_1d_cache"
$GeometryCache = "D:\code\duomotai\chem5090\chemprop\data\newdata\gotennet_3d_cache"

New-Item -ItemType Directory -Path $OutputRoot,(Split-Path -Parent $QueueLog) -Force | Out-Null
Set-Content -LiteralPath $StatusPath -Value "QUEUED waiting_for_active_training queued=$(Get-Date -Format o)" -Encoding ascii
Add-Content -LiteralPath $QueueLog -Value "[$(Get-Date -Format o)] queue started pid=$PID" -Encoding utf8

while ($true) {
    $activeTraining = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        $_.Name -eq "python.exe" -and
        $_.CommandLine -match "dgmf train|run_dgmf\.py|run_paper_suite\.py|run_selected_12_backbone_baselines\.py|run_framework_comparison|run_tdc12_sggrl\.py"
    }
    if (-not $activeTraining) {
        break
    }
    Set-Content -LiteralPath $StatusPath -Value "QUEUED waiting_for_active_training active_pids=$($activeTraining.ProcessId -join ',') checked=$(Get-Date -Format o)" -Encoding ascii
    Start-Sleep -Seconds 60
}

Set-Location -LiteralPath $ReleaseRoot
$env:PYTHONPATH = $ReleaseRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONHASHSEED = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:NVIDIA_TF32_OVERRIDE = "0"

Set-Content -LiteralPath $StatusPath -Value "SMOKE_RUNNING started=$(Get-Date -Format o)" -Encoding ascii
& $Python -u (Join-Path $PSScriptRoot "run_dgmf.py") `
    --variant matched_shared_target_agnostic `
    --tasks hia_hou `
    --seeds 1 `
    --epochs 2 `
    --patience 1 `
    --output-root (Join-Path $ReleaseRoot "results\matched_shared_target_agnostic_smoke") `
    --data-root $DataRoot `
    --molformer-model $MolformerModel `
    --molformer-cache-dir $MolformerCache `
    --geometry-cache-dir $GeometryCache `
    --num-workers 0 `
    --accelerator gpu `
    --devices 1 2>&1 | Tee-Object -FilePath $QueueLog -Append
if ($LASTEXITCODE -ne 0) {
    Set-Content -LiteralPath $StatusPath -Value "FAILED smoke exit_code=$LASTEXITCODE time=$(Get-Date -Format o)" -Encoding ascii
    exit $LASTEXITCODE
}

Add-Content -LiteralPath $QueueLog -Value "[$(Get-Date -Format o)] smoke passed; starting formal run" -Encoding utf8
& (Join-Path $PSScriptRoot "run_matched_shared_target_agnostic_5090.ps1") 2>&1 |
    Tee-Object -FilePath $QueueLog -Append
exit $LASTEXITCODE
