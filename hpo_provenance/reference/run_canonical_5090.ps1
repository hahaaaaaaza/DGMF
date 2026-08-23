param(
    [string]$Python = "C:\Users\Lenovo\miniconda3\envs\chemprop-5090d\python.exe",
    [string]$DataRoot = "D:\code\duomotai\chem5090\chemprop\data\newdata\results_selected_12_backbone_baselines_consistent\consistent_data",
    [string]$MolformerModel = "D:\code\duomotai\chem5090\hf_models\MoLFormer-XL-both-10pct",
    [string]$MolformerCache = "D:\code\duomotai\chem5090\chemprop\data\newdata\molformer_1d_cache",
    [string]$GeometryCache = "D:\code\duomotai\chem5090\chemprop\data\newdata\gotennet_3d_cache",
    [ValidateSet("main", "mechanism", "all")]
    [string]$Suite = "main",
    [int]$Epochs = 80,
    [int]$Patience = 15
)

$ErrorActionPreference = "Stop"
$ReleaseRoot = Split-Path -Parent $PSScriptRoot
$OutputRoot = Join-Path $ReleaseRoot "results\canonical"
$LogRoot = Join-Path $ReleaseRoot "logs"
$LogPath = Join-Path $LogRoot "canonical_main.combined.log"
$StatusPath = Join-Path $OutputRoot "worker_status.txt"

Set-Location -LiteralPath $ReleaseRoot
New-Item -ItemType Directory -Path $OutputRoot, $LogRoot -Force | Out-Null
Set-Content -LiteralPath $StatusPath -Value "RUNNING pid=$PID started=$(Get-Date -Format o)" -Encoding ascii

& $Python -u (Join-Path $PSScriptRoot "run_paper_suite.py") `
    --suite $Suite `
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

$ExitCode = $LASTEXITCODE
Set-Content -LiteralPath $StatusPath -Value "FINISHED exit_code=$ExitCode finished=$(Get-Date -Format o)" -Encoding ascii
exit $ExitCode
