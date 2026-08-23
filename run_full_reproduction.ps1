param(
    [string]$Python = "python",
    [string]$MolformerModel = "",
    [string]$MolformerCacheDir = "",
    [string]$GeometryCacheDir = "",
    [string]$OutputRoot = "",
    [ValidateSet("gpu", "cpu")]
    [string]$Accelerator = "gpu",
    [string]$Devices = "1"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$DataRoot = Join-Path $Root "data\fixed_inputs"

if ([string]::IsNullOrWhiteSpace($MolformerModel)) {
    $LocalModel = Join-Path $Root "assets\MoLFormer-XL-both-10pct"
    $MolformerModel = if (Test-Path -LiteralPath $LocalModel) {
        $LocalModel
    } else {
        "ibm-research/MoLFormer-XL-both-10pct"
    }
}
if ([string]::IsNullOrWhiteSpace($MolformerCacheDir)) {
    $MolformerCacheDir = Join-Path $Root "assets\molformer_1d_cache"
}
if ([string]::IsNullOrWhiteSpace($GeometryCacheDir)) {
    $GeometryCacheDir = Join-Path $Root "assets\gotennet_3d_cache"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $Root "results\reproduced"
}

$SplitFiles = Get-ChildItem -Recurse -File -Filter "seed_*.csv" -LiteralPath $DataRoot
if ($SplitFiles.Count -ne 60) {
    throw "Expected 60 fixed split files under $DataRoot, found $($SplitFiles.Count)."
}

New-Item -ItemType Directory -Path $MolformerCacheDir, $GeometryCacheDir, $OutputRoot -Force | Out-Null

$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:NVIDIA_TF32_OVERRIDE = "0"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Set-Location -LiteralPath $Root
& $Python -u (Join-Path $Root "scripts\run_dgmf.py") `
    --variant full `
    --tasks all `
    --seeds 1 2 3 4 5 `
    --data-root $DataRoot `
    --output-root $OutputRoot `
    --molformer-model $MolformerModel `
    --molformer-cache-dir $MolformerCacheDir `
    --geometry-cache-dir $GeometryCacheDir `
    --epochs 80 `
    --patience 15 `
    --num-workers 0 `
    --accelerator $Accelerator `
    --devices $Devices

if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE."
}

& (Join-Path $Root "verify_reference.ps1") -Python $Python -ObservedSummary (Join-Path $OutputRoot "summary.csv")
