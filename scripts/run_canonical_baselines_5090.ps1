param(
    [string]$Python = "C:\Users\Lenovo\miniconda3\envs\chemprop-5090d\python.exe",
    [string]$ProjectRoot = "D:\code\duomotai\chem5090",
    [string]$ReleaseRoot = "D:\code\duomotai\chem5090\DGMF_repro_20260808"
)

$ErrorActionPreference = "Stop"
$NewData = Join-Path $ProjectRoot "chemprop\data\newdata"
$OutputRoot = Join-Path $ReleaseRoot "results\canonical_baselines"
$DataRoot = Join-Path $OutputRoot "consistent_data"
$LogRoot = Join-Path $ReleaseRoot "logs"
$LogPath = Join-Path $LogRoot "canonical_baselines.combined.log"
$StatusPath = Join-Path $OutputRoot "worker_status.txt"
$MainStatusPath = Join-Path $ReleaseRoot "results\canonical\worker_status.txt"

New-Item -ItemType Directory -Path $OutputRoot, $LogRoot -Force | Out-Null
Set-Content -LiteralPath $StatusPath -Value "WAITING_FOR_MAIN pid=$PID started=$(Get-Date -Format o)" -Encoding ascii

while ($true) {
    $MainStatus = Get-Content -LiteralPath $MainStatusPath -ErrorAction SilentlyContinue
    if ($MainStatus -match "^FINISHED exit_code=0") {
        break
    }
    if ($MainStatus -match "^FINISHED exit_code=(?!0)") {
        Set-Content -LiteralPath $StatusPath -Value "BLOCKED main_status=$MainStatus" -Encoding ascii
        exit 1
    }
    Start-Sleep -Seconds 60
}

$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:NVIDIA_TF32_OVERRIDE = "0"
$env:PYTHONIOENCODING = "utf-8"
Set-Content -LiteralPath $StatusPath -Value "RUNNING pid=$PID started=$(Get-Date -Format o)" -Encoding ascii

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

try {
    Invoke-LoggedPython @(
        "-u", (Join-Path $ReleaseRoot "scripts\freeze_canonical_results.py"),
        "--output-root", (Join-Path $ReleaseRoot "results\canonical"),
        "--suite", "main"
    )

    $MechanismRoot = Join-Path $ReleaseRoot "results\canonical_mechanism"
    Invoke-LoggedPython @(
        "-u", (Join-Path $ReleaseRoot "scripts\run_paper_suite.py"),
        "--suite", "mechanism",
        "--tasks", "all",
        "--seeds", "1", "2", "3", "4", "5",
        "--epochs", "80",
        "--patience", "15",
        "--output-root", $MechanismRoot,
        "--data-root", (Join-Path $NewData "results_selected_12_backbone_baselines_consistent\consistent_data"),
        "--molformer-model", (Join-Path $ProjectRoot "hf_models\MoLFormer-XL-both-10pct"),
        "--molformer-cache-dir", (Join-Path $NewData "molformer_1d_cache"),
        "--geometry-cache-dir", (Join-Path $NewData "gotennet_3d_cache"),
        "--num-workers", "0",
        "--accelerator", "gpu",
        "--devices", "1"
    )
    Invoke-LoggedPython @(
        "-u", (Join-Path $ReleaseRoot "scripts\freeze_canonical_results.py"),
        "--output-root", $MechanismRoot,
        "--suite", "mechanism"
    )

    Invoke-LoggedPython @(
        "-u", (Join-Path $NewData "run_selected_12_backbone_baselines.py"),
        "--models", "attentivefp", "dmpnn", "molformer", "chemberta", "gotennet",
        "--output-root", $OutputRoot,
        "--source-data-root", (Join-Path $NewData "results_selected_12_backbone_baselines_consistent\consistent_data"),
        "--consistent-data-root", $DataRoot,
        "--seeds", "1", "2", "3", "4", "5",
        "--chemprop-epochs", "80", "--chemprop-patience", "15",
        "--pyg-epochs", "80", "--pyg-patience", "15",
        "--molformer-model", (Join-Path $ProjectRoot "hf_models\MoLFormer-XL-both-10pct"),
        "--chemberta-model", (Join-Path $ProjectRoot "hf_models\ChemBERTa-77M-MLM"),
        "--num-workers", "0", "--train-accelerator", "gpu", "--train-devices", "1"
    )

    $CommonUniMol = @(
        "--configs", "unimodal_3d_unimol",
        "--data-root", $DataRoot,
        "--seeds", "1", "2", "3", "4", "5",
        "--epochs", "80", "--patience", "15", "--batch-size", "16",
        "--dropout", "0.2", "--init-lr", "1e-5", "--max-lr", "1e-4", "--final-lr", "1e-5",
        "--embed-dim", "128", "--x-d-encoder-heads", "4", "--x-d-fp-groups", "64",
        "--unimol-cache-dir", (Join-Path $NewData "unimol_3d_cache"),
        "--num-workers", "0", "--train-accelerator", "gpu", "--train-devices", "1"
    )
    Invoke-LoggedPython (@(
        "-u", (Join-Path $NewData "run_framework_comparison_classification.py"),
        "--output-root", (Join-Path $OutputRoot "unimol_classification"),
        "--tasks", "bioavailability_ma", "hia_hou", "pgp_broccatelli", "bbb_martins",
        "cyp2d6_substrate_carbonmangels", "cyp2c9_substrate_carbonmangels", "cyp3a4_veith", "herg"
    ) + $CommonUniMol)
    Invoke-LoggedPython (@(
        "-u", (Join-Path $NewData "run_framework_comparison.py"),
        "--output-root", (Join-Path $OutputRoot "unimol_regression"),
        "--tasks", "vdss_lombardo", "clearance_hepatocyte_az", "clearance_microsome_az", "ld50_zhu"
    ) + $CommonUniMol)

    Invoke-LoggedPython @(
        "-u", (Join-Path $ProjectRoot "sggrl_tools\prepare_tdc12_for_sggrl.py"),
        "--repo-root", $ProjectRoot, "--source-root", $DataRoot, "--overwrite"
    )
    Invoke-LoggedPython @(
        "-u", (Join-Path $ProjectRoot "sggrl_tools\run_tdc12_sggrl.py"),
        "--repo-root", $ProjectRoot,
        "--output-root", (Join-Path $OutputRoot "sggrl"),
        "--seeds", "1", "2", "3", "4", "5", "--epochs", "100", "--batch-size", "32", "--gpu", "0"
    )

    Set-Content -LiteralPath $StatusPath -Value "FINISHED exit_code=0 finished=$(Get-Date -Format o)" -Encoding ascii
    exit 0
}
catch {
    $_ | Out-String | Tee-Object -FilePath $LogPath -Append
    Set-Content -LiteralPath $StatusPath -Value "FINISHED exit_code=1 finished=$(Get-Date -Format o)" -Encoding ascii
    exit 1
}
