param(
    [string]$ReleaseRoot = "D:\code\duomotai\chem5090\DGMF_repro_20260808"
)

$ErrorActionPreference = "Stop"
$MainRoot = Join-Path $ReleaseRoot "results\canonical"
$MainWorker = Join-Path $ReleaseRoot "scripts\run_canonical_5090.ps1"
$QueueWorker = Join-Path $ReleaseRoot "scripts\run_canonical_baselines_5090.ps1"

function Test-WorkerRunning {
    param([string]$ScriptName)
    return $null -ne (Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "python|powershell" -and $_.CommandLine -like "*$ScriptName*"
    } | Select-Object -First 1)
}

$PredictionCount = (Get-ChildItem $MainRoot -Recurse -Filter test_predictions.csv -ErrorAction SilentlyContinue).Count
$MainStatusPath = Join-Path $MainRoot "worker_status.txt"
$MainStatus = (Get-Content $MainStatusPath -ErrorAction SilentlyContinue) -join " "

if ($PredictionCount -lt 300 -and -not (Test-WorkerRunning "run_canonical_5090.ps1")) {
    $Command = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $MainWorker + '" -Suite main -Epochs 80 -Patience 15'
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $Command} | Out-Null
}

if (-not (Test-WorkerRunning "run_canonical_baselines_5090.ps1")) {
    if ($PredictionCount -lt 300 -or $MainStatus -match "^FINISHED exit_code=0") {
        $Command = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $QueueWorker + '"'
        Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $Command} | Out-Null
    }
}
