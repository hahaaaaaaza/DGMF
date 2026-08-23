$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$ReferenceRoot = Join-Path $Root "reference_results\full"
$ManifestPath = Join-Path $ReferenceRoot "run_manifest.json"
$Failures = [System.Collections.Generic.List[string]]::new()

if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Missing reference run manifest: $ManifestPath"
}
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json

foreach ($Property in $Manifest.data_hashes.PSObject.Properties) {
    $Path = Join-Path (Join-Path $Root "data\fixed_inputs") $Property.Name
    if (-not (Test-Path -LiteralPath $Path)) {
        $Failures.Add("Missing data file: $($Property.Name)")
        continue
    }
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne ([string]$Property.Value).ToLowerInvariant()) {
        $Failures.Add("Data hash mismatch: $($Property.Name)")
    }
}

$ConfigFiles = @{
    "endpoints.json" = Join-Path $Root "configs\endpoints.json"
    "best_hyperparameters.csv" = Join-Path $Root "configs\best_hyperparameters.csv"
}
foreach ($Name in $ConfigFiles.Keys) {
    $Actual = (Get-FileHash -LiteralPath $ConfigFiles[$Name] -Algorithm SHA256).Hash.ToLowerInvariant()
    $Expected = ([string]$Manifest.configuration_hashes.$Name).ToLowerInvariant()
    if ($Actual -ne $Expected) {
        $Failures.Add("Configuration hash mismatch: $Name")
    }
}

$Builder = [System.Text.StringBuilder]::new()
foreach ($Directory in @("dgmf", "chemprop", "scripts")) {
    Get-ChildItem -Recurse -File -Filter "*.py" -LiteralPath (Join-Path $Root $Directory) |
        Sort-Object FullName |
        ForEach-Object {
            $Relative = $_.FullName.Substring($Root.Length + 1).Replace("\", "/")
            [void]$Builder.Append($Relative)
            [void]$Builder.Append((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())
        }
}
$Sha = [System.Security.Cryptography.SHA256]::Create()
$Bytes = [System.Text.Encoding]::UTF8.GetBytes($Builder.ToString())
$SourceHash = ([BitConverter]::ToString($Sha.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
if ($SourceHash -ne ([string]$Manifest.configuration_hashes.source_tree).ToLowerInvariant()) {
    $Failures.Add("Source-tree hash mismatch")
}

$Counts = [ordered]@{
    FixedInputFiles = (Get-ChildItem -Recurse -File -Filter "seed_*.csv" -LiteralPath (Join-Path $Root "data\fixed_inputs")).Count
    Commands = (Get-ChildItem -Recurse -File -Filter "command.json" -LiteralPath $ReferenceRoot).Count
    Configs = (Get-ChildItem -Recurse -File -Filter "config.toml" -LiteralPath $ReferenceRoot).Count
    Predictions = (Get-ChildItem -Recurse -File -Filter "test_predictions.csv" -LiteralPath $ReferenceRoot).Count
    HpoStudies = (Get-ChildItem -Recurse -File -Filter "study.db" -LiteralPath (Join-Path $Root "hpo_provenance")).Count
}
foreach ($Name in @("FixedInputFiles", "Commands", "Configs", "Predictions")) {
    if ($Counts[$Name] -ne 60) {
        $Failures.Add("Expected 60 $Name, found $($Counts[$Name])")
    }
}
if ($Counts.HpoStudies -ne 12) {
    $Failures.Add("Expected 12 HpoStudies, found $($Counts.HpoStudies)")
}

$PackageManifestPath = Join-Path $Root "PACKAGE_MANIFEST_SHA256.csv"
if (Test-Path -LiteralPath $PackageManifestPath) {
    foreach ($Row in Import-Csv -LiteralPath $PackageManifestPath) {
        $Path = Join-Path $Root $Row.path
        if (-not (Test-Path -LiteralPath $Path)) {
            $Failures.Add("Missing packaged file: $($Row.path)")
            continue
        }
        $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Row.sha256.ToLowerInvariant()) {
            $Failures.Add("Packaged file hash mismatch: $($Row.path)")
        }
    }
}

[pscustomobject]$Counts | Format-List
Write-Output "SourceTreeHash: $SourceHash"
Write-Output "ExpectedSourceTreeHash: $($Manifest.configuration_hashes.source_tree)"

if ($Failures.Count -gt 0) {
    $Failures | ForEach-Object { Write-Error $_ }
    exit 1
}
Write-Output "Package verification passed."

