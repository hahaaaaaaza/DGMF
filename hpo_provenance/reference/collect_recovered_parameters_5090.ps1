$ErrorActionPreference = "Stop"

$base = "D:\code\duomotai\chem5090\chemprop\data\newdata"
$out = "D:\code\duomotai\chem5090\paper_reproduction_archive_20260815"
$config = "fusion_1d2d3d_embedding_xattn_molformer_gotennet"

$tasks = @(
    [pscustomobject]@{ Endpoint = "Bioavailability"; Type = "classification"; Task = "bioavailability_ma"; Root = "hpo20_cls" },
    [pscustomobject]@{ Endpoint = "HIA"; Type = "classification"; Task = "hia_hou"; Root = "hpo20_cls" },
    [pscustomobject]@{ Endpoint = "P-gp"; Type = "classification"; Task = "pgp_broccatelli"; Root = "hpo20_cls" },
    [pscustomobject]@{ Endpoint = "BBB"; Type = "classification"; Task = "bbb_martins"; Root = "hpo20_cls" },
    [pscustomobject]@{ Endpoint = "CYP2C9 substrate"; Type = "classification"; Task = "cyp2c9_substrate_carbonmangels"; Root = "hpo20_cls" },
    [pscustomobject]@{ Endpoint = "CYP2D6 substrate"; Type = "classification"; Task = "cyp2d6_substrate_carbonmangels"; Root = "hpo20_cls" },
    [pscustomobject]@{ Endpoint = "CYP3A4 Veith"; Type = "classification"; Task = "cyp3a4_veith"; Root = "hpo20_cls" },
    [pscustomobject]@{ Endpoint = "hERG"; Type = "classification"; Task = "herg"; Root = "hpo20_cls" },
    [pscustomobject]@{ Endpoint = "VDss"; Type = "regression"; Task = "vdss_lombardo"; Root = "hpo20_reg" },
    [pscustomobject]@{ Endpoint = "Hepatocyte clearance"; Type = "regression"; Task = "clearance_hepatocyte_az"; Root = "hpo20_reg" },
    [pscustomobject]@{ Endpoint = "Microsome clearance"; Type = "regression"; Task = "clearance_microsome_az"; Root = "hpo20_reg" },
    [pscustomobject]@{ Endpoint = "LD50"; Type = "regression"; Task = "ld50_zhu"; Root = "hpo20_reg" }
)

New-Item -ItemType Directory -Force -Path $out | Out-Null
$provenance = @()
$parameters = @()

foreach ($entry in $tasks) {
    $hpo = Join-Path $base "$($entry.Root)\$($entry.Type)\$($entry.Task)\hpo\$config"
    $destination = Join-Path $out "endpoints\$($entry.Task)"
    New-Item -ItemType Directory -Force -Path $destination | Out-Null

    foreach ($name in @(
        "best_params.json",
        "study.db",
        "optuna_trials.csv",
        "best_trial_test_seed_metrics.csv",
        "best_trial_test_summary.csv"
    )) {
        Copy-Item (Join-Path $hpo $name) (Join-Path $destination $name) -Force
    }

    $best = Get-Content (Join-Path $hpo "best_params.json") -Raw | ConvertFrom-Json
    for ($seed = 1; $seed -le 5; $seed++) {
        $source = Join-Path $hpo ("trial_{0:D4}\seed_{1}\config.toml" -f [int]$best.best_trial, $seed)
        Copy-Item $source (Join-Path $destination "selected_trial_seed_${seed}_config.toml") -Force
    }

    $trials = Import-Csv (Join-Path $hpo "optuna_trials.csv")
    $starts = $trials | ForEach-Object { [datetime]$_.datetime_start }
    $ends = $trials | ForEach-Object { [datetime]$_.datetime_complete }
    $provenance += [pscustomobject]@{
        endpoint = $entry.Endpoint
        task = $entry.Task
        task_type = $entry.Type
        trials = $trials.Count
        earliest_trial = ($starts | Measure-Object -Minimum).Minimum.ToString("yyyy-MM-dd HH:mm:ss")
        latest_trial = ($ends | Measure-Object -Maximum).Maximum.ToString("yyyy-MM-dd HH:mm:ss")
        best_trial = $best.best_trial
        best_value = $best.best_value
        best_params_sha256 = (Get-FileHash (Join-Path $hpo "best_params.json") -Algorithm SHA256).Hash
        study_db_sha256 = (Get-FileHash (Join-Path $hpo "study.db") -Algorithm SHA256).Hash
    }

    $params = $best.best_params
    $parameters += [pscustomobject]@{
        endpoint = $entry.Endpoint
        task = $entry.Task
        batch_size = $params.batch_size
        depth = $params.depth
        message_hidden_dim = $params.message_hidden_dim
        dropout = $params.dropout
        ffn_hidden_dim = $params.ffn_hidden_dim
        ffn_num_layers = $params.ffn_num_layers
        embed_dim = $params.x_d_embed_dim
        encoder_heads = $params.x_d_encoder_heads
        fp_groups = $params.x_d_fp_groups
        warmup_epochs = $params.warmup_epochs
        init_lr = $params.init_lr
        max_lr = $params.max_lr
        final_lr = $params.final_lr
        gotennet_cutoff = $params.gotennet_cutoff
        gotennet_pooling = $params.gotennet_pooling
        gotennet_lr_scale = $params.gotennet_lr_scale
        best_trial = $best.best_trial
        best_value = $best.best_value
    }
}

$provenance | Export-Csv (Join-Path $out "hpo_provenance.csv") -NoTypeInformation -Encoding utf8
$parameters | Export-Csv (Join-Path $out "recovered_best_hyperparameters.csv") -NoTypeInformation -Encoding utf8

$zip = "D:\code\duomotai\chem5090\paper_reproduction_archive_20260815_from5090.zip"
Compress-Archive -Path "$out\*" -DestinationPath $zip -Force
Get-Item $zip | Select-Object FullName, Length, LastWriteTime
