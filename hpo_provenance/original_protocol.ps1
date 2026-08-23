$ErrorActionPreference = "Stop"

$PY = "C:\Users\Lenovo\miniconda3\envs\chemprop-5090d\python.exe"
$ROOT = "D:\code\duomotai\chem5090"
$RUNNER = Join-Path $ROOT "chemprop\data\newdata\run_two_config_5seed_hpo.py"
$MOLFORMER = Join-Path $ROOT "hf_models\MoLFormer-XL-both-10pct"
$CONFIG = "fusion_1d2d3d_embedding_xattn_molformer_gotennet"

$classificationTasks = @(
    "bioavailability_ma",
    "hia_hou",
    "pgp_broccatelli",
    "bbb_martins",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_veith",
    "herg"
)

$regressionTasks = @(
    "vdss_lombardo",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "ld50_zhu"
)

Set-Location $ROOT

foreach ($task in $classificationTasks) {
    & $PY -u $RUNNER `
        --task-type classification `
        --task $task `
        --configs $CONFIG `
        --n-trials 20 `
        --hpo-seeds 1 2 3 4 5 `
        --epochs 40 `
        --patience 10 `
        --final-epochs 80 `
        --final-patience 15 `
        --output-root hpo20_cls `
        --molformer-model $MOLFORMER `
        --molformer-device cpu `
        --molformer-pooling pooler `
        --geometry-num-conformers 8 `
        --train-accelerator gpu `
        --train-devices 1 `
        --num-workers 0
}

foreach ($task in $regressionTasks) {
    & $PY -u $RUNNER `
        --task-type regression `
        --task $task `
        --configs $CONFIG `
        --n-trials 20 `
        --hpo-seeds 1 2 3 4 5 `
        --epochs 40 `
        --patience 10 `
        --final-epochs 80 `
        --final-patience 15 `
        --output-root hpo20_reg `
        --molformer-model $MOLFORMER `
        --molformer-device cpu `
        --molformer-pooling pooler `
        --geometry-num-conformers 8 `
        --train-accelerator gpu `
        --train-devices 1 `
        --num-workers 0
}
