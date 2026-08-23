$ErrorActionPreference = "Stop"
$Root = "D:\code\duomotai\chem5090\DGMF_repro_20260808"
$Python = "C:\Users\Lenovo\miniconda3\envs\chemprop-5090d\python.exe"
Set-Location $Root
$env:PYTHONPATH = $Root
$env:PYTHONIOENCODING = "utf-8"
& $Python -u scripts\run_dgmf.py `
    --variant direction_id_gate `
    --tasks bioavailability_ma `
    --seeds 1 `
    --epochs 2 `
    --patience 1 `
    --output-root results\smoke_direction_id_gate_20260809 `
    --data-root D:\code\duomotai\chem5090\chemprop\data\newdata\results_selected_12_backbone_baselines_consistent\consistent_data `
    --molformer-model D:\code\duomotai\chem5090\hf_models\MoLFormer-XL-both-10pct `
    --molformer-cache-dir D:\code\duomotai\chem5090\chemprop\data\newdata\molformer_1d_cache `
    --geometry-cache-dir D:\code\duomotai\chem5090\chemprop\data\newdata\gotennet_3d_cache `
    --num-workers 0 `
    --accelerator gpu `
    --devices 1 `
    --force
exit $LASTEXITCODE
