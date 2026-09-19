# Reproducibility

## Fixed protocol

- 12 TDC endpoints: eight classification and four regression tasks.
- Five fixed Bemis-Murcko scaffold splits, using seeds 1-5.
- 20 Optuna trials per endpoint.
- HPO budget: 40 epochs with patience 10.
- Final training budget: 80 epochs with patience 15.
- Classification checkpoint selection: validation AUROC.
- VDss and clearance checkpoint selection: validation Spearman correlation.
- LD50 checkpoint selection: validation loss.
- Reported values: mean and sample standard deviation over five seeds.

The selected endpoint hyperparameters are stored in
`configs/best_hyperparameters.csv`. The current runner passes both the data seed
and PyTorch seed for every run.

## Reference tables

Use these files when checking the manuscript:

- `results/reference/full.csv`
- `results/reference/ablation.csv`
- `results/reference/controls.csv`

The Full table contains the final manuscript values. The ablation table contains
48 rows covering concat and the three branch-removal variants. Those rows were
recomputed from the archived per-seed test predictions and matched the reference
within floating-point precision. The controls file contains the shared-gate and
target-agnostic comparisons.

## Retraining and archived results

The Full runner is the deterministic training protocol for the released code.
The historical ablation archive also contains its original configurations,
checkpoints, and predictions. Its old configurations did not record a PyTorch
training seed, so a fresh ablation retraining can reproduce the architecture,
data, and training settings but is not a bitwise regeneration of the archived
weights. Exact checking of the paper table should use the archived predictions
or checkpoint replay supplied with the release package.

## Validation

```powershell
python scripts/validate_release.py
python -m pytest tests -q
```

For a completed run, compare its summary with a reference table:

```powershell
python scripts/verify_results.py `
  --observed results/runs/summary.csv `
  --reference results/reference/full.csv
```
