# STGXAttn manuscript parameter recovery archive

This archive contains the endpoint-specific hyperparameters selected by the
original 20-trial Optuna studies used for the 12 manuscript endpoints.

## What was recovered

- `best_hyperparameters.csv`: consolidated parameters for all 12 endpoints.
- `endpoints.json`: endpoint names, task types, metrics, and retained sample counts.
- `paper_primary_metrics_full_precision.csv`: the archived manuscript results.
- `hpo_provenance.csv`: original Optuna trial windows, selected trial numbers,
  objective values, and SHA-256 hashes of the recovered parameter files.
- `endpoints/<task>/best_params.json`: the selected parameter file written from
  the original Optuna study.
- `endpoints/<task>/study.db`: the original 20-trial Optuna SQLite database.
- `endpoints/<task>/optuna_trials.csv`: exported trial history.
- `endpoints/<task>/best_trial_test_seed_metrics.csv`: five-seed test metrics for
  the selected HPO trial.
- `endpoints/<task>/best_trial_test_summary.csv`: summary of those five runs.
- `original_protocol.ps1`: the original HPO/final-training command structure.

## Provenance conclusion

The 12 Optuna databases contain 20 completed trials each, dated from
2026-06-19 to 2026-06-23. The selected trial number and objective value in every
`best_params.json` agree with its original `study.db`. A later command on
2026-07-21 used `--force` with the same 20-trial studies. Because the runner
computes `remaining_trials = max(0, n_trials - completed_trials)`, no new HPO
trials were added; only final five-seed training was rerun. Therefore, the
recovered hyperparameters are the parameters selected for the manuscript runs.

## Reproducibility limitation

The original runner fixed the scaffold split files and used seeds 1--5 to
select data partitions, but did not pass a PyTorch training seed. The parameters,
data splits, HPO studies, software protocol, and aggregate manuscript values are
traceable. Exact bit-for-bit reproduction of the archived manuscript means and
standard deviations is not guaranteed without the original final checkpoints,
per-seed predictions, and PyTorch RNG states. A deterministic rerun should pass
both the data seed and PyTorch seed and should be reported as a canonical rerun
rather than silently replacing the archived manuscript snapshot.
