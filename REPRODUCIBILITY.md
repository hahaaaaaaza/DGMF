# Reproducibility checklist

## Archived experiment status

- The five scaffold split files are fixed and auditable.
- The original June full-model and ablation runs did not pass
  `--pytorch-seed`; their aggregate values are archival snapshots rather than
  bit-for-bit retraining targets.
- The current release runner passes both `--data-seed` and `--pytorch-seed` and
  defines the canonical deterministic rerun protocol.
- Formal full-model, baseline, and mechanism-control results were produced on
  the RTX 5090 workstation. Retained branch-removal and concat aggregate
  snapshots were produced on the local RTX 4060 workstation.
- The supercomputer was used only for early exploratory HPO and did not supply
  values to the retained 12-endpoint manuscript tables.

- Python: 3.11.15
- GPU used in the paper: NVIDIA GeForce RTX 5090 D v2
- CUDA build: PyTorch 2.11.0+cu128
- Endpoints: 8 classification and 4 regression tasks
- Data split: fixed Bemis-Murcko scaffold partitions, seeds 1-5
- Split ratio target: approximately 7:1:2
- HPO: 20 Optuna trials per endpoint
- HPO budget: 30 epochs, patience 8
- Final budget: 80 epochs, patience 15
- Classification selection metric: validation AUROC
- Regression selection metric: validation Spearman for VDss and the two
  clearance endpoints; validation loss for LD50
- MoLFormer: `ibm-research/MoLFormer-XL-both-10pct`, frozen, pooler output,
  maximum length 256
- GotenNet conformer input: lowest-energy conformer selected from eight ETKDGv3
  candidates after MMFF94 optimization
- Reported values: mean and sample standard deviation over five scaffold seeds

The final selected hyperparameters are in
`configs/best_hyperparameters.csv`. The release runner uses these values
directly and never consults the test set for model selection.

The complete device and artifact audit is in `RESULT_PROVENANCE.md`. Use
`results/reference/paper_dgmf_primary_metrics_full_precision.csv` and
`results/reference/paper_ablation_metrics_full_precision.csv` when checking
the manuscript tables; the shorter original reference CSV is retained for
backward compatibility.

The archived June values cannot be made into exact retraining targets because
their PyTorch RNG states and per-seed predictions were not retained. For a
strictly reproducible submission, run the canonical suite below and populate
the manuscript tables from its generated `paper_suite_summary.csv`. Do not mix
rows from the archived June snapshots with rows from the canonical rerun.
