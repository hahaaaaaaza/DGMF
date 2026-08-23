# Result provenance and reproducibility status

This document separates archived manuscript values from results that can be
regenerated deterministically with the current release runner. This distinction
is necessary because the original June 2026 training scripts fixed the scaffold
partition files but did not pass a PyTorch training seed to Chemprop.

## Device-to-result mapping

| Device | Role in the study | Results traced to this device | Evidence |
|---|---|---|---|
| Supercomputer | Early exploratory HPO before migration to the workstation | Preliminary HPO only. No result from this device is used in the 12-endpoint primary or ablation tables. | Project history and absence of a matching manuscript result archive in the retained workspace. |
| Local workstation (`DESKTOP-H8VIUGK`, RTX 4060 Laptop GPU) | Development and June ablation runs | Concat and the three branch-removal ablations used in the manuscript. | The retained classification and regression aggregate CSV files were written on 18 June and 17 June 2026, respectively, and match the manuscript ablation values. |
| RTX 5090 workstation (`DESKTOP-N80T786`, RTX 5090 D v2) | Formal HPO, final full-model runs, baseline comparison, and reviewer controls | The 12 full-model primary values, single-model baselines, and the target-agnostic/shared-gate mechanism controls. | June manuscript workbooks, the 5090 result-tree timestamps, and byte-identical reviewer-control metadata and summaries. |

The machine-readable version of this mapping is stored in
`results/reference/result_provenance.csv`. Package versions are recorded in
`results/reference/environment_matrix.csv`.

The baseline group has mixed reproducibility status. The PyG AttentiveFP and
adapted SGGRL runners explicitly set their training seeds, although a repeated
prediction-hash check is still required before calling those archives
bit-identical. The retained D-MPNN, MoLFormer, ChemBERTa-2, Uni-Mol, and
GotenNet runs were launched through the legacy Chemprop comparison scripts,
which did not pass `--pytorch-seed`. Their prediction files and checkpoints are
traceable, but deterministic retraining requires a canonical rerun. In
particular, representative ChemBERTa-2 and Uni-Mol configs record
`data-seed = 0` and no fixed PyTorch seed.

## Archived manuscript snapshots

The full-model values in `paper_dgmf_primary_metrics_full_precision.csv` are the
June manuscript snapshot exported from the 5090 workstation. The branch-removal
and concat values in `paper_ablation_metrics_full_precision.csv` are the June
aggregate snapshots retained on the local workstation.

The current `hpo20_cls` and `hpo20_reg` trees on the 5090 workstation were
written again on 21 July 2026. Their selected values no longer equal the June
manuscript snapshot. A record of the currently observed selected summaries is
provided in `current_5090_hpo_snapshot_20260721.csv`; it must not be substituted
silently for the manuscript table.

File hashes for all retained source snapshots are listed in
`legacy_source_hashes.csv`. These hashes verify that an archived table has not
changed. They do not make a stochastic training run deterministic.

## Why the old values cannot be promised bit-for-bit from retraining

The original `run_two_config_5seed_hpo.py` and
`run_stgxattn_supplementary.py` selected `seed_1.csv` through `seed_5.csv`, but
did not pass `--pytorch-seed` to Chemprop. Chemprop therefore generated a fresh
training seed and disabled deterministic training. Consequently:

1. the five scaffold partitions and test samples are reproducible;
2. the original network initialization and CUDA operation sequence are not;
3. rerunning the legacy command can produce nearby, but not identical, metrics;
4. changing the GPU or RDKit/PyG build can introduce additional differences.

Exact recovery of the June values would require the original five checkpoints
or per-seed test predictions. The retained aggregate workbooks alone are not
sufficient to reconstruct them.

## Canonical reproducible protocol from this release

`scripts/run_dgmf.py` explicitly passes both `--data-seed` and
`--pytorch-seed`. For a canonical rerun, use the RTX 5090 environment in
`requirements-paper.txt`, the released split manifests, and seeds 1--5. Keep
the generated checkpoints, per-seed predictions, `summary.csv`, environment
report, and source commit together as one immutable result release.

Within the same software and hardware environment, this protocol is intended
to reproduce the canonical rerun deterministically. Cross-device equality
should be evaluated with a documented numerical tolerance rather than bitwise
identity. If the manuscript must claim exact reproducibility, its tables should
be updated once to the canonical deterministic rerun and then frozen with the
corresponding checkpoints and predictions.
