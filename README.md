# DGMF

Official implementation of **Directed Gated Molecular Fusion of Semantic,
Topological, and Geometric Representations for ADMET Property Prediction**.

DGMF combines frozen MoLFormer embeddings, trainable D-MPNN topology features,
and trainable GotenNet geometry features. Six directed gates exchange information
between the three molecular representations before endpoint prediction.

## Repository layout

```text
dgmf/                  Public DGMF API
chemprop/              Training engine and model implementation
configs/               Endpoint definitions and selected hyperparameters
data/split_manifests/  Fixed scaffold splits for seeds 1-5
scripts/               Data, training, verification, and archiving commands
results/reference/     Full, ablation, and mechanism-control reference results
docs/                  Data and reproducibility documentation
tests/                 Release checks
```

Large model assets, checkpoints, conformer caches, and predictions are provided
separately with the GitHub Release.

## Installation

DGMF uses Python 3.11. Install a PyTorch and PyG build compatible with the local
CUDA runtime, then install the remaining dependencies and this repository:

```powershell
conda create -n dgmf python=3.11 -y
conda activate dgmf
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The internal `chemprop` package name is retained for checkpoint compatibility.

## Prepare data

```powershell
python scripts/prepare_data.py --output-root data/processed
python scripts/verify_data.py --data-root data/processed
```

The scripts download the 12 public TDC endpoints and verify them against the
committed split manifests. See [docs/data.md](docs/data.md).

## Smoke test

```powershell
python scripts/run_experiment.py `
  --variant full `
  --tasks bbb_martins `
  --seeds 1 `
  --epochs 2 `
  --patience 1 `
  --output-root results/smoke
```

## Run experiments

```powershell
# Full DGMF
python scripts/run_suite.py --suite full

# Concat and three branch-removal variants
python scripts/run_suite.py --suite ablation

# Shared-gate and target-agnostic controls
python scripts/run_suite.py --suite controls
```

All commands accept `--tasks`, `--seeds`, `--epochs`, `--patience`, and
`--output-root`. Completed seed runs are skipped unless `--force` is supplied.
The combined result table is written to `results/runs/summary.csv`.

After a suite completes, validate and archive its artifacts:

```powershell
python scripts/archive_results.py --suite full --output-root results/runs
```

## Reference results

- `results/reference/full.csv`: final DGMF results used in the manuscript.
- `results/reference/ablation.csv`: concat and branch-removal results.
- `results/reference/controls.csv`: shared-gate and target-agnostic controls.

The historical ablation table was verified directly from its archived test
predictions. A fresh seeded retraining is a new run and is not expected to be
bitwise identical to the historical checkpoints. The distinction is documented
in [docs/reproducibility.md](docs/reproducibility.md) and
[docs/result_provenance.md](docs/result_provenance.md).

## Validate the release

```powershell
python -m pytest tests -q
python scripts/validate_release.py
```

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). DGMF is released under the
MIT License. The embedded Chemprop-derived code retains its upstream attribution
in [NOTICE.md](NOTICE.md) and
[third_party/chemprop_license.txt](third_party/chemprop_license.txt).
