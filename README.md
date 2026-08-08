# DGMF

Official research code for **Directed Gated Molecular Fusion of Semantic,
Topological, and Geometric Representations for ADMET Property Prediction**.

DGMF combines frozen MoLFormer molecular semantics, a trainable D-MPNN
topological encoder, and a trainable GotenNet geometric encoder. Six directed
source-to-target paths use target-conditioned feature gates before residual
updates and endpoint prediction.

## Repository scope

This release contains the model implementation, endpoint-level
hyperparameters, fixed five-seed scaffold split manifests, archived manuscript
metrics, and a deterministic runner for the 12 primary DGMF tasks. Large model
files, generated conformer caches, checkpoints, and temporary experiment
outputs are intentionally excluded. The original June runs were not fully
deterministic, so archived manuscript values and deterministic reruns are
reported as distinct result records. See
[`RESULT_PROVENANCE.md`](RESULT_PROVENANCE.md).

The internal `chemprop` Python namespace is retained for compatibility with
Chemprop 2.2.1 checkpoints and APIs. The public `dgmf` package, command-line
entry point, model class, and experiment scripts use the paper name. See
[`NOTICE.md`](NOTICE.md) for upstream attribution.

```python
from dgmf import DGMFFusionEncoder
```

The installed `dgmf` command and `python -m dgmf` both expose the training CLI.

## Repository layout

```text
dgmf/                  Public DGMF Python API and module entry point
chemprop/              Chemprop-compatible training engine and DGMF implementation
configs/               Twelve endpoints and selected paper hyperparameters
data/split_manifests/  Hash-only fixed scaffold partitions for seeds 1-5
scripts/               Data preparation, training, controls, and result checks
results/reference/     Manuscript means and standard deviations
tests/                 Dependency-free release-structure test
third_party/           Upstream Chemprop license
```

## Paper environment

The formal full-model, baseline, and mechanism-control experiments used
Windows, Python 3.11.15, PyTorch 2.11.0+cu128, Transformers 4.57.6, RDKit
2026.3.3, Optuna 4.9.0, GotenNet 1.1.2, and an NVIDIA GeForce RTX 5090 D v2.
The retained June branch-removal and concat aggregate snapshots were produced
on the local RTX 4060 workstation. Exact observed environments and result roles
are listed in `results/reference/environment_matrix.csv` and
`results/reference/result_provenance.csv`.

On the RTX 5090 machine used for the paper:

```powershell
conda create -n dgmf python=3.11 -y
conda activate dgmf

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster `
  -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
python -m pip install -r requirements-paper.txt
python -m pip install -e . --no-deps
```

`torch_spline_conv` is not required by DGMF and is intentionally omitted from
the Windows installation command.

## Prepare the 12 TDC endpoints

The repository stores hashes and split labels, not redistributed raw endpoint
tables. Rebuild the exact processed CSV layout from TDC as follows:

```powershell
python scripts/prepare_tdc12.py --output-root data/processed
```

The script downloads the public TDC endpoints, verifies every molecule/label
pair against the release manifests, applies the fixed scaffold partitions for
seeds 1-5, and writes:

```text
data/processed/<endpoint>/seed_<1..5>.csv
```

If a future TDC release changes a dataset, preparation stops with a checksum
mismatch instead of silently evaluating a different sample set.

## MoLFormer weights

The default model identifier is
`ibm-research/MoLFormer-XL-both-10pct`. Transformers can download it on first
use. For an offline machine, place the model directory anywhere and pass its
path with `--molformer-model`.

## Smoke test

```powershell
python scripts/run_dgmf.py `
  --tasks bbb_martins `
  --seeds 1 `
  --epochs 2 `
  --patience 1 `
  --output-root results/smoke
```

## Reproduce the paper suite

```powershell
python scripts/run_paper_suite.py `
  --suite main `
  --tasks all `
  --seeds 1 2 3 4 5 `
  --epochs 80 `
  --patience 15 `
  --output-root results/canonical
```

For an offline MoLFormer copy:

```powershell
python scripts/run_paper_suite.py `
  --suite main `
  --tasks all `
  --seeds 1 2 3 4 5 `
  --molformer-model "D:\models\MoLFormer-XL-both-10pct" `
  --output-root results/canonical
```

The main suite runs the full model, concat fusion, and the three branch-removal
ablations. Completed seed runs are skipped automatically. Per-variant summaries
are written below `results/canonical/<variant>/summary.csv`; the combined file
used to populate the paper is `results/canonical/paper_suite_summary.csv`.

After all runs finish, validate completeness and freeze every paper artifact:

```powershell
python scripts/freeze_canonical_results.py --output-root results/canonical
```

This command requires all 12 endpoints, all five scaffold seeds, and all five
main variants. It writes `paper_metrics_4dp.csv`,
`canonical_artifact_manifest.csv`, and `canonical_freeze.json`. Manuscript
values must come from `paper_metrics_4dp.csv`; archived June values must not be
mixed with the canonical deterministic suite.

Append `--save-directional-messages` to export per-sample values and the
five-seed inputs required for the six directed source-to-target message
strength analysis. These values describe internal information flow and are not
causal feature attributions.

The release runner explicitly sets both the data-loader seed and PyTorch seed
to the scaffold seed. Repeated execution in the same software and hardware
environment is therefore the canonical deterministic protocol. The archived
June manuscript runs did not set `--pytorch-seed`, so their exact aggregate
values cannot be guaranteed by retraining unless the original checkpoints or
per-seed predictions are recovered. Small differences can also remain across
GPU models or library builds, so archived values are compared with a documented
numeric tolerance rather than bitwise equality.

The historical June values cannot be guaranteed by retraining because those
runs did not retain their PyTorch RNG states. The canonical paper must therefore
use the deterministic suite output above. Historical comparisons remain
available for provenance auditing:

```powershell
python scripts/check_reproduction.py `
  --observed results/canonical/full/summary.csv
```

For auditing result origin, full-precision archived values, overwritten-tree
records, and source hashes, see `results/reference/` and
[`RESULT_PROVENANCE.md`](RESULT_PROVENANCE.md).

## Mechanism controls

The same runner exposes the controlled variants used to isolate the fusion
mechanism:

```powershell
python scripts/run_dgmf.py --variant shared_gate --tasks all
python scripts/run_dgmf.py --variant target_agnostic --tasks all
python scripts/run_dgmf.py --variant concat --tasks all
```

All variants reuse the endpoint-specific DGMF hyperparameters and fixed data
partitions.

Run both mechanism controls through the same deterministic suite interface:

```powershell
python scripts/run_paper_suite.py --suite mechanism --tasks all
```

The archived manuscript reference metrics for these two controls are stored in
`results/reference/paper_mechanism_control_metrics.csv`. Check a combined
control run with:

```powershell
python scripts/check_reproduction.py `
  --observed results/canonical/target_agnostic/summary.csv `
  --reference results/reference/paper_mechanism_control_metrics.csv
```

## Citation

Please cite the accompanying DGMF manuscript and the upstream methods listed
in [`CITATION.cff`](CITATION.cff), including Chemprop, MoLFormer, GotenNet, and
Therapeutics Data Commons.
