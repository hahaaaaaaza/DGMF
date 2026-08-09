# DGMF

Research implementation of **Directed Gated Molecular Fusion of Semantic,
Topological, and Geometric Representations for ADMET Property Prediction**.

DGMF integrates three complementary molecular representations: frozen
MoLFormer embeddings derived from SMILES, trainable D-MPNN representations of
molecular topology, and trainable GotenNet representations of molecular
geometry. Directed target-conditioned gates select source information for each
target representation before residual fusion and endpoint prediction.

## Highlights

- Joint semantic, topological, and geometric molecular modeling.
- Six directed source-to-target gated interaction paths.
- Fixed five-seed scaffold partitions for 12 TDC ADMET endpoints.
- Endpoint-specific hyperparameters used by the manuscript experiments.
- Deterministic experiment runners with per-seed predictions and summaries.
- Controlled variants for concat fusion, branch removal, shared gates, and
  target-agnostic gates.

The public package and command-line interface use the `dgmf` name. The internal
`chemprop` namespace is retained for compatibility with the Chemprop training
engine and existing checkpoints. See [`NOTICE.md`](NOTICE.md) for attribution.

```python
from dgmf import DGMFFusionEncoder
```

## Repository layout

```text
dgmf/                  Public DGMF API and module entry point
chemprop/              Chemprop-compatible training engine and DGMF implementation
configs/               Endpoint definitions and selected hyperparameters
data/split_manifests/  Hash-only scaffold partitions for seeds 1-5
scripts/               Data preparation, training, controls, and result export
results/reference/     Reference tables and result-audit metadata
tests/                 Release validation tests
third_party/           Upstream license notices
```

Large pretrained models, generated conformer caches, checkpoints, and temporary
training outputs are not stored in this repository.

## Installation

DGMF requires Python 3.11. Install a PyTorch build compatible with the CUDA
runtime available on your system, followed by the paper dependencies and the
local package.

Example for CUDA 12.8 on Windows:

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

`torch_spline_conv` is not required. For other CUDA versions or operating
systems, install the corresponding PyTorch and PyG wheels before installing the
remaining dependencies.

## Prepare the TDC endpoints

Raw endpoint tables are not redistributed. The committed manifests contain
hashed sample identifiers and fixed split assignments. Rebuild the processed
datasets from Therapeutics Data Commons with:

```powershell
python scripts/prepare_tdc12.py --output-root data/processed
```

The preparation script validates the downloaded samples against the manifests
and writes:

```text
data/processed/<endpoint>/seed_<1..5>.csv
```

A manifest mismatch stops data preparation rather than silently evaluating a
different dataset version. See [`DATA.md`](DATA.md) for details.

## Pretrained MoLFormer model

The default model identifier is
`ibm-research/MoLFormer-XL-both-10pct`. Transformers may download the model on
first use. For offline execution, pass a local directory with
`--molformer-model`.

## Smoke test

Use one endpoint, one seed, and two epochs to verify the installation:

```powershell
python scripts/run_dgmf.py `
  --tasks bbb_martins `
  --seeds 1 `
  --epochs 2 `
  --patience 1 `
  --output-root results/smoke
```

## Reproduce the main experiments

Run the full model, concat fusion, and the three branch-removal variants on all
12 endpoints and five scaffold seeds:

```powershell
python scripts/run_paper_suite.py `
  --suite main `
  --tasks all `
  --seeds 1 2 3 4 5 `
  --epochs 80 `
  --patience 15 `
  --output-root results/canonical
```

For an offline MoLFormer copy, add:

```powershell
--molformer-model "D:\models\MoLFormer-XL-both-10pct"
```

Completed seed runs are skipped automatically. Each variant writes per-seed
metrics and a summary under:

```text
results/canonical/<variant>/
```

The combined summary is written to:

```text
results/canonical/paper_suite_summary.csv
```

After all runs finish, validate completeness and freeze the reported artifacts:

```powershell
python scripts/freeze_canonical_results.py `
  --suite main `
  --output-root results/canonical
```

The command produces:

```text
paper_metrics_4dp.csv
canonical_artifact_manifest.csv
canonical_freeze.json
```

Use `paper_metrics_4dp.csv` as the source for manuscript tables.

## Mechanism controls

Run the target-agnostic and shared-gate controls through the same deterministic
interface:

```powershell
python scripts/run_paper_suite.py `
  --suite mechanism `
  --tasks all `
  --seeds 1 2 3 4 5 `
  --output-root results/canonical_mechanism
```

Freeze the completed control results with:

```powershell
python scripts/freeze_canonical_results.py `
  --suite mechanism `
  --output-root results/canonical_mechanism
```

## Directional message analysis

Add `--save-directional-messages` to export the six source-to-target message
paths for each sample. These statistics describe information-flow patterns
inside the fusion module and are not causal feature attributions.

## Reproducibility

The experiment runner binds the data seed and PyTorch seed to each scaffold
seed. It records the command, configuration, source hashes, environment
metadata, checkpoint, and test predictions for every run. Repeated execution
under the same software and hardware environment is the strict deterministic
protocol. Small numerical differences may occur across GPU architectures or
library builds.

Run the release checks with:

```powershell
python -m pytest tests -q
python scripts/validate_release.py
```

## Citation

Please cite the accompanying DGMF manuscript and the upstream methods listed in
[`CITATION.cff`](CITATION.cff).

## License

DGMF is released under the MIT License. The internal training engine is derived
from Chemprop 2.2.1 and retains the applicable upstream notice and license in
[`NOTICE.md`](NOTICE.md) and [`third_party/CHEMPROP_LICENSE.txt`](third_party/CHEMPROP_LICENSE.txt).

