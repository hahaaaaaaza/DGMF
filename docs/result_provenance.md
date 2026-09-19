# Result provenance

The repository keeps one reference file for each manuscript result group.

| Result group | Reference file | Verification |
| --- | --- | --- |
| Full DGMF | `results/reference/full.csv` | Final manuscript table audit |
| Concat and branch removal | `results/reference/ablation.csv` | Recomputed from 240 archived prediction files |
| Mechanism controls | `results/reference/controls.csv` | Checked against retained five-seed summaries |

The ablation archive covers four variants, 12 endpoints, and five seeds, for 240
completed runs. Its 48 aggregate rows were reproduced from the saved test
predictions with no verification errors. The reference CSV in that archive is
byte-identical to `results/reference/ablation.csv`.

Large checkpoints and predictions are kept in release assets instead of Git.
Each release package should include its own SHA-256 manifest. Repository result
files are for table checking and do not replace the archived per-seed artifacts.
