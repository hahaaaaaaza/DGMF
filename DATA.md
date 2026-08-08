# Data

DGMF uses 12 public endpoints from Therapeutics Data Commons (TDC). Raw labels
and SMILES are obtained through PyTDC and remain subject to the licenses listed
on the corresponding TDC dataset pages.

The repository includes only deterministic split manifests. Each manifest maps
a SHA-256 digest of a TDC identifier, SMILES, and label tuple to `train`, `val`,
or `test`. Including the identifier preserves repeated SMILES/label rows. The
hashes allow the exact five scaffold partitions used in the manuscript to be
rebuilt without redistributing the raw endpoint tables.

Run `python scripts/prepare_tdc12.py` to download, validate, and materialize the
processed CSV files. Do not bypass a manifest mismatch: it means the retrieved
dataset is not identical to the version used for the reported experiments.
