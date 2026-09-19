# Data

DGMF uses 12 public ADMET endpoints from Therapeutics Data Commons. Raw SMILES
and labels are downloaded through PyTDC and are not redistributed here.

The repository stores 60 hash-only split manifests: 12 endpoints, five scaffold
seeds per endpoint. Each manifest maps a SHA-256 sample identifier to `train`,
`val`, or `test`.

Build and verify the processed files with:

```powershell
python scripts/prepare_data.py --output-root data/processed
python scripts/verify_data.py --data-root data/processed
```

A manifest mismatch means the downloaded dataset differs from the version used
for the reported experiments. Do not bypass this check.
