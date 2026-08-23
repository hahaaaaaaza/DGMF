# Optional deterministic assets

The optional asset archive contains the exact external inputs used by the reference run:

- `MoLFormer-XL-both-10pct`: 31 files, 375,055,309 bytes.
- `molformer_1d_cache`: 65,309 files, 319,549,116 bytes.
- `gotennet_3d_cache`: 46,200 files, 854,055,218 bytes.

Key MoLFormer SHA-256 values:

- `model.safetensors`: `0795977FE7192C4ACDAF052F0E8464AF57BC4BB59211271C5E61AABA2637B9C6`
- `config.json`: `3EF9EAAC8C7CA6282FD6256ED038D151BD4FF42A4FF855367E0D7197BBC1C284`
- `tokenizer.json`: `3DF1F2219653C44FAC9FA03B7F788B372EB2544ECC176737BB9ACA8411B471A5`

Final asset archive:

- File: `STGXAttn_deterministic_assets_20260815.tar`
- Size: 1,633,824,768 bytes
- Entries: 111,548
- SHA-256: `C570F4403B091F84E4D3C2DEB335CA31807CEB8CDADE70020663EB78B6DE5701`

Extract the archive into this package and place the three directories under `assets/`:

```powershell
New-Item -ItemType Directory -Path .\assets -Force
tar -xf .\STGXAttn_deterministic_assets_20260815.tar -C .\assets
```
