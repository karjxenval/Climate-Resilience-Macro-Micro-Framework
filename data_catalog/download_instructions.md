# Download instructions

The pipeline downloads most open sources automatically when credentials and internet access are available.

1. Authenticate Google Earth Engine:

```bash
earthengine authenticate
```

2. Run the pipeline:

```bash
python scripts/09_reproduce_all.py --config configs/uganda.yaml
```

3. Optional restricted/controlled data:

- For LSMS/UNPS or DHS, download raw files yourself under the data provider's access rules.
- Do not commit raw restricted files to GitHub.
- Put derived, license-compatible validation summaries under `data/processed/validation/` if allowed.
