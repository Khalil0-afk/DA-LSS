# Local release and publication

The English source release is maintained at [Khalil0-afk/DA-LSS](https://github.com/Khalil0-afk/DA-LSS). The local packaging script builds a ZIP and checksum; it does not push to GitHub automatically.

The MIT license covers the new implementation only. Original manuscript/review documents, private BMS records and OEM calibration files are excluded. Supplementary Methods is delivered as a Times New Roman PDF with vector equations and an editable Markdown source. Other explanatory documents, license text and archived logs remain Markdown. Machine-readable configuration, numerical data and model artifacts retain their operational formats.

Python comments/docstrings are prohibited by the user's code-format preference. The release packager checks source syntax, comments/docstrings and CJK text. It permits only the generated Supplementary Methods PDF, keeps other prose in Markdown and excludes development runs, caches, virtual environments and build products.

```bash
python -m unittest discover -s tests -v
python scripts/reproduce.py --output runs/pre-release --extended
python scripts/check_response_workflows.py --output runs/response-pre-release
python scripts/package_release.py --output ../DA-LSS-v0.3-English-Times.zip
```

Edit `supplementary/SUPPLEMENTARY_METHODS.md` directly, then rebuild using the commands below. The two stages can use one environment with both optional dependencies installed. On platforms other than Windows, supply `--font-dir` pointing to licensed Times New Roman files named times.ttf, timesi.ttf, timesbd.ttf and timesbi.ttf. Font files are not bundled.

```bash
python -m pip install -e ".[supplementary]"
python scripts/build_supplement.py --layout
python scripts/build_supplement.py
```

The renderer embeds Times New Roman regular/italic/bold variants and draws equations as vector glyphs and fraction/radical rules. Three operator glyphs requiring Times-compatible STIX variants are recorded in `supplementary/font_audit.json`; equation letters remain Times New Roman. This avoids silent default-font substitution and rasterized formula images. Generated workflow logs remain fenced Markdown.

Before publishing empirical claims, supply and verify the author assets listed in `docs/response_commitments.md`, especially the four/20-vehicle split, original complete model, external baselines and hardware traces. Local synthetic evidence does not backfill a claimed published comparison.

The public release includes125 supplied processed MIT CSVs and77 supplied processed HUST CSVs. The packager includes only CSVs from those two named public-data directories and Markdown access documentation under `data/`. The `Leapmotor_EV` directory contains an access statement only. Private vehicle measurements, environments, caches and generated run directories remain excluded.
