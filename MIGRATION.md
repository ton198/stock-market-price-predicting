# Migration snapshot

Snapshot date: 2026-09-06

This file describes the quantitative project portion of the workspace bundle.
For the complete workspace migration, including resumes, career materials, and
the portfolio site, read `../WORKSPACE_MIGRATION.md`. The bundle intentionally
contains source material and the original dataset, but not virtual environments,
model checkpoints, training runs, or other generated artifacts.

## Snapshot identity

- Repository directory: `stock-market-price-predicting`
- Branch at snapshot time: `main`
- HEAD at snapshot time: `032abe7fafefd1b0a55f4ceb7242c8724ec6c093`
- Dataset: `datasets/nasdaq_multivariate.csv`
- Dataset SHA-256:
  `3f8f3dcd571fceca215778333d6c97d45352bb2544ebc4c2d7467d79cf540ef7`

The working tree was intentionally not clean. The bundle includes the current
modified and untracked source files. The snapshot had these modified tracked
files:

- `.gitignore`
- `diagram.md`
- `dqn_notebook_explanation.md`
- `report/report.tex`
- `source_code/stage1/hybrid_multivariate_LSTM.ipynb`
- `source_code/stage2/DQN.ipynb`

It also had untracked project documentation, packaging files, pipeline code,
scripts, tests, and the canonical audit description. Those files are included
in the bundle.

The source-only bundle omits `.git/`. The companion workspace bundle ending in
`-with-git.tar.gz` carries a migration snapshot branch based on this `main`
commit, includes the current source state, and keeps the extracted working tree
clean while leaving generated model/result files out of the working tree.

## Included

- `datasets/nasdaq_multivariate.csv`
- `quant_pipeline/`
- `scripts/`
- `tests/`
- `source_code/`
- `proposal/`
- `report/`
- project-level Markdown, notebooks, configuration, and requirements files
- `.superpowers/sdd/` task notes and review records
- `docs/superpowers/plans/2026-09-04-quantitative-retraining.md`
- `docs/superpowers/specs/2026-09-04-quantitative-retraining-design.md`

## Intentionally excluded

- `.git/` repository metadata
- `.venv*` virtual environments
- `runs/` training logs, predictions, checkpoints, and evaluation outputs
- `models/` saved model/scaler artifacts
- Python caches such as `__pycache__/` and `*.pyc`
- generated result CSVs and large experiment outputs under `results/`
- model/checkpoint extensions such as `*.pt`, `*.pth`, `*.joblib`, and `*.zip`
- notebook checkpoint directories and common build/test caches

The old generated models and canonical retraining run can be recreated from
the included source and dataset. Their current status and limitations are
documented in `METHODOLOGY.md` and `results/canonical/dqn/README.md`.

## Restore on the destination computer

Extract the archive from its parent workspace directory so the project and
the `docs/superpowers/` files keep their original relative locations:

```bash
tar -xzf stock-market-price-predicting-transfer-20260906.tar.gz -C /path/to/workspace
cd /path/to/workspace/stock-market-price-predicting
```

Create a new environment and install dependencies:

```bash
python3 -m venv .venv-quant
source .venv-quant/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

The previous environment was Python 3.12.3 with CPU PyTorch 2.8.0, scikit-learn
1.9.0, and Stable-Baselines3 2.9.0. These are reference versions only; the
destination machine may install compatible versions according to its hardware.

Before any new experiment, verify the dataset hash:

```bash
sha256sum datasets/nasdaq_multivariate.csv
```

The canonical audit is included as small reproducibility metadata and can also
be regenerated with:

```bash
python scripts/audit_pipeline.py --output results/canonical_data_audit.json
```

That command recreates an excluded generated JSON file; it is not part of the
source bundle by design.
