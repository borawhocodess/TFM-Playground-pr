# AGENTS.md

## Cursor Cloud specific instructions

TFM-Playground is a single Python 3.12 research library (`tfmplayground`, an open reimplementation of TabPFNv2 called nanoTabPFN). There are no long-running services, servers, databases, or a frontend — "running" the product means executing Python via the sklearn-style API or the CLI training/evaluation scripts. See `README.md` for the documented commands.

### Environment
- Dependencies are installed into a virtualenv at `.venv/` (gitignored). The startup/update script creates it and runs `pip install -e .` plus `ruff`. Run tools via `.venv/bin/python` / `.venv/bin/ruff`, or activate with `source .venv/bin/activate`.
- The VM is CPU-only (no CUDA GPU). Device selection is automatic via `tfmplayground.utils.get_default_device()`, so scripts run on CPU; keep training runs tiny (small `--epochs`/`--steps`/`--batchsize`) or they will be very slow.
- `python3.12-venv` is a system (apt) package required to create the venv; it is baked into the environment and is intentionally not part of the update script.

### Network dependencies (non-obvious)
- First use of `NanoTabPFNClassifier()` / `NanoTabPFNRegressor()` with no `model` argument downloads pretrained checkpoints from `ml.informatik.uni-freiburg.de` into `checkpoints/` (gitignored). Requires outbound network.
- The evaluation pipeline (`tfmplayground/evaluation.py`, and the eval callbacks in the pretrain scripts) downloads datasets/tasks from OpenML on first run; requires network.
- Pretraining needs a prior data source: either a large `.h5` prior dump (download links in `README.md`) or on-the-fly generation via the `TabICLPriorDataLoader` / `TICLPriorDataLoader` classes (no download, good for smoke tests).

### Lint / test / run
- Lint: `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`. `ruff check .` passes on all `*.py`. There are pre-existing findings that are NOT from setup: import ordering in `prior_visualization.ipynb` and formatting of fenced code blocks in `README.md`. Do not "fix" these unless asked.
- Tests: there is no automated test suite in this repo.
- Quick offline-ish smoke test (core inference): the `README.md` breast-cancer snippet using `NanoTabPFNClassifier` (downloads a checkpoint once, then runs `fit`/`predict`).
- Training entry points: `python pretrain_classification.py ...` and `python pretrain_regression.py ...` (need a prior dump `.h5`). Evaluation: `python -m tfmplayground.evaluation -model_type classification -tasks toy_tasks`.

### Gotcha
- `TabICLPriorDataLoader` requires `num_datapoints_min` in addition to `num_datapoints_max` (the `README.md` on-the-fly example omits it).
