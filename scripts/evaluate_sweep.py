"""
Evaluate one or more trained regression checkpoints on TabArena regression tasks.

Usage examples:

  # evaluate all sweep checkpoints auto-discovered under experiments/regression/
  python evaluate_sweep.py --auto

  # evaluate specific checkpoints
  python evaluate_sweep.py \
      "workdir/experiments/regression/sweep-bs8-s1000-muon-mlr5e3-f10-r100/*/*-regressor-checkpoint.pth"

  # also compare against official baselines
  python evaluate_sweep.py --auto --baselines

  # limit dataset size (default: 10000 samples, 500 features)
  python evaluate_sweep.py --auto --max_samples 5000

Output: a table sorted by mean R2, one row per checkpoint.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from tfmplayground.evaluation import get_openml_predictions, TABARENA_TASKS
from tfmplayground.interface import NanoTabPFNRegressor, get_feature_preprocessor

OFFICIAL_CHECKPOINTS = {
    "official-ticl-mlp": "workdir/checkpoints/nanotabpfn-official-regressor-ticl-mlp-checkpoint.pth",
    "official-tabpfn-mlp": "workdir/checkpoints/nanotabpfn-official-regressor-tabpfn-mlp-checkpoint.pth",
}

OPENML_CACHE = os.path.expanduser("~/.cache/openml")


def short_name(checkpoint_path: str) -> str:
    """Extract a readable label from a checkpoint path."""
    base = os.path.basename(checkpoint_path)
    # strip the timestamp-uid prefix and the '-regressor-checkpoint.pth' suffix
    # pattern: YYMMDD-HHMMSS-xxxxxxxx-<name>-regressor-checkpoint.pth
    parts = base.split("-")
    if len(parts) > 3:
        # drop date, time, uid (first 3 dash-separated tokens)
        remainder = "-".join(parts[3:])
        remainder = remainder.replace("-regressor-checkpoint.pth", "").replace("-regressor-best.pth", " (best)")
        return remainder
    return base.replace("-regressor-checkpoint.pth", "").replace("-regressor-best.pth", " (best)")


class _RFWithPreprocessing:
    """RandomForestRegressor using the same get_feature_preprocessor as NanoTabPFNRegressor."""
    def __init__(self):
        self._rf = RandomForestRegressor()

    def fit(self, X, y):
        self._preprocessor = get_feature_preprocessor(X)
        X_proc = self._preprocessor.fit_transform(X)
        self._rf.fit(X_proc, y)
        return self

    def predict(self, X):
        return self._rf.predict(self._preprocessor.transform(X))


def evaluate_random_forest(tasks: list, max_samples: int, max_features: int, cache_dir: str) -> dict[str, float] | None:
    rf = _RFWithPreprocessing()
    try:
        predictions = get_openml_predictions(
            model=rf,
            tasks=tasks,
            max_n_samples=max_samples,
            max_n_features=max_features,
            classification=False,
            cache_directory=cache_dir,
        )
    except Exception as e:
        print(f"  [RF ERROR] {e}", file=sys.stderr)
        return None
    return {name: (r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred) ** 0.5)
            for name, (y_true, y_pred, _) in predictions.items()}


def evaluate_checkpoint(checkpoint_path: str, tasks: list, max_samples: int, max_features: int, device: str, cache_dir: str):
    try:
        regressor = NanoTabPFNRegressor(model=checkpoint_path, device=device)
        regressor.model.eval()
    except Exception as e:
        print(f"  [LOAD ERROR] {checkpoint_path}: {e}", file=sys.stderr)
        return None

    predictions = get_openml_predictions(
        model=regressor,
        tasks=tasks,
        max_n_samples=max_samples,
        max_n_features=max_features,
        classification=False,
        cache_directory=cache_dir,
    )

    scores = {}
    for dataset_name, (y_true, y_pred, _) in predictions.items():
        scores[dataset_name] = (r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred) ** 0.5)
    return scores


def auto_discover(experiments_dir: str) -> list[tuple[str, str]]:
    """Find the most recent checkpoint for each named sweep config.
    Prefers *-regressor-best.pth over *-regressor-checkpoint.pth when available."""
    pattern = os.path.join(experiments_dir, "*", "*", "*-regressor-checkpoint.pth")
    all_ckpts = sorted(glob.glob(pattern))
    # group by config name (parent-parent dir = config name)
    by_config: dict[str, list[str]] = {}
    for path in all_ckpts:
        config = os.path.basename(os.path.dirname(os.path.dirname(path)))
        by_config.setdefault(config, []).append(path)
    # pick the most recent run per config, preferring best.pth if it exists
    result = []
    for config, paths in sorted(by_config.items()):
        latest = sorted(paths)[-1]
        best = latest.replace("-regressor-checkpoint.pth", "-regressor-best.pth")
        if os.path.exists(best):
            result.append((f"{config} (best)", best))
        else:
            result.append((config, latest))
    return result


def print_table(rows: list[tuple[str, dict[str, tuple]]], all_datasets: list[str]):
    """Print a results table sorted by mean R2, showing only mean R2 and mean RMSE."""
    scored = []
    for name, scores in rows:
        r2s = [scores[d][0] for d in all_datasets if d in scores]
        rmses = [scores[d][1] for d in all_datasets if d in scores]
        mean_r2 = np.mean(r2s) if r2s else float("nan")
        mean_rmse = np.mean(rmses) if rmses else float("nan")
        scored.append((name, mean_r2, mean_rmse))
    pin_order = ["RandomForestRegressor", 'DummyRegressor(strategy="mean")']
    pinned = [x for p in pin_order for x in scored if x[0] == p]
    rest = sorted([x for x in scored if x[0] not in pin_order], key=lambda x: x[1], reverse=True)
    scored = pinned + rest

    name_w = max(len(r[0]) for r in scored) if scored else 20
    sep = "-" * (name_w + 24)
    header = f"{'Model':<{name_w}}  {'Mean R2':>10}  {'Mean RMSE':>10}"


    for name, mean_r2, mean_rmse in pinned:
        print(f"{name:<{name_w}}  {mean_r2:>10.2f}  {mean_rmse:>10.2f}")
    print(sep)
    print(header)
    print(sep)
    for name, mean_r2, mean_rmse in rest:
        print(f"{name:<{name_w}}  {mean_r2:>10.2f}  {mean_rmse:>10.2f}")


def main():
    global OPENML_CACHE
    parser = argparse.ArgumentParser(description="Evaluate regression checkpoints on TabArena tasks")
    parser.add_argument("checkpoints", nargs="*",
                        help="Checkpoint .pth paths to evaluate (supports glob patterns)")
    parser.add_argument("--auto", action="store_true", default=True,
                        help="Auto-discover all sweep checkpoints under --experiments_dir")
    parser.add_argument("--experiments_dir", type=str,
                        default="workdir/experiments/regression",
                        help="Root directory to search for checkpoints when --auto is set")
    parser.add_argument("--baselines", action="store_true",
                        help="Also evaluate official baseline checkpoints")
    parser.add_argument("--rf", action="store_true", default=True,
                        help="Include Random Forest as a baseline")
    parser.add_argument("--max_samples", type=int, default=10_000)
    parser.add_argument("--max_features", type=int, default=500)
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: auto)")
    parser.add_argument("--cache_dir", type=str, default=OPENML_CACHE,
                        help="OpenML cache directory")
    args = parser.parse_args()

    OPENML_CACHE = args.cache_dir

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect checkpoints to evaluate
    named_ckpts: list[tuple[str, str]] = []
    extra_results: list[tuple[str, dict[str, float]]] = []

    dummy_scores = get_openml_predictions(
        model=DummyRegressor(strategy="mean"),
        tasks=TABARENA_TASKS,
        max_n_samples=args.max_samples,
        max_n_features=args.max_features,
        classification=False,
        cache_directory=OPENML_CACHE,
    )
    extra_results.append(('DummyRegressor(strategy="mean")', {
        name: (r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred) ** 0.5)
        for name, (y_true, y_pred, _) in dummy_scores.items()
    }))

    if args.rf:
        rf_scores = evaluate_random_forest(TABARENA_TASKS, args.max_samples, args.max_features, OPENML_CACHE)
        if rf_scores is not None:
            extra_results.append(("RandomForestRegressor", rf_scores))

    if args.baselines:
        for label, path in OFFICIAL_CHECKPOINTS.items():
            if os.path.exists(path):
                named_ckpts.append((label, path))
            else:
                print(f"  [SKIP] baseline not found: {path}", file=sys.stderr)

    if args.auto:
        discovered = auto_discover(args.experiments_dir)
        named_ckpts.extend(discovered)

    # explicit paths / globs from positional args
    for pattern in args.checkpoints:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            named_ckpts.append((short_name(path), path))

    if not named_ckpts:
        print("No checkpoints found. Use --auto or pass checkpoint paths.", file=sys.stderr)
        sys.exit(1)


    all_results: list[tuple[str, dict[str, float]]] = list(extra_results)
    all_datasets: list[str] = []
    for _, scores in extra_results:
        for d in scores:
            if d not in all_datasets:
                all_datasets.append(d)

    for name, ckpt_path in named_ckpts:
        scores = evaluate_checkpoint(ckpt_path, TABARENA_TASKS, args.max_samples, args.max_features, device, OPENML_CACHE)
        if scores is None:
            continue
        all_results.append((name, scores))
        for d in scores:
            if d not in all_datasets:
                all_datasets.append(d)

    if not all_results:
        print("No results to show.", file=sys.stderr)
        sys.exit(1)

    print_table(all_results, all_datasets)


if __name__ == "__main__":
    main()
