# %%
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

import openml
from openml.tasks import TaskType
from tfmplayground.evaluation import TABARENA_TASKS

PLOT_DIR     = "workdir/plots/eval"
MAX_SAMPLES  = 10_000
MAX_FEATURES = 500
os.makedirs(PLOT_DIR, exist_ok=True)
ts = datetime.now().strftime("%y%m%d-%H%M%S")

# %%
def plot_dataset(name, X, y, save_path):
    n, f = X.shape
    F_show = min(f, 6)

    # z-score y
    y_z = (y - y.mean()) / (y.std() + 1e-8)

    # pearson r per feature
    r_vals = np.zeros(f)
    for fi in range(f):
        col = X[:, fi]
        valid = ~np.isnan(col)
        if valid.sum() > 10 and col[valid].std() > 1e-8:
            r_vals[fi] = np.corrcoef(col[valid], y[valid])[0, 1]
    top_feat = int(np.argmax(np.abs(r_vals)))

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"{name}  |  n={n}  f={f}", fontsize=14)
    gs  = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # 1. raw y histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(y, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_title("1· raw y", fontsize=10); ax.set_xlabel("y")

    # 2. z-scored y histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(y_z, bins=50, color="mediumseagreen", edgecolor="none", alpha=0.85)
    ax.set_title("2· z-scored y", fontsize=10); ax.set_xlabel("z-scored y")

    # 3. top feature vs y scatter
    ax = fig.add_subplot(gs[0, 2])
    col = X[:, top_feat]; valid = ~np.isnan(col)
    ax.scatter(col[valid], y[valid], s=6, alpha=0.4, color="darkorange", rasterized=True)
    ax.set_title(f"3· top feature x{top_feat} vs y  r={r_vals[top_feat]:.2f}", fontsize=10)
    ax.set_xlabel(f"x{top_feat}"); ax.set_ylabel("y")

    # 4. feature-target |r| bar
    ax = fig.add_subplot(gs[0, 3])
    ax.bar(range(f), np.abs(r_vals), color="steelblue", width=0.8)
    ax.set_title("4· |Pearson r| per feature", fontsize=10)
    ax.set_xlabel("feature"); ax.set_ylabel("|r|"); ax.set_ylim(0, 1)

    # 5. pairplot: top F_show features vs y
    top_feats = np.argsort(np.abs(r_vals))[::-1][:F_show]
    for idx, fi in enumerate(top_feats):
        ax = fig.add_subplot(gs[1, idx] if idx < 4 else gs[2, idx - 4])
        col = X[:, fi]; valid = ~np.isnan(col)
        ax.scatter(col[valid], y[valid], s=5, alpha=0.4, color="darkorange", rasterized=True)
        ax.set_xlabel(f"x{fi}"); ax.set_ylabel("y" if idx % 4 == 0 else "")
        ax.set_title(f"5· x{fi} vs y  r={r_vals[fi]:.2f}", fontsize=9)

    # 6. feature-feature correlation heatmap (top 8 features by |r|)
    top8 = np.argsort(np.abs(r_vals))[::-1][:8]
    X8   = X[:, top8]
    # drop rows with any nan for correlation
    valid_rows = ~np.isnan(X8).any(axis=1)
    if valid_rows.sum() > 10:
        XY      = np.concatenate([X8[valid_rows], y[valid_rows, None]], axis=1)
        labels  = [f"x{i}" for i in top8] + ["y"]
        corr    = np.corrcoef(XY.T)
        ax      = fig.add_subplot(gs[2, 0])
        im      = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6, rotation=45)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_title("6· corr heatmap (top 8 feats)", fontsize=9)
        plt.colorbar(im, ax=ax)

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


# %%
print("loading and plotting TABARENA tasks...")
for task_id in TABARENA_TASKS:
    task = openml.tasks.get_task(task_id, download_splits=False)
    if task.task_type_id != TaskType.SUPERVISED_REGRESSION:
        continue
    dataset = task.get_dataset(download_data=False)
    n_feat  = dataset.qualities["NumberOfFeatures"]
    n_samp  = dataset.qualities["NumberOfInstances"]
    if n_feat > MAX_FEATURES or n_samp > MAX_SAMPLES:
        print(f"  skip {dataset.name} (n={int(n_samp)} f={int(n_feat)})")
        continue

    X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    X_np = X.select_dtypes(include="number").to_numpy(dtype=np.float32, na_value=np.nan)
    y_np = y.to_numpy(dtype=np.float32)

    safe_name = dataset.name.replace("/", "_").replace(" ", "_")[:30]
    path      = os.path.join(PLOT_DIR, f"{ts}-{safe_name}.png")
    print(f"plotting {dataset.name}  n={int(n_samp)}  f={int(n_feat)}")
    plot_dataset(dataset.name, X_np, y_np, path)

print("done")
