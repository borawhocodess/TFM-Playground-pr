# %%
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from tfmplayground.priors.dataloader import LiveRegressionPriorDataLoader

BATCH_SIZE = 4
N_ROWS     = 80
N_FEATURES = 4
EVAL_FRAC  = 0.5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_DIR   = "workdir/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

loader = LiveRegressionPriorDataLoader(
    num_steps=1,
    batch_size=BATCH_SIZE,
    max_features=N_FEATURES,
    max_rows=N_ROWS,
    min_features=N_FEATURES,
    min_rows=N_ROWS,
    device=DEVICE,
)

batch = next(iter(loader))
x_np  = batch["x"].cpu().float().numpy()     # (B, R, F)
y_np  = batch["y"].cpu().float().numpy()     # (B, R)
sep   = batch["single_eval_pos"]
B, R, F = x_np.shape
print(f"sampled B={B} R={R} F={F} sep={sep} device={DEVICE}")

# %%
n_pair = min(F, 4)
fig    = plt.figure(figsize=(20, 16))
fig.suptitle(f"TICL prior  |  B={B} R={R} F={F} sep={sep} device={DEVICE}", fontsize=13)
gs     = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

# ── 1. sample draws (2x2) ────────────────────────────────────────────────────
for i in range(B):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    ax.scatter(x_np[i, :sep, 0], y_np[i, :sep], s=8, alpha=0.6, color="steelblue", label="train")
    ax.scatter(x_np[i, sep:, 0], y_np[i, sep:], s=8, alpha=0.6, color="darkorange", label="eval")
    ax.set_title(f"1· draw {i}  (x0 vs y)", fontsize=9)
    ax.set_xlabel("x0"); ax.set_ylabel("y")
    if i == 0:
        ax.legend(fontsize=7)

# ── 2. target distributions ───────────────────────────────────────────────────
y_z = (y_np - y_np.mean(axis=1, keepdims=True)) / (y_np.std(axis=1, keepdims=True) + 1e-8)

ax2a = fig.add_subplot(gs[0, 2])
ax2a.hist(y_z.ravel(), bins=60, color="steelblue", edgecolor="none", alpha=0.85)
ax2a.set_title("2· pooled z-scored y", fontsize=10); ax2a.set_xlabel("z-scored y")

ax2b = fig.add_subplot(gs[1, 2])
for i in range(B):
    ax2b.hist(y_z[i], bins=25, alpha=0.45, density=True, label=f"ds{i}")
ax2b.set_title("2· per-dataset y density", fontsize=10)
ax2b.set_xlabel("z-scored y"); ax2b.legend(fontsize=6)

# ── 3. feature–target |r| ─────────────────────────────────────────────────────
corr_matrix = np.zeros((B, F))
for i in range(B):
    for f in range(F):
        xf, yi = x_np[i, :, f], y_np[i]
        if xf.std() > 1e-8 and yi.std() > 1e-8:
            corr_matrix[i, f] = np.corrcoef(xf, yi)[0, 1]

ax3a = fig.add_subplot(gs[0, 3])
im3  = ax3a.imshow(np.abs(corr_matrix), aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax3a.set_title("3· |Pearson(xf, y)|", fontsize=10)
ax3a.set_xlabel("feature"); ax3a.set_ylabel("dataset")
plt.colorbar(im3, ax=ax3a)

ax3b = fig.add_subplot(gs[1, 3])
ax3b.bar(range(F), np.abs(corr_matrix).mean(axis=0), color="steelblue")
ax3b.set_title("3· mean |r| per feature", fontsize=10)
ax3b.set_xlabel("feature"); ax3b.set_ylabel("|r|"); ax3b.set_ylim(0, 1)

# ── 4. full correlation heatmap (dataset 0) ───────────────────────────────────
XY         = np.concatenate([x_np[0], y_np[0, :, None]], axis=1)
col_labels = [f"x{f}" for f in range(F)] + ["y"]
corr_full  = np.corrcoef(XY.T)

ax4 = fig.add_subplot(gs[2, :2])
im4 = ax4.imshow(corr_full, cmap="RdBu_r", vmin=-1, vmax=1)
ax4.set_xticks(range(F + 1)); ax4.set_xticklabels(col_labels, fontsize=8)
ax4.set_yticks(range(F + 1)); ax4.set_yticklabels(col_labels, fontsize=8)
ax4.set_title("4· full corr heatmap (dataset 0)", fontsize=10)
plt.colorbar(im4, ax=ax4)
for ii in range(F + 1):
    for jj in range(F + 1):
        ax4.text(jj, ii, f"{corr_full[ii, jj]:.2f}", ha="center", va="center", fontsize=7,
                 color="black" if abs(corr_full[ii, jj]) < 0.6 else "white")

# ── 5. scale diagnostics ──────────────────────────────────────────────────────
ax5a = fig.add_subplot(gs[2, 2])
ax5a.bar(range(B), y_np.std(axis=1), color="steelblue")
ax5a.set_title("5· std(y)", fontsize=10); ax5a.set_xlabel("dataset")

ax5b = fig.add_subplot(gs[2, 3])
ax5b.bar(range(B), np.abs(y_np).max(axis=1), color="darkorange")
ax5b.set_title("5· max|y|", fontsize=10); ax5b.set_xlabel("dataset")

ts   = datetime.now().strftime("%y%m%d-%H%M%S")
path = os.path.join(PLOT_DIR, f"{ts}-ticl-viz.png")
plt.savefig(path, dpi=120, bbox_inches="tight")
print(f"saved: {path}")
plt.show()
