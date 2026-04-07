"""
Regression prior closely following TabICLv2 (arxiv 2602.11139, Appendix E/F).

Architecture:
  - Each synthetic dataset is a DAG of N nodes (N ~ Uniform(6, 25))
  - Each node has d_node latent dimensions (d_node ~ Uniform(2, 8))
  - Root nodes: mixed distributions (Normal / Uniform / Cauchy)
  - Child nodes: apply random function to aggregated parent values
  - Multi-parent aggregation randomly chosen per node from {sum, product, max, logsumexp}
  - Features X: each column extracted from a random (node, dim) pair — noise is implicit
    (unobserved node dimensions act as latent variables)
  - Target y: one random non-root node dimension
  - DAG-level independence check: resample target/feature assignments if X and y
    share no common ancestors in the DAG

8 function types (same as TabICLv2 Figure 4d):
  Linear, MLP, GP (RFF), Quadratic, Additive, Tree Ensemble (oblivious), EM Assignment, Product

Post-extraction column processing (TabICLv2 converters):
  - Column rescaling: each column × LogNormal(0, 0.5) scale (random feature importance)
  - Random converters: {identity, log1p·sign, √·sign, sigmoid, tanh} per column
  - Categorical features: random subset NN-centroid discretized to K ∈ [2,10] categories

Independence filter (two-stage):
  1. DAG-level: structural check that target and features share ancestors in the DAG
  2. ExtraTrees R² on eval split (sklearn, n_estimators=5); falls back to ridge regression

Missing values: Uniform(0, 0.3) fraction of X replaced with 0.
"""

import math
import random

import torch
import torch.nn.functional as F


class TabPFNRegressionPrior:
    """
    TabICLv2-style regression prior.

    Call get_batch(batch_size, n_rows, n_features, single_eval_pos) →
        x: (batch_size, n_rows, n_features)
        y: (batch_size, n_rows)
    """

    def __init__(self, device='cuda'):
        self.device = device

    @torch.no_grad()
    def get_batch(
        self,
        batch_size: int,
        n_rows: int,
        n_features: int,
        single_eval_pos: int,
        device=None,
    ):
        if device is None:
            device = self.device
        b = batch_size

        # ── shared DAG structure for this batch ──────────────────────────────
        n_nodes = random.randint(max(n_features + 2, 6), 25)
        d_node  = random.randint(2, 8)

        # Cauchy-based edge probability per dataset
        u = torch.empty(b, device=device).uniform_(0.05, 0.95)
        edge_prob = torch.sigmoid(
            torch.tan((u - 0.5) * math.pi) * 0.3
        ).clamp(0.05, 0.70)
        raw      = torch.rand(b, n_nodes, n_nodes, device=device)
        tril_mask = torch.ones(n_nodes, n_nodes, device=device).tril(diagonal=-1)
        adj = (raw < edge_prob.view(b, 1, 1)).float() * tril_mask  # (b, n_nodes, n_nodes)

        # function + aggregation type per (dataset, node)
        fn_types  = torch.randint(0, 8, (b, n_nodes), device=device)
        agg_types = torch.randint(0, 4, (b, n_nodes), device=device)
        root_dist = torch.randint(0, 3, (b,),         device=device)

        # ── DAG-level independence check ──────────────────────────────────────
        # Compute transitive closure: reach[b,j,i]=1 if node i can reach node j
        reach = adj.clone()
        for _ in range(int(math.ceil(math.log2(max(n_nodes, 2)))) + 1):
            reach = (reach + torch.bmm(reach, reach)).clamp(0, 1)

        # Sample target and feature assignments; resample if structurally independent
        target_node, node_assign, dim_assign = self._sample_assignments(
            reach, b, n_nodes, d_node, n_features, device
        )

        # ── generate all node values ──────────────────────────────────────────
        node_vals = self._generate_nodes(
            adj, fn_types, agg_types, root_dist,
            b, n_rows, n_nodes, d_node, device,
        )

        # ── feature extraction: n_features (node, dim) pairs ─────────────────
        x = torch.stack(
            [node_vals[node_assign[f].item()][:, :, dim_assign[f].item()]
             for f in range(n_features)],
            dim=2,
        )  # (b, n_rows, n_features)

        target_dim = random.randint(0, d_node - 1)
        y = node_vals[target_node][:, :, target_dim]  # (b, n_rows)

        # ── column rescaling (random feature importance) ───────────────────────
        col_scale = torch.empty(b, 1, n_features, device=device).log_normal_(0.0, 0.5)
        x = x * col_scale

        # ── random converters ─────────────────────────────────────────────────
        x = self._apply_converters(x, device)

        # ── categorical features (NN-centroid discretization) ─────────────────
        x = self._make_categorical(x, device)

        # ── small additive noise (mostly implicit via latent dims) ────────────
        train_y    = y[:, :single_eval_pos]
        signal_std = train_y.std(dim=1, keepdim=True).clamp(min=1e-6)
        log_nf     = torch.randn(b, 1, device=device) * 0.5 - 2.0
        y = y + torch.randn_like(y) * signal_std * log_nf.exp().clamp(0.001, 1.0)

        # ── missing values ────────────────────────────────────────────────────
        miss_frac = torch.empty(b, device=device).uniform_(0.0, 0.3)
        miss_mask = torch.rand_like(x) < miss_frac.view(b, 1, 1)
        x = x.masked_fill(miss_mask, 0.0)

        # ── independence filter (ExtraTrees / ridge fallback) ─────────────────
        n_eval = n_rows - single_eval_pos
        if n_eval >= 5:
            for _ in range(3):
                bad = self._find_independent(x, y, single_eval_pos)
                if not bad.any():
                    break
                n_bad = int(bad.sum().item())
                x_new, y_new = self.get_batch(
                    n_bad, n_rows, n_features, single_eval_pos, device
                )
                x[bad] = x_new
                y[bad] = y_new

        return x, y

    # ------------------------------------------------------------------ #
    #  DAG-level independence check + assignment                           #
    # ------------------------------------------------------------------ #

    def _sample_assignments(self, reach, b, n_nodes, d_node, n_features, device):
        """
        Sample target node and feature (node, dim) assignments.
        Retry up to 5 times to find assignments where target and features
        share ancestors in the DAG (structural independence avoidance).
        reach: (b, n_nodes, n_nodes) transitive closure, reach[b,j,i]=1 if i→...→j
        """
        for _ in range(5):
            target_node  = random.randint(1, n_nodes - 1)
            node_assign  = torch.randint(0, n_nodes, (n_features,), device=device)
            dim_assign   = torch.randint(0, d_node,  (n_features,), device=device)

            # ancestors of target (including itself)
            target_set   = reach[:, target_node, :].clone()   # (b, n_nodes)
            target_set[:, target_node] = 1.0

            # ancestors of all feature nodes (including themselves)
            feat_set = torch.zeros(b, n_nodes, device=device)
            for na in node_assign.tolist():
                feat_set = (feat_set + reach[:, na, :]).clamp(0, 1)
                feat_set[:, na] = 1.0

            # fraction of datasets that have structural signal
            has_signal = (target_set * feat_set).sum(dim=-1) > 0  # (b,)
            if has_signal.float().mean() > 0.5:
                break  # good enough

        return target_node, node_assign, dim_assign

    # ------------------------------------------------------------------ #
    #  DAG node generation                                                 #
    # ------------------------------------------------------------------ #

    def _generate_nodes(self, adj, fn_types, agg_types, root_dist,
                        b, n_rows, n_nodes, d_node, device):
        node_vals = []
        for j in range(n_nodes):
            root = self._sample_root(b, n_rows, d_node, root_dist, device)
            if j == 0:
                node_vals.append(root)
                continue

            parent_mask = adj[:, j, :j]               # (b, j)
            has_parents = parent_mask.sum(dim=1) > 0   # (b,)
            if not has_parents.any():
                node_vals.append(root)
                continue

            stacked  = torch.stack(node_vals, dim=0)   # (j, b, n_rows, d_node)
            mask_4d  = parent_mask.T.float().view(j, b, 1, 1)

            agg_sum = (stacked * mask_4d).sum(dim=0)
            agg_prd = (stacked * mask_4d + (1.0 - mask_4d)).prod(dim=0)
            agg_max = (stacked * mask_4d + (-1e4) * (1.0 - mask_4d)).amax(dim=0)
            agg_lse = torch.logsumexp(
                stacked * mask_4d + (-1e4) * (1.0 - mask_4d), dim=0
            )
            agg_id  = agg_types[:, j].view(b, 1, 1)
            agg_val = torch.where(agg_id == 0, agg_sum,
                      torch.where(agg_id == 1, agg_prd,
                      torch.where(agg_id == 2, agg_max, agg_lse)))

            fn_out = self._apply_fn(agg_val, fn_types[:, j], b, n_rows, d_node, device)
            out    = torch.where(has_parents.view(b, 1, 1), fn_out, root)
            out    = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-50, 50)
            node_vals.append(out)

        return node_vals

    def _sample_root(self, b, n_rows, d_node, root_dist, device):
        n   = torch.randn(b, n_rows, d_node, device=device)
        u   = torch.empty(b, n_rows, d_node, device=device).uniform_(-3, 3)
        u_c = torch.rand(b, n_rows, d_node, device=device)
        c   = torch.tan((u_c - 0.5) * math.pi).clamp(-10, 10)
        rd  = root_dist.view(b, 1, 1)
        return torch.where(rd == 0, n, torch.where(rd == 1, u, c))

    # ------------------------------------------------------------------ #
    #  8 node-function types: (nb, n_rows, d) → (nb, n_rows, d)           #
    # ------------------------------------------------------------------ #

    def _apply_fn(self, x, fn_ids, b, n_rows, d_node, device):
        out = torch.zeros_like(x)
        fns = [self._fn_linear, self._fn_mlp, self._fn_gp, self._fn_quadratic,
               self._fn_additive, self._fn_tree_ens, self._fn_em, self._fn_product]
        for idx, fn in enumerate(fns):
            mask = fn_ids == idx
            if mask.any():
                out[mask] = fn(x[mask], d_node)
        return out

    def _fn_linear(self, x, d):
        nb, r, _ = x.shape
        W = torch.randn(nb, d, d, device=x.device) / d ** 0.5
        return torch.bmm(x, W)

    def _fn_mlp(self, x, d):
        nb, r, _ = x.shape
        n_layers = random.randint(1, 3)
        h = x
        for i in range(n_layers):
            W = torch.randn(nb, d, d, device=x.device) / d ** 0.5
            h = torch.bmm(h, W)
            if i < n_layers - 1:
                h = F.gelu(h)
        return h

    def _fn_gp(self, x, d):
        nb, r, _ = x.shape
        n_basis   = 64
        ls        = torch.randn(nb, device=x.device).exp().view(nb, 1, 1)
        omega     = torch.randn(nb, d, n_basis, device=x.device) / ls
        phi       = torch.rand(nb, 1, n_basis, device=x.device) * 2 * math.pi
        z         = ((2.0 / n_basis) ** 0.5) * torch.cos(torch.bmm(x, omega) + phi)
        alpha     = torch.randn(nb, n_basis, d, device=x.device) / n_basis ** 0.5
        return torch.bmm(z, alpha)

    def _fn_quadratic(self, x, d):
        nb, r, _ = x.shape
        A    = torch.randn(nb, d, d, device=x.device) / d
        W    = torch.randn(nb, d, d, device=x.device) / d ** 0.5
        quad = (torch.bmm(x, A) * x).sum(dim=-1, keepdim=True).expand(-1, -1, d)
        return quad + torch.bmm(x, W)

    def _fn_additive(self, x, d):
        nb, r, _ = x.shape
        out = torch.zeros(nb, r, d, device=x.device)
        for fi in range(d):
            xi = x[:, :, fi:fi+1]
            W  = torch.randn(nb, 1, d, device=x.device) / d ** 0.5
            if random.random() > 0.5:
                freq  = torch.empty(nb, 1, 1, device=x.device).uniform_(0.3, 4.0)
                phase = torch.empty(nb, 1, 1, device=x.device).uniform_(0, 2 * math.pi)
                nl = torch.sin(freq * xi + phase)
            else:
                nl = xi + 0.3 * xi ** 2 + 0.1 * xi ** 3
            out = out + W * nl
        return out

    def _fn_tree_ens(self, x, d):
        nb, r, _ = x.shape
        n_trees  = random.randint(3, 10)
        depth    = random.randint(2, 5)
        n_leaves = 2 ** depth
        out      = torch.zeros(nb, r, d, device=x.device)
        b_idx    = torch.arange(nb, device=x.device).unsqueeze(1).expand(nb, r)
        for _ in range(n_trees):
            split_feat   = torch.randint(0, d, (nb, depth), device=x.device)
            split_thresh = torch.randn(nb, depth, device=x.device)
            leaf_vals    = torch.randn(nb, n_leaves, d, device=x.device)
            leaf_idx     = torch.zeros(nb, r, dtype=torch.long, device=x.device)
            for level in range(depth):
                feat   = split_feat[:, level].unsqueeze(1).expand(nb, r)
                thresh = split_thresh[:, level].unsqueeze(1).expand(nb, r)
                x_val  = x.gather(2, feat.unsqueeze(-1)).squeeze(-1)
                leaf_idx = leaf_idx * 2 + (x_val > thresh).long()
            leaf_idx = leaf_idx.clamp(0, n_leaves - 1)
            out = out + leaf_vals[b_idx, leaf_idx]
        return out / n_trees

    def _fn_em(self, x, d):
        nb, r, _ = x.shape
        k       = random.randint(2, min(20, max(3, r // 4)))
        centers = torch.randn(nb, k, d, device=x.device)
        dists   = ((x.unsqueeze(2) - centers.unsqueeze(1)) ** 2).sum(dim=-1)
        assign  = dists.argmin(dim=-1)
        cvals   = torch.randn(nb, k, d, device=x.device)
        b_idx   = torch.arange(nb, device=x.device).unsqueeze(1).expand(nb, r)
        return cvals[b_idx, assign]

    def _fn_product(self, x, d):
        nb, r, _ = x.shape
        sub_fns  = [self._fn_linear, self._fn_gp, self._fn_additive]
        y = torch.ones(nb, r, d, device=x.device)
        for _ in range(random.randint(2, 3)):
            factor = sub_fns[random.randint(0, len(sub_fns) - 1)](x, d)
            std    = factor.std(dim=1, keepdim=True).clamp(min=1e-6)
            y      = y * (factor / std)
        return y

    # ------------------------------------------------------------------ #
    #  Column post-processing (TabICLv2 converters)                        #
    # ------------------------------------------------------------------ #

    def _apply_converters(self, x: torch.Tensor, device) -> torch.Tensor:
        """Per-column random warp: identity / log1p·sign / √·sign / sigmoid / tanh."""
        b, r, d = x.shape
        conv_ids = torch.randint(0, 5, (d,), device=device)
        x_out = x.clone()
        for f in range(d):
            xf  = x[:, :, f]
            cid = conv_ids[f].item()
            if cid == 1:
                x_out[:, :, f] = torch.log1p(xf.abs()) * xf.sign()
            elif cid == 2:
                x_out[:, :, f] = xf.abs().sqrt() * xf.sign()
            elif cid == 3:
                x_out[:, :, f] = torch.sigmoid(xf)
            elif cid == 4:
                x_out[:, :, f] = torch.tanh(xf)
            # cid == 0: identity
        return x_out

    def _make_categorical(self, x: torch.Tensor, device) -> torch.Tensor:
        """
        Randomly mark 0–50% of columns as categorical via NN-centroid discretization.
        Each categorical column is mapped to K random centroids (K ∈ [2,10]),
        then normalized to [0, 1].
        """
        b, r, d = x.shape
        n_cat = random.randint(0, max(0, d // 2))
        if n_cat == 0:
            return x
        cat_feats = torch.randperm(d, device=device)[:n_cat]
        x_out = x.clone()
        for f_idx in cat_feats:
            f = f_idx.item()
            xf = x[:, :, f]                                            # (b, r)
            k  = random.randint(2, 10)
            centroids = torch.empty(b, k, device=device).uniform_(-2, 2)
            # NN assignment: argmin |xf - centroid| along k dim
            dists  = (xf.unsqueeze(2) - centroids.unsqueeze(1)).abs()  # (b, r, k)
            assign = dists.argmin(dim=2).float()                        # (b, r) int categories
            x_out[:, :, f] = assign / max(k - 1, 1)                   # normalize to [0, 1]
        return x_out

    # ------------------------------------------------------------------ #
    #  Independence filter                                                  #
    # ------------------------------------------------------------------ #

    def _find_independent(self, x, y, single_eval_pos, threshold=0.01):
        """
        GPU ridge regression R² filter — fast batched independence check.
        Returns bool mask (b,) — True for datasets where y appears independent of X.
        """
        return self._find_independent_ridge(x, y, single_eval_pos, threshold)

    def _find_independent_ridge(self, x, y, single_eval_pos, threshold=0.01):
        """GPU ridge regression R² fallback."""
        b, n_rows, d = x.shape
        n_ev = n_rows - single_eval_pos
        if n_ev < 5:
            return torch.zeros(b, dtype=torch.bool, device=x.device)
        sep  = single_eval_pos
        X_tr = torch.cat([x[:, :sep],  torch.ones(b, sep,  1, device=x.device)], dim=2)
        X_ev = torch.cat([x[:, sep:],  torch.ones(b, n_ev, 1, device=x.device)], dim=2)
        y_tr = y[:, :sep]
        y_ev = y[:, sep:]
        reg  = 1e-4 * torch.eye(d + 1, device=x.device).unsqueeze(0)
        XtX  = torch.bmm(X_tr.transpose(1, 2), X_tr) + reg
        Xty  = torch.bmm(X_tr.transpose(1, 2), y_tr.unsqueeze(2))
        try:
            w = torch.linalg.solve(XtX, Xty)
        except Exception:
            return torch.zeros(b, dtype=torch.bool, device=x.device)
        y_pred = torch.bmm(X_ev, w).squeeze(2)
        ss_res = ((y_ev - y_pred) ** 2).sum(dim=1)
        ss_tot = ((y_ev - y_ev.mean(dim=1, keepdim=True)) ** 2).sum(dim=1).clamp(min=1e-8)
        return (1.0 - ss_res / ss_tot) < threshold


# --------------------------------------------------------------------------- #
# DataLoader                                                                    #
# --------------------------------------------------------------------------- #


class CustomRegressionPriorDataLoader:
    """
    Generates regression datasets on-the-fly using TabPFNRegressionPrior.
    Drop-in replacement for LiveRegressionPriorDataLoader.
    Supports curriculum learning via on_epoch_start().
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        max_features: int = 100,
        max_rows: int = 1000,
        min_features: int = 1,
        min_rows: int = 50,
        device='cuda',
        eval_pos_min_frac: float = 0.33,
        eval_pos_max_frac: float = 0.90,
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.max_features = max_features
        self.max_rows = max_rows
        self.min_features = min_features
        self.min_rows = min_rows
        self.device = device
        self.eval_pos_min_frac = eval_pos_min_frac
        self.eval_pos_max_frac = eval_pos_max_frac
        self.cur_features = max_features
        self.cur_rows = max_rows
        self._prior = TabPFNRegressionPrior(device=device)

    def on_epoch_start(self, epoch: int, total_epochs: int):
        t = min(1.0, (epoch - 1) / max(1, total_epochs - 1))
        self.cur_features = max(
            self.min_features,
            round(self.min_features + t * (self.max_features - self.min_features)),
        )
        self.cur_rows = max(
            self.min_rows,
            round(self.min_rows + t * (self.max_rows - self.min_rows)),
        )

    def __iter__(self):
        for _ in range(self.num_steps):
            n_rows     = self.cur_rows
            n_features = self.cur_features
            min_sep    = max(2, int(n_rows * self.eval_pos_min_frac))
            max_sep    = min(n_rows - 1, int(n_rows * self.eval_pos_max_frac))
            single_eval_pos = torch.randint(min_sep, max_sep + 1, (1,)).item()
            x, y = self._prior.get_batch(
                batch_size=self.batch_size,
                n_rows=n_rows,
                n_features=n_features,
                single_eval_pos=single_eval_pos,
                device=self.device,
            )
            yield dict(x=x, y=y, target_y=y, single_eval_pos=single_eval_pos)

    def __len__(self):
        return self.num_steps


# --------------------------------------------------------------------------- #
# Bucket edge utilities                                                          #
# --------------------------------------------------------------------------- #


def make_bucket_edges_from_custom_prior(
    n_buckets: int,
    batch_size: int,
    num_features: int,
    num_rows: int,
    device,
    n_batches: int = 50,
):
    """Quantile-based bucket edges from prior samples (used when not using quantile loss)."""
    import numpy as np
    prior           = TabPFNRegressionPrior(device=device)
    single_eval_pos = num_rows // 2
    all_y           = []
    with torch.no_grad():
        for _ in range(n_batches):
            _, y = prior.get_batch(
                batch_size=batch_size,
                n_rows=num_rows,
                n_features=num_features,
                single_eval_pos=single_eval_pos,
                device=device,
            )
            y_np    = y.cpu().float().numpy()
            y_means = y_np.mean(axis=1, keepdims=True)
            y_stds  = y_np.std(axis=1, ddof=1, keepdims=True) + 1e-8
            all_y.append(((y_np - y_means) / y_stds).ravel())
    ys        = torch.tensor(np.concatenate(all_y), dtype=torch.float32, device=device)
    print(f"Using {ys.numel()} y samples to estimate {n_buckets} buckets.")
    quantiles = torch.linspace(0, 1, n_buckets + 1, device=device)
    return torch.quantile(ys, quantiles)
