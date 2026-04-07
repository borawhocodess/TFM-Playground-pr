"""
Custom regression prior following TabPFN v1 / TabICL v2 papers.

Prior bag mixing (8 function types):
  - Linear                  (12%)  y = Wx + b
  - MLP                     (25%)  random BNN, depth/width from Gamma, weights from LogNormal
  - GP via RFF              (15%)  RBF kernel, lengthscale/outputscale from LogNormal
  - Quadratic               (10%)  y = x^T A x + Wx (pairwise interactions)
  - Additive                (10%)  sum of 1-D polynomial + sinusoidal terms
  - Oblivious Tree Ensemble (13%)  CatBoost-style symmetric trees (TabICLv2)
  - EM Assignment           (10%)  prototype-cluster assignment (TabICLv2)
  - Product                 ( 5%)  product of simpler random functions (TabICLv2)

Features: SCM with random DAG — each dataset gets its own random DAG.
  - Root distributions: mixed (Normal, Uniform, Cauchy) chosen per dataset
  - Edge probabilities: heavy-tailed via Cauchy-based sampling (sparse to dense)
  - Child nonlinearities: {tanh, relu, linear, sin, abs, sign, step}
  - Kumaraswamy warping: applied to a random subset of features after DAG generation
    CDF(u; a, b) = 1 - (1 - sigmoid(x)^a)^b, a,b ~ LogNormal → non-Gaussian marginals

Noise: LogNormal scale relative to training-split signal std (TabPFN v1).
Missing values: randomly injected with probability ~ Uniform(0, 0.3), replaced with 0.
Independence filter: GPU linear R² check — datasets where y ⊥ x (R² < 0.01) are
  regenerated up to 3 times, discarding trivially uninformative samples (TabICLv2).

All generation runs on-device in float32.
"""

import math
import torch
import torch.nn.functional as F


class TabPFNRegressionPrior:
    """
    Generates synthetic tabular regression datasets.

    Call get_batch(batch_size, n_rows, n_features, single_eval_pos) to get
    (x, y) tensors of shape (batch_size, n_rows, n_features) and (batch_size, n_rows).
    """

    # linear, mlp, gp_rff, quadratic, additive, oblivious_tree_ensemble, em_assignment, product
    _FUNCTION_PROBS = torch.tensor([0.12, 0.25, 0.15, 0.10, 0.10, 0.13, 0.10, 0.05])

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
        """
        Returns:
            x: (batch_size, n_rows, n_features)
            y: (batch_size, n_rows)
        """
        if device is None:
            device = self.device

        x = self._sample_x(batch_size, n_rows, n_features, device)

        probs = self._FUNCTION_PROBS.to(device)
        types = torch.multinomial(probs.expand(batch_size, -1), 1).squeeze(1)

        y = torch.zeros(batch_size, n_rows, device=device)
        fns = [
            self._linear,
            self._mlp,
            self._gp_rff,
            self._quadratic,
            self._additive,
            self._oblivious_tree_ensemble,
            self._em_assignment,
            self._product,
        ]
        for fn_idx, fn in enumerate(fns):
            mask = types == fn_idx
            if mask.any():
                y[mask] = fn(x[mask])

        # LogNormal noise scale relative to training-split signal std (TabPFN v1 style)
        train_y = y[:, :single_eval_pos]
        signal_std = train_y.std(dim=1, keepdim=True).clamp(min=1e-6)
        log_noise_frac = torch.randn(batch_size, 1, device=device) * 1.0 - 1.5
        noise_frac = log_noise_frac.exp().clamp(0.001, 5.0)
        y = y + torch.randn_like(y) * signal_std * noise_frac

        # Missing values: random fraction ~ Uniform(0, 0.3) of x set to 0
        missing_frac = torch.empty(batch_size, device=device).uniform_(0.0, 0.3)
        missing_mask = torch.rand_like(x) < missing_frac.view(batch_size, 1, 1)
        x = x.masked_fill(missing_mask, 0.0)

        # Independence filter: discard datasets where y is nearly independent of x.
        # Up to 3 retries; each failed dataset is replaced by a fresh sample.
        n_eval = n_rows - single_eval_pos
        if n_eval >= 5:
            for _ in range(3):
                bad = self._find_independent(x, y, single_eval_pos)
                if not bad.any():
                    break
                n_bad = int(bad.sum().item())
                x_new, y_new = self.get_batch(n_bad, n_rows, n_features, single_eval_pos, device)
                x[bad] = x_new
                y[bad] = y_new

        return x, y

    # -------------------------------------------------------------------------
    # independence filter

    def _find_independent(self, x, y, single_eval_pos, threshold=0.01):
        """
        GPU-native linear R² filter.
        Returns bool mask (b,) — True for datasets where y appears nearly independent of x.
        Fits ridge regression on train split; evaluates on eval split.
        """
        b, n_rows, d = x.shape
        n_ev = n_rows - single_eval_pos
        if n_ev < 5:
            return torch.zeros(b, dtype=torch.bool, device=x.device)

        sep = single_eval_pos
        x_tr = x[:, :sep]              # (b, sep, d)
        y_tr = y[:, :sep]              # (b, sep)
        x_ev = x[:, sep:]              # (b, n_ev, d)
        y_ev = y[:, sep:]              # (b, n_ev)

        ones_tr = torch.ones(b, sep, 1, device=x.device)
        X_tr = torch.cat([x_tr, ones_tr], dim=2)                       # (b, sep, d+1)
        reg = 1e-4 * torch.eye(d + 1, device=x.device).unsqueeze(0)
        XtX = torch.bmm(X_tr.transpose(1, 2), X_tr) + reg              # (b, d+1, d+1)
        Xty = torch.bmm(X_tr.transpose(1, 2), y_tr.unsqueeze(2))       # (b, d+1, 1)
        try:
            w = torch.linalg.solve(XtX, Xty)                           # (b, d+1, 1)
        except Exception:
            return torch.zeros(b, dtype=torch.bool, device=x.device)

        ones_ev = torch.ones(b, n_ev, 1, device=x.device)
        X_ev = torch.cat([x_ev, ones_ev], dim=2)
        y_pred = torch.bmm(X_ev, w).squeeze(2)                         # (b, n_ev)

        ss_res = ((y_ev - y_pred) ** 2).sum(dim=1)
        ss_tot = ((y_ev - y_ev.mean(dim=1, keepdim=True)) ** 2).sum(dim=1).clamp(min=1e-8)
        r2 = 1.0 - ss_res / ss_tot                                     # (b,)
        return r2 < threshold

    # -------------------------------------------------------------------------
    # feature generation

    def _sample_x(self, batch_size, n_rows, n_features, device):
        """
        SCM with random DAG structure.

        Enhancements over original:
        - Root distributions: Normal / Uniform / Cauchy chosen per dataset
        - Edge probabilities: Cauchy-based heavy-tailed sampling (sparse to dense)
        - Child nonlinearities: {tanh, relu, linear, sin, abs, sign, step}
        - Kumaraswamy warping on a random subset of features
        - Feature order permuted at the end (DAG topo order != index order)
        """
        if n_features == 1:
            return torch.randn(batch_size, n_rows, 1, device=device)

        # Heavy-tailed edge probabilities via sigmoid of a Cauchy sample
        # Cauchy(0, 0.3): gives a wide range from very sparse to very dense graphs
        u = torch.empty(batch_size, device=device).uniform_(0.05, 0.95)
        edge_prob = torch.sigmoid(
            torch.tan((u - 0.5) * math.pi) * 0.3
        ).clamp(0.05, 0.70)                                             # (b,)

        raw = torch.rand(batch_size, n_features, n_features, device=device)
        tril_mask = torch.ones(n_features, n_features, device=device).tril(diagonal=-1)
        adj = (raw < edge_prob.view(batch_size, 1, 1)).float() * tril_mask  # (b, j, i)

        # 7 nonlinearities for child nodes
        act_ids = torch.randint(0, 7, (batch_size,), device=device)

        # root distribution: 0=Normal, 1=Uniform(-3,3), 2=Cauchy(0,1)
        root_dist_id = torch.randint(0, 3, (batch_size,), device=device)

        x = torch.zeros(batch_size, n_rows, n_features, device=device)

        for j in range(n_features):
            noise_n = torch.randn(batch_size, n_rows, device=device)

            if j == 0:
                # root node — mixed distribution per dataset
                noise_u = torch.empty(batch_size, n_rows, device=device).uniform_(-3, 3)
                u_c = torch.rand(batch_size, n_rows, device=device)
                noise_c = torch.tan((u_c - 0.5) * math.pi).clamp(-10, 10)
                rd = root_dist_id.view(-1, 1)
                feat = torch.where(rd == 0, noise_n,
                       torch.where(rd == 1, noise_u, noise_c))
                x[:, :, j] = feat
                continue

            parent_w = adj[:, j, :j]                # (b, j) — 1 if i -> j
            has_parents = parent_w.sum(dim=1) > 0   # (b,)

            if not has_parents.any():
                x[:, :, j] = noise_n
                continue

            w = torch.randn(batch_size, j, device=device) / (j ** 0.5) * parent_w
            signal = torch.einsum('bri,bi->br', x[:, :, :j], w)        # (b, n_rows)

            out = torch.zeros_like(signal)
            out[act_ids == 0] = torch.tanh(signal[act_ids == 0])
            out[act_ids == 1] = F.relu(signal[act_ids == 1])
            out[act_ids == 2] = signal[act_ids == 2]
            out[act_ids == 3] = torch.sin(signal[act_ids == 3])
            out[act_ids == 4] = signal[act_ids == 4].abs()
            out[act_ids == 5] = signal[act_ids == 5].sign()
            out[act_ids == 6] = (signal[act_ids == 6] > 0).float()

            noise_scale = torch.empty(batch_size, 1, device=device).uniform_(0.1, 1.0)
            x_j = noise_n.clone()
            x_j[has_parents] = out[has_parents] + noise_n[has_parents] * noise_scale[has_parents]
            x[:, :, j] = x_j

        # Kumaraswamy warping on a random subset of features
        # CDF(u; a, b) = 1 - (1 - sigmoid(x)^a)^b, mapped back via logit
        n_warp = torch.randint(0, n_features // 2 + 2, (1,)).item()
        if n_warp > 0 and n_features >= 2:
            n_warp = min(n_warp, n_features)
            warp_feats = torch.randperm(n_features, device=device)[:n_warp]
            a = torch.empty(batch_size, 1, n_warp, device=device).log_normal_(0.0, 0.5).clamp(0.1, 5.0)
            b_ = torch.empty(batch_size, 1, n_warp, device=device).log_normal_(0.0, 0.5).clamp(0.1, 5.0)
            x_warp = x[:, :, warp_feats]                               # (b, r, n_warp)
            u_warp = torch.sigmoid(x_warp)
            x_warped = 1.0 - (1.0 - u_warp.pow(a)).pow(b_)
            # map back to R via logit
            x_warped = x_warped.clamp(1e-6, 1.0 - 1e-6)
            x_warped = torch.log(x_warped / (1.0 - x_warped))
            x[:, :, warp_feats] = x_warped

        # random permutation of feature indices so DAG topo order != index order
        perm = torch.argsort(torch.rand(batch_size, n_features, device=device), dim=1)
        x = torch.gather(x, 2, perm.unsqueeze(1).expand(-1, n_rows, -1))

        return x

    # -------------------------------------------------------------------------
    # function types

    def _linear(self, x):
        """y = x @ w  (random weight vector, LogNormal scale)"""
        b, r, d = x.shape
        w_scale = torch.randn(b, device=x.device).exp()
        w = torch.randn(b, d, 1, device=x.device) / (d ** 0.5) * w_scale.view(b, 1, 1)
        return torch.bmm(x, w).squeeze(-1)

    def _mlp(self, x):
        """
        Random MLP following TabPFN v1.
        - depth ~ Normal(2, 1) clipped to [1, 5]
        - width ~ Normal(64, 48) clipped to [4, 256]
        - weight scale ~ LogNormal
        - activation: {tanh, relu, gelu}
        """
        b, r, d = x.shape
        device = x.device
        n_hidden = max(4, int(abs(torch.empty(1).normal_(64, 48).item())))
        n_hidden = min(n_hidden, 256)
        n_layers = max(1, int(abs(torch.empty(1).normal_(2, 1).item())))
        n_layers = min(n_layers, 5)
        act_idx = torch.randint(0, 3, (1,)).item()
        acts = [torch.tanh, F.relu, F.gelu]
        act = acts[act_idx]
        w_scale = torch.randn(b, device=device).exp().view(b, 1, 1)
        h = x
        in_dim = d
        for layer in range(n_layers):
            out_dim = n_hidden if layer < n_layers - 1 else 1
            W = torch.randn(b, in_dim, out_dim, device=device) / (in_dim ** 0.5) * w_scale
            b_bias = torch.zeros(b, 1, out_dim, device=device)
            h = torch.bmm(h, W) + b_bias
            if layer < n_layers - 1:
                h = act(h)
            in_dim = out_dim
        return h.squeeze(-1)

    def _gp_rff(self, x):
        """
        GP via Random Fourier Features (Bochner's theorem for RBF kernel).
        lengthscale ~ LogNormal, outputscale ~ LogNormal  (TabPFN v1 style).
        """
        b, r, d = x.shape
        device = x.device
        n_basis = 128
        ls = torch.randn(b, device=device).exp().view(b, 1, 1)
        outputscale = torch.randn(b, device=device).exp().view(b, 1)
        omega = torch.randn(b, d, n_basis, device=device) / ls
        phi = torch.rand(b, 1, n_basis, device=device) * 2 * math.pi
        z = torch.bmm(x, omega) + phi
        z = ((2.0 / n_basis) ** 0.5) * torch.cos(z)
        alpha = torch.randn(b, n_basis, 1, device=device) / (n_basis ** 0.5)
        return torch.bmm(z, alpha).squeeze(-1) * outputscale

    def _quadratic(self, x):
        """
        Quadratic model: y = diag(x @ A @ x^T) + x @ w
        A = U U^T (random low-rank symmetric), captures pairwise interactions.
        """
        b, r, d = x.shape
        device = x.device
        rank = min(d, torch.randint(1, max(2, d // 2 + 1), (1,)).item())
        U = torch.randn(b, d, rank, device=device) / (d * rank) ** 0.5
        A = torch.bmm(U, U.transpose(1, 2))
        quad = (torch.bmm(x, A) * x).sum(dim=-1)
        lin = torch.bmm(x, torch.randn(b, d, 1, device=device) / (d ** 0.5)).squeeze(-1)
        return quad + lin

    def _additive(self, x):
        """
        Additive model: y = sum_i f_i(x_i) where each f_i is a random 1-D function
        combining a linear term with a sinusoidal or cubic polynomial term.
        """
        b, r, d = x.shape
        device = x.device
        feat_w = torch.randn(b, d, device=device)
        feat_w = feat_w / feat_w.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        y = torch.zeros(b, r, device=device)
        for fi in range(d):
            xi = x[:, :, fi]
            y = y + feat_w[:, fi:fi+1] * xi
            if torch.rand(1) > 0.5:
                freq = torch.empty(b, 1, device=device).uniform_(0.3, 4.0)
                phase = torch.empty(b, 1, device=device).uniform_(0, 2 * math.pi)
                nl = torch.sin(freq * xi + phase)
            else:
                a = torch.randn(b, 1, device=device) * 0.5
                b2 = torch.randn(b, 1, device=device) * 0.3
                c = torch.randn(b, 1, device=device) * 0.1
                nl = a * xi + b2 * xi ** 2 + c * xi ** 3
            y = y + feat_w[:, fi:fi+1] * nl
        return y

    def _oblivious_tree_ensemble(self, x):
        """
        CatBoost-style oblivious tree ensemble (TabICLv2).
        All nodes at the same depth share the same split feature and threshold.
        Averages over n_trees ~ Uniform(3, 14) trees.
        """
        b, r, d = x.shape
        device = x.device
        n_trees = torch.randint(3, 15, (1,)).item()
        depth = torch.randint(2, 7, (1,)).item()
        n_leaves = 2 ** depth

        y = torch.zeros(b, r, device=device)
        for _ in range(n_trees):
            # oblivious: one split per depth level, shared across all nodes
            split_feat = torch.randint(0, d, (b, depth), device=device)   # (b, depth)
            split_thresh = torch.randn(b, depth, device=device)            # (b, depth)
            leaf_values = torch.randn(b, n_leaves, device=device)

            leaf_idx = torch.zeros(b, r, dtype=torch.long, device=device)
            for level in range(depth):
                feat = split_feat[:, level].unsqueeze(1).expand(b, r)     # (b, r)
                thresh = split_thresh[:, level].unsqueeze(1).expand(b, r)
                x_val = x.gather(2, feat.unsqueeze(-1)).squeeze(-1)
                go_right = (x_val > thresh).long()
                leaf_idx = leaf_idx * 2 + go_right

            leaf_idx = leaf_idx.clamp(0, n_leaves - 1)
            # gather leaf values: leaf_values (b, n_leaves), leaf_idx (b, r)
            b_idx = torch.arange(b, device=device).unsqueeze(1).expand(b, r)
            y = y + leaf_values[b_idx, leaf_idx]

        return y / n_trees

    def _em_assignment(self, x):
        """
        Prototype-based cluster assignment (TabICLv2 EM Assignment function).
        Assigns each row to the nearest random cluster center; each cluster maps to
        a random scalar output value + small Gaussian noise.
        """
        b, r, d = x.shape
        device = x.device
        n_clusters = torch.randint(2, max(3, min(20, r // 5) + 1), (1,)).item()
        centers = torch.randn(b, n_clusters, d, device=device)

        x_exp = x.unsqueeze(2)                                          # (b, r, 1, d)
        c_exp = centers.unsqueeze(1)                                    # (b, 1, k, d)
        dists = ((x_exp - c_exp) ** 2).sum(dim=-1)                     # (b, r, k)
        assignments = dists.argmin(dim=-1)                              # (b, r)

        cluster_values = torch.randn(b, n_clusters, device=device)
        b_idx = torch.arange(b, device=device).unsqueeze(1).expand(b, r)
        return cluster_values[b_idx, assignments]                       # (b, r)

    def _product(self, x):
        """
        Product of 2–3 simpler random functions (TabICLv2 Product function).
        Sub-functions drawn from {linear, gp_rff, additive}; each factor is
        z-scored before multiplication to prevent output explosion.
        """
        b, r, _ = x.shape
        device = x.device
        sub_fns = [self._linear, self._gp_rff, self._additive]
        n_factors = torch.randint(2, 4, (1,)).item()
        y = torch.ones(b, r, device=device)
        for _ in range(n_factors):
            idx = torch.randint(0, len(sub_fns), (1,)).item()
            factor = sub_fns[idx](x)
            f_std = factor.std(dim=1, keepdim=True).clamp(min=1e-6)
            y = y * (factor / f_std)
        return y


# -----------------------------------------------------------------------------
# DataLoader


class CustomRegressionPriorDataLoader:
    """
    DataLoader that generates regression datasets on-the-fly using TabPFNRegressionPrior.

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
        """Linearly ramp difficulty from (min_features, min_rows) to (max, max)."""
        t = min(1.0, (epoch - 1) / max(1, total_epochs - 1))
        self.cur_features = max(self.min_features,
                                round(self.min_features + t * (self.max_features - self.min_features)))
        self.cur_rows = max(self.min_rows,
                            round(self.min_rows + t * (self.max_rows - self.min_rows)))

    def __iter__(self):
        for _ in range(self.num_steps):
            n_rows = self.cur_rows
            n_features = self.cur_features

            min_sep = max(2, int(n_rows * self.eval_pos_min_frac))
            max_sep = min(n_rows - 1, int(n_rows * self.eval_pos_max_frac))
            single_eval_pos = torch.randint(min_sep, max_sep + 1, (1,)).item()

            x, y = self._prior.get_batch(
                batch_size=self.batch_size,
                n_rows=n_rows,
                n_features=n_features,
                single_eval_pos=single_eval_pos,
                device=self.device,
            )

            yield dict(
                x=x,
                y=y,
                target_y=y,
                single_eval_pos=single_eval_pos,
            )

    def __len__(self):
        return self.num_steps


# -----------------------------------------------------------------------------
# Bucket edge utilities


def make_bucket_edges_from_custom_prior(
    n_buckets: int,
    batch_size: int,
    num_features: int,
    num_rows: int,
    device,
    n_batches: int = 50,
):
    """
    Compute quantile-based bucket edges by sampling from TabPFNRegressionPrior.
    Targets are z-scored per dataset before pooling, matching what train.py does.
    """
    import numpy as np

    prior = TabPFNRegressionPrior(device=device)
    single_eval_pos = num_rows // 2
    all_y = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = prior.get_batch(
                batch_size=batch_size,
                n_rows=num_rows,
                n_features=num_features,
                single_eval_pos=single_eval_pos,
                device=device,
            )
            y_np = y.cpu().float().numpy()
            y_means = y_np.mean(axis=1, keepdims=True)
            y_stds = y_np.std(axis=1, ddof=1, keepdims=True) + 1e-8
            all_y.append(((y_np - y_means) / y_stds).ravel())

    ys = torch.tensor(np.concatenate(all_y), dtype=torch.float32, device=device)
    print(f"Using {ys.numel()} y samples to estimate {n_buckets} buckets.")
    quantiles = torch.linspace(0, 1, n_buckets + 1, device=device)
    edges = torch.quantile(ys, quantiles)
    return edges
