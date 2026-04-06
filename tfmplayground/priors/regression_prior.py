"""
Custom regression prior following TabPFN v1 / TabICL papers.

Prior bag mixing:
  - Linear          (15%)  y = Wx + b
  - MLP             (40%)  random BNN, depth/width from Gamma, weights from LogNormal
  - GP via RFF      (25%)  RBF kernel, lengthscale/outputscale from LogNormal
  - Quadratic       (10%)  y = x^T A x + Wx (pairwise interactions)
  - Additive        (10%)  sum of 1-D polynomial + sinusoidal terms

Features: SCM with random DAG — each dataset gets its own random DAG, root nodes are
N(0,1), child nodes are nonlinear functions of parents + Gaussian noise.
Edge probability ~ Uniform(0.1, 0.5), nonlinearity ~ {tanh, relu, linear, sin}.
Noise: LogNormal scale relative to training-split signal std (following TabPFN v1).

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

    # (linear, mlp, gp_rff, quadratic, additive)
    _FUNCTION_PROBS = torch.tensor([0.15, 0.40, 0.25, 0.10, 0.10])

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

        # assign a function type to each dataset in the batch
        probs = self._FUNCTION_PROBS.to(device)
        types = torch.multinomial(probs.expand(batch_size, -1), 1).squeeze(1)

        y = torch.zeros(batch_size, n_rows, device=device)
        fns = [self._linear, self._mlp, self._gp_rff, self._quadratic, self._additive]
        for fn_idx, fn in enumerate(fns):
            mask = types == fn_idx
            if mask.any():
                y[mask] = fn(x[mask])

        # LogNormal noise scale relative to training-split signal std (TabPFN v1 style)
        train_y = y[:, :single_eval_pos]
        signal_std = train_y.std(dim=1, keepdim=True).clamp(min=1e-6)
        # log_noise_frac ~ LogNormal(mean=-1.5, std=1.0) → E[noise/signal] ≈ 0.22, heavy tail
        log_noise_frac = torch.randn(batch_size, 1, device=device) * 1.0 - 1.5
        noise_frac = log_noise_frac.exp().clamp(0.001, 5.0)
        y = y + torch.randn_like(y) * signal_std * noise_frac

        return x, y

    # -------------------------------------------------------------------------
    # feature generation

    def _sample_x(self, batch_size, n_rows, n_features, device):
        """
        SCM with random DAG structure (per TabPFN v1).

        Each dataset gets its own DAG:
          - edge inclusion probability ~ Uniform(0.1, 0.5)
          - adj[b, j, i] = 1 means feature i is a parent of feature j (i < j in topo order)
          - root nodes (no parents) ~ N(0, 1)
          - child nodes = nonlinear_fn(weighted sum of parents) + Gaussian noise
          - nonlinearity sampled per dataset from {tanh, relu, linear, sin}
        Feature indices are randomly permuted at the end so roots aren't always feature 0.
        """
        if n_features == 1:
            return torch.randn(batch_size, n_rows, 1, device=device)

        # edge probability per dataset
        edge_prob = torch.empty(batch_size, device=device).uniform_(0.1, 0.5)

        # lower-triangular DAG adjacency: adj[b, j, i] = 1 iff i -> j (i < j)
        raw = torch.rand(batch_size, n_features, n_features, device=device)
        tril_mask = torch.ones(n_features, n_features, device=device).tril(diagonal=-1)
        adj = (raw < edge_prob.view(batch_size, 1, 1)).float() * tril_mask  # (b, j, i)

        # nonlinearity assignment per dataset
        act_ids = torch.randint(0, 4, (batch_size,), device=device)  # 0=tanh,1=relu,2=linear,3=sin

        x = torch.zeros(batch_size, n_rows, n_features, device=device)

        for j in range(n_features):
            noise = torch.randn(batch_size, n_rows, device=device)

            if j == 0:
                x[:, :, j] = noise
                continue

            parent_w = adj[:, j, :j]          # (b, j)   — 1 if parent, 0 otherwise
            has_parents = parent_w.sum(dim=1) > 0  # (b,)

            if not has_parents.any():
                x[:, :, j] = noise
                continue

            # random edge weights, zeroed for non-edges
            w = torch.randn(batch_size, j, device=device) / (j ** 0.5) * parent_w

            # linear combination of parents
            signal = torch.einsum('bri,bi->br', x[:, :, :j], w)  # (b, n_rows)

            # apply per-dataset nonlinearity
            out = torch.zeros_like(signal)
            out[act_ids == 0] = torch.tanh(signal[act_ids == 0])
            out[act_ids == 1] = F.relu(signal[act_ids == 1])
            out[act_ids == 2] = signal[act_ids == 2]
            out[act_ids == 3] = torch.sin(signal[act_ids == 3])

            noise_scale = torch.empty(batch_size, 1, device=device).uniform_(0.1, 1.0)
            x_j = noise.clone()
            x_j[has_parents] = out[has_parents] + noise[has_parents] * noise_scale[has_parents]
            x[:, :, j] = x_j

        # random permutation of feature indices so DAG topo order != index order
        perm = torch.argsort(torch.rand(batch_size, n_features, device=device), dim=1)
        x = torch.gather(x, 2, perm.unsqueeze(1).expand(-1, n_rows, -1))

        return x

    # -------------------------------------------------------------------------
    # function types

    def _linear(self, x):
        """y = x @ w  (random weight vector, LogNormal scale)"""
        b, r, d = x.shape
        # weight scale ~ LogNormal(0, 1)
        w_scale = torch.randn(b, device=x.device).exp()  # (b,)
        w = torch.randn(b, d, 1, device=x.device) / (d ** 0.5) * w_scale.view(b, 1, 1)
        return torch.bmm(x, w).squeeze(-1)

    def _mlp(self, x):
        """
        Random MLP following TabPFN v1.
        - depth ~ Gamma (1-5 layers)
        - width ~ Gamma (4-256 hidden units)
        - weight scale ~ LogNormal
        - activation: {tanh, relu, gelu} sampled per dataset batch
        """
        b, r, d = x.shape
        device = x.device

        # sample shared architecture for this sub-batch (all datasets get same shape)
        # but each dataset gets its own random weights
        n_hidden = max(4, int(abs(torch.empty(1).normal_(64, 48).item())))
        n_hidden = min(n_hidden, 256)
        n_layers = max(1, int(abs(torch.empty(1).normal_(2, 1).item())))
        n_layers = min(n_layers, 5)
        act_idx = torch.randint(0, 3, (1,)).item()
        acts = [torch.tanh, F.relu, F.gelu]
        act = acts[act_idx]

        # weight scale ~ LogNormal(mean=0, std=1) per TabPFN v1
        w_scale = torch.randn(b, device=device).exp().view(b, 1, 1)

        h = x  # (b, r, d_in)
        in_dim = d
        for layer in range(n_layers):
            out_dim = n_hidden if layer < n_layers - 1 else 1
            W = torch.randn(b, in_dim, out_dim, device=device) / (in_dim ** 0.5) * w_scale
            b_bias = torch.zeros(b, 1, out_dim, device=device)
            h = torch.bmm(h, W) + b_bias
            if layer < n_layers - 1:
                h = act(h)
            in_dim = out_dim

        return h.squeeze(-1)  # (b, r)

    def _gp_rff(self, x):
        """
        GP via Random Fourier Features (Bochner's theorem for RBF kernel).
        lengthscale ~ LogNormal, outputscale ~ LogNormal  (TabPFN v1 style).
        """
        b, r, d = x.shape
        device = x.device
        n_basis = 128

        # lengthscale ~ LogNormal: log(ls) ~ N(0, 1)
        log_ls = torch.randn(b, device=device)
        ls = log_ls.exp().view(b, 1, 1)  # (b, 1, 1)

        # outputscale ~ LogNormal
        log_os = torch.randn(b, device=device)
        outputscale = log_os.exp().view(b, 1)  # (b, 1)

        # random frequencies omega ~ N(0, 1/ls^2)
        omega = torch.randn(b, d, n_basis, device=device) / ls  # (b, d, n_basis)

        # random phase offsets
        phi = torch.rand(b, 1, n_basis, device=device) * 2 * math.pi

        # RFF feature map: sqrt(2/n_basis) * cos(x @ omega + phi)
        z = torch.bmm(x, omega) + phi  # (b, r, n_basis)
        z = ((2.0 / n_basis) ** 0.5) * torch.cos(z)

        # random output weights ~ N(0, 1/n_basis)
        alpha = torch.randn(b, n_basis, 1, device=device) / (n_basis ** 0.5)

        y = torch.bmm(z, alpha).squeeze(-1) * outputscale  # (b, r)
        return y

    def _quadratic(self, x):
        """
        Quadratic model: y = diag(x @ A @ x^T) + x @ w
        Captures pairwise feature interactions (TabICL "quadratic" type).
        """
        b, r, d = x.shape
        device = x.device

        # random symmetric low-rank interaction matrix A = U U^T / rank
        rank = min(d, torch.randint(1, max(2, d // 2 + 1), (1,)).item())
        U = torch.randn(b, d, rank, device=device) / (d * rank) ** 0.5
        A = torch.bmm(U, U.transpose(1, 2))  # (b, d, d)

        # quadratic term: for each row, x_i @ A @ x_i^T (scalar)
        xA = torch.bmm(x, A)  # (b, r, d)
        quad = (xA * x).sum(dim=-1)  # (b, r)

        # linear term
        w = torch.randn(b, d, 1, device=device) / (d ** 0.5)
        lin = torch.bmm(x, w).squeeze(-1)  # (b, r)

        return quad + lin

    def _additive(self, x):
        """
        Additive model: y = sum_i f_i(x_i) where f_i is a random 1-D function.
        Each f_i is a weighted sum of polynomial and sinusoidal terms (TabICL additive).
        """
        b, r, d = x.shape
        device = x.device

        # random feature weights (sparse: some features matter more than others)
        feat_w = torch.randn(b, d, device=device)
        feat_w = feat_w / feat_w.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)

        y = torch.zeros(b, r, device=device)
        for fi in range(d):
            xi = x[:, :, fi]  # (b, r)

            # linear component
            y = y + feat_w[:, fi:fi+1] * xi

            # random nonlinear component: sin or polynomial (50/50 per feature)
            if torch.rand(1) > 0.5:
                # sinusoidal: sin(freq * x + phase)
                freq = torch.empty(b, 1, device=device).uniform_(0.3, 4.0)
                phase = torch.empty(b, 1, device=device).uniform_(0, 2 * math.pi)
                nl = torch.sin(freq * xi + phase)
            else:
                # cubic polynomial: a*x + b*x^2 + c*x^3
                a = torch.randn(b, 1, device=device) * 0.5
                b2 = torch.randn(b, 1, device=device) * 0.3
                c = torch.randn(b, 1, device=device) * 0.1
                nl = a * xi + b2 * xi ** 2 + c * xi ** 3

            y = y + feat_w[:, fi:fi+1] * nl

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
        eval_pos_min_frac: float = 0.33,  # min fraction of rows used as train context
        eval_pos_max_frac: float = 0.90,  # max fraction of rows used as train context
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

            # randomize single_eval_pos between [min_frac, max_frac] * n_rows
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
            y_np = y.cpu().float().numpy()  # (batch_size, num_rows)
            y_means = y_np.mean(axis=1, keepdims=True)
            y_stds = y_np.std(axis=1, ddof=1, keepdims=True) + 1e-8
            all_y.append(((y_np - y_means) / y_stds).ravel())

    ys = torch.tensor(np.concatenate(all_y), dtype=torch.float32, device=device)
    print(f"Using {ys.numel()} y samples to estimate {n_buckets} buckets.")
    # use uniform quantiles
    quantiles = torch.linspace(0, 1, n_buckets + 1, device=device)
    edges = torch.quantile(ys, quantiles)
    return edges
