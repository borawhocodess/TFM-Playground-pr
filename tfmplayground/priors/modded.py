# Vendored from https://github.com/borawhocodess/modded-nanotabpfn (train_prior.py),
# adapted to the Prior contract. Modified from the original.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Salih Bora Ozturk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.priors.base import Batch, Prior
from tfmplayground.utils import get_default_device

# ----- vendored, do not edit: keep it identical to upstream so updates can be pulled straight in


@dataclass
class PriorConfig:
    min_num_classes: int = 2
    max_num_classes: int = 8
    min_num_cols: int = 20
    max_num_cols: int = 20
    min_num_parent_attempts: int = 3
    max_num_parent_attempts: int = 3
    min_redirection: float = 0.5
    max_redirection: float = 0.5
    min_num_rows: int = 1000
    max_num_rows: int = 1000
    min_num_test_rows: int = 128
    max_num_test_rows: int = 128


class ModdedNanoPriorUpstream:
    activations = (lambda z: z, torch.tanh, torch.sin, torch.abs, torch.square, F.softplus)

    def __init__(self, config, device):
        self.config = config
        self.device = device
        assert self.config.max_num_test_rows < self.config.min_num_rows

    def hyperparameters(self):
        c = self.config
        self.num_cols = int(np.random.randint(c.min_num_cols, c.max_num_cols + 1))
        self.num_rows = int(np.random.randint(c.min_num_rows, c.max_num_rows + 1))
        self.num_test_rows = int(np.random.randint(c.min_num_test_rows, c.max_num_test_rows + 1))
        self.redirection = np.random.uniform(c.min_redirection, c.max_redirection)
        self.num_classes = int(np.random.randint(c.min_num_classes, c.max_num_classes + 1))
        self.num_parent_attempts = int(np.random.randint(c.min_num_parent_attempts, c.max_num_parent_attempts + 1))
        self.sep = self.num_rows - self.num_test_rows
        self.nodes = self.num_cols + 1

    def gnr(self):
        parents = [[] for _ in range(self.nodes)]
        for child in range(1, self.nodes):
            chosen = set()
            for _ in range(self.num_parent_attempts):
                candidate = int(np.random.randint(child))
                if np.random.rand() < self.redirection and parents[candidate]:
                    candidate = int(np.random.choice(parents[candidate]))
                chosen.add(candidate)
            parents[child] = sorted(chosen)
        return parents

    def propagate(self):
        parents = self.gnr()
        w = np.zeros((self.nodes, self.nodes), dtype=np.float32)
        for i in range(1, self.nodes):
            w[i, parents[i]] = np.random.randn(len(parents[i]))
        w = torch.from_numpy(w).to(self.device)
        acts = np.random.randint(len(self.activations), size=self.nodes)
        z = torch.randn(self.num_rows, self.nodes, device=self.device)
        for i in range(1, self.nodes):
            zi = self.activations[acts[i]](z @ w[i]) + 0.1 * z[:, i]
            std, mean = torch.std_mean(zi)
            z[:, i] = (zi - mean) / (std + 1e-6)
        return z

    def target(self, z):
        target = int(np.random.randint(1, self.nodes))
        zt = z[:, target].contiguous()
        cuts = torch.linspace(0, 1, self.num_classes + 1, device=self.device)[1:-1]
        y = torch.bucketize(zt, zt.quantile(cuts))
        x = torch.cat([z[:, :target], z[:, target + 1 :]], dim=1)
        return x, y.float()

    def postprocess(self, x):
        return x

    def dataset(self):
        z = self.propagate()
        x, y = self.target(z)
        x = self.postprocess(x)
        return x, y

    def batch(self, batch_size):
        self.hyperparameters()
        datasets = [self.dataset() for _ in range(batch_size)]
        x = torch.stack([d[0] for d in datasets])
        y = torch.stack([d[1] for d in datasets])
        sep = self.sep
        return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]


# ----- ours: the adapter onto the Prior contract


class ModdedNanoPrior(ModdedNanoPriorUpstream, Prior):
    """
    The prior the modded nanotabpfn speedrun trains on, sampled on the fly, nothing to download.

    A growing network with redirection gives every node its parents, a random activation maps each
    node's parent sum plus a little of its own noise, and one node is bucketized into classes to
    become the target. Classification only, the target is quantile cut by construction.

    The sampling above is upstream's, unedited, so a newer version can be pasted over it. This
    class only adds what the Prior contract needs: a device, the problem it sits on, and the class
    cap. Settings come from PriorConfig, the same dataclass upstream uses.

    Args:
        config (PriorConfig): the upstream settings, defaults to upstream's own defaults.
        device (torch.device): device the batches end up on, defaults to the best available one.
    """

    def __init__(self, config: PriorConfig = None, device: torch.device = None):
        config = config if config is not None else PriorConfig()
        device = device if device is not None else get_default_device()
        super().__init__(config, device)
        self.problem_type = "classification"
        self.max_num_classes = config.max_num_classes
        self.num_features = config.max_num_cols
        self.num_datapoints_max = config.max_num_rows

    def batch(self, batch_size: int) -> Batch:
        return super().batch(batch_size)
