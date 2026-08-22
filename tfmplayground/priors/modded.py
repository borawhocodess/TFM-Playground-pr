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

import numpy as np
import torch
import torch.nn.functional as F

from tfmplayground.priors.base import Batch, Prior
from tfmplayground.utils import get_default_device


class ModdedNanoPrior(Prior):
    """
    The prior the modded nanotabpfn speedrun trains on, sampled on the fly, nothing to download.

    A growing network with redirection gives every node its parents, a random activation maps
    each node's parent sum plus a little of its own noise, and one node is bucketized into
    classes to become the target. Classification only, the target is quantile cut by construction.

    Upstream samples every setting from a min/max range, but ships with all of them collapsed
    except the class count, so only that one is a range here.

    Args:
        num_datapoints_max (int): rows per table, train and test together.
        num_features (int): number of input features.
        num_test_datapoints (int): rows held out per table, must be fewer than num_datapoints_max.
        min_num_classes (int): fewest classes the target is cut into.
        max_num_classes (int): most classes the target is cut into.
        redirection (float): chance a parent is redirected onto a grandparent, the r of the growing network.
        parent_attempts (int): draws per node, deduplicated, so a node can end up with fewer parents.
        device (torch.device): device the batches end up on, defaults to the best available one.
    """

    activations = (lambda z: z, torch.tanh, torch.sin, torch.abs, torch.square, F.softplus)

    def __init__(
        self,
        num_datapoints_max: int = 1000,
        num_features: int = 20,
        num_test_datapoints: int = 128,
        min_num_classes: int = 2,
        max_num_classes: int = 8,
        redirection: float = 0.5,
        parent_attempts: int = 3,
        device: torch.device = None,
    ):
        if num_test_datapoints >= num_datapoints_max:
            raise ValueError(
                f"num_test_datapoints must be smaller than num_datapoints_max, "
                f"got {num_test_datapoints} and {num_datapoints_max}"
            )
        if min_num_classes < 2 or max_num_classes < min_num_classes:
            raise ValueError(
                f"need 2 <= min_num_classes <= max_num_classes, got {min_num_classes} and {max_num_classes}"
            )
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.num_test_datapoints = num_test_datapoints
        self.min_num_classes = min_num_classes
        self.redirection = redirection
        self.parent_attempts = parent_attempts
        self.device = device if device is not None else get_default_device()
        self.problem_type = "classification"
        self.max_num_classes = max_num_classes
        self.nodes = num_features + 1  # one node is spent on the target
        self.sep = num_datapoints_max - num_test_datapoints

    def gnr(self) -> list[list[int]]:
        parents = [[] for _ in range(self.nodes)]
        for child in range(1, self.nodes):
            chosen = set()
            for _ in range(self.parent_attempts):
                candidate = int(np.random.randint(child))
                if np.random.rand() < self.redirection and parents[candidate]:
                    candidate = int(np.random.choice(parents[candidate]))
                chosen.add(candidate)
            parents[child] = sorted(chosen)
        return parents

    def propagate(self) -> torch.Tensor:
        parents = self.gnr()
        weights = np.zeros((self.nodes, self.nodes), dtype=np.float32)
        for i in range(1, self.nodes):
            weights[i, parents[i]] = np.random.randn(len(parents[i]))
        weights = torch.from_numpy(weights).to(self.device)
        acts = np.random.randint(len(self.activations), size=self.nodes)
        z = torch.randn(self.num_datapoints_max, self.nodes, device=self.device)
        for i in range(1, self.nodes):
            zi = self.activations[acts[i]](z @ weights[i]) + 0.1 * z[:, i]
            std, mean = torch.std_mean(zi)
            z[:, i] = (zi - mean) / (std + 1e-6)
        return z

    def target(self, z: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = int(np.random.randint(1, self.nodes))
        zt = z[:, target].contiguous()
        cuts = torch.linspace(0, 1, num_classes + 1, device=self.device)[1:-1]
        y = torch.bucketize(zt, zt.quantile(cuts))
        x = torch.cat([z[:, :target], z[:, target + 1 :]], dim=1)
        return x, y.float()

    def table(self, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.target(self.propagate(), num_classes)

    def batch(self, batch_size: int) -> Batch:
        num_classes = int(np.random.randint(self.min_num_classes, self.max_num_classes + 1))
        tables = [self.table(num_classes) for _ in range(batch_size)]
        x = torch.stack([table[0] for table in tables])
        y = torch.stack([table[1] for table in tables])
        return x[:, : self.sep], y[:, : self.sep], x[:, self.sep :], y[:, self.sep :]
