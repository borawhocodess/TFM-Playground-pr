"""Data loading utilities for tabular priors."""

from collections.abc import Callable, Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tfmplayground.utils import get_default_device

MAX_NUM_CLASSES = 10  # tabpfnv2 paper target generation subsection, natively limited to at most 10 classes


class PriorDataLoader(DataLoader):
    """Generic DataLoader for synthetic data generation using a get_batch function.

    Args:
        get_batch_function (Callable): A function returning batches of data.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_max (int): Max sequence length per function.
        num_features (int): Number of input features.
        device (torch.device): Device to move tensors to, defaults to the best available device.
        problem (str): "classification" or "regression", forwarded to the get_batch function
            as a keyword argument and reported as problem_type. Left out when None.
        max_num_classes (int): Highest number of classes the get_batch function can produce.
    """

    def __init__(
        self,
        get_batch_function: Callable[..., dict[str, torch.Tensor | int]],
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        device: torch.device = None,
        problem: str | None = None,
        max_num_classes: int | None = None,
    ):
        self.get_batch_function = get_batch_function
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.device = device if device is not None else get_default_device()
        self.problem_type = problem
        self.max_num_classes = max_num_classes

    def get_batch(self) -> dict[str, torch.Tensor | int]:
        problem = {} if self.problem_type is None else {"problem": self.problem_type}
        return self.get_batch_function(self.batch_size, self.num_datapoints_max, self.num_features, **problem)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int]]:
        return iter(self.get_batch() for _ in range(self.num_steps))

    def __len__(self) -> int:
        return self.num_steps


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto, defaults to the best available device.
    """

    def __init__(self, filename, num_steps, batch_size, device=None, starting_index=0):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r") as f:
            self.num_datapoints_max = f["X"].shape[0]
            if "max_num_classes" in f:
                self.max_num_classes = f["max_num_classes"][0]
            else:
                self.max_num_classes = None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            self.has_num_datapoints = "num_datapoints" in f
            self.stored_max_seq_len = f["X"].shape[1]
        self.device = device if device is not None else get_default_device()
        self.pointer = starting_index

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size

                num_features = f["num_features"][self.pointer : end].max()
                if self.has_num_datapoints:
                    num_datapoints_batch = f["num_datapoints"][self.pointer : end]
                    max_seq_in_batch = int(num_datapoints_batch.max())
                else:
                    max_seq_in_batch = int(self.stored_max_seq_len)

                x = torch.from_numpy(f["X"][self.pointer : end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer : end, :max_seq_in_batch])
                if "train_test_split_index" in f:
                    train_test_split_index = f["train_test_split_index"][self.pointer : end]
                else:
                    train_test_split_index = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print(
                        """Finished iteration over all stored datasets! """
                        """Will start reusing the same data with different splits now."""
                    )
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
                    train_test_split_index=train_test_split_index[0].item(),
                )

    def __len__(self):
        return self.num_steps


def dump_prior_to_h5(
    prior, max_classes: int, batch_size: int, save_path: str, problem_type: str, max_seq_len: int, max_features: int
):
    """Dumps synthetic prior data into an HDF5 file for later training."""

    with h5py.File(save_path, "w") as f:
        dump_X = f.create_dataset(
            "X",
            shape=(0, max_seq_len, max_features),
            maxshape=(None, max_seq_len, max_features),
            chunks=(batch_size, max_seq_len, max_features),
            compression="lzf",
        )
        dump_num_features = f.create_dataset(
            "num_features", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )
        dump_num_datapoints = f.create_dataset(
            "num_datapoints", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )
        dump_y = f.create_dataset(
            "y", shape=(0, max_seq_len), maxshape=(None, max_seq_len), chunks=(batch_size, max_seq_len)
        )
        dump_train_test_split_index = f.create_dataset(
            "train_test_split_index", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )

        if problem_type == "classification":
            f.create_dataset("max_num_classes", data=np.array((max_classes,)), chunks=(1,))
        f.create_dataset("original_batch_size", data=np.array((batch_size,)), chunks=(1,))
        f.create_dataset("problem_type", data=problem_type, dtype=h5py.string_dtype())

        for e in tqdm(prior):
            x = e["x"].to("cpu").numpy()
            y = e["y"].to("cpu").numpy()
            train_test_split_index = e["train_test_split_index"]
            if isinstance(train_test_split_index, torch.Tensor):
                train_test_split_index = train_test_split_index.item()

            # pad x and y to the maximum sequence length and number of features needed for tabicl
            x_padded = np.pad(
                x, ((0, 0), (0, max_seq_len - x.shape[1]), (0, max_features - x.shape[2])), mode="constant"
            )
            y_padded = np.pad(y, ((0, 0), (0, max_seq_len - y.shape[1])), mode="constant")

            dump_X.resize(dump_X.shape[0] + batch_size, axis=0)
            dump_X[-batch_size:] = x_padded

            dump_y.resize(dump_y.shape[0] + batch_size, axis=0)
            dump_y[-batch_size:] = y_padded

            dump_num_features.resize(dump_num_features.shape[0] + batch_size, axis=0)
            dump_num_features[-batch_size:] = x.shape[2]

            dump_num_datapoints.resize(dump_num_datapoints.shape[0] + batch_size, axis=0)
            dump_num_datapoints[-batch_size:] = x.shape[1]

            dump_train_test_split_index.resize(dump_train_test_split_index.shape[0] + batch_size, axis=0)
            dump_train_test_split_index[-batch_size:] = train_test_split_index


def hyperparameters():
    features = int(round(1 + np.random.beta(0.95, 8.0) * 159))  # tabpfnv2 paper training details section
    rows = int(np.random.randint(2, 2049))  # tabpfnv2 paper training details section, NO RANGE floor
    rows = min(rows, 75000 // features - 128)  # tabpfnv2 paper training details section
    nodes = int(round(np.exp(np.random.uniform(np.log(4), np.log(32)))))  # tabpfnv2 paper graph structure sampling subsection, NO RANGE
    redirection = min(np.random.gamma(4.0, 1 / 8), 1.0)  # tabpfnv2 paper graph structure sampling subsection, NO RANGE
    subgraphs = int(min(np.random.geometric(0.7), 4))  # tabpfnv2 paper graph structure sampling subsection, NO RANGE
    mechanism = str(np.random.choice(["normal", "uniform", "mixed"]))  # tabpfnv2 paper initialization data sampling subsection, equal choice like tabicl sampling
    scale = float(np.exp(np.random.uniform(np.log(0.01), np.log(10))))  # tabpfnv2 paper initialization data sampling subsection, NO RANGE, taken from tabicl init_std
    sigma = float(np.exp(np.random.uniform(np.log(1e-4), np.log(0.3))))  # tabpfnv2 paper computational edge mappings subsection, NO RANGE, taken from tabpfnv1 gaussian noise std
    prototypes = float(np.random.uniform(0.1, 0.9)) if np.random.rand() < 0.5 else 0.0  # tabpfnv2 paper initialization data sampling subsection, NO RANGE
    temperature = float(np.exp(np.random.uniform(np.log(0.05), np.log(5.0))))  # tabpfnv2 paper initialization data sampling subsection, NO RANGE
    dimension = int(round(np.exp(np.random.uniform(np.log(2), np.log(16)))))  # tabpfnv2 paper computational edge mappings subsection, NO RANGE
    dimension = max(dimension, -(-(features + 1) // nodes))  # not in tabpfnv2 paper, width grows with demand like nanotabicl
    categoricals = float(np.clip(np.random.uniform(-0.5, 1.2), 0.0, 1.0))  # NO RANGE, clipped uniform like nanotabicl
    return {
        "features": features,
        "rows": rows,
        "nodes": nodes,
        "redirection": redirection,
        "subgraphs": subgraphs,
        "mechanism": mechanism,
        "scale": scale,
        "sigma": sigma,
        "prototypes": prototypes,
        "temperature": temperature,
        "dimension": dimension,
        "categoricals": categoricals,
    }


def gnr(nodes, redirection, attempts=3):  # tabpfnv2 paper graph structure sampling subsection, NO RANGE for attempts
    parents = [[] for _ in range(nodes)]
    for child in range(1, nodes):
        chosen = set()
        for _ in range(attempts):
            target = int(np.random.randint(child))
            if np.random.rand() < redirection and parents[target]:
                target = int(np.random.choice(parents[target]))
            chosen.add(target)
        parents[child] = sorted(chosen)
    return parents


activations = [  # tabpfnv2 paper computational edge mappings subsection
    lambda x: x,  # identity
    lambda x: torch.log(torch.clamp(torch.abs(x), min=1e-6)),  # logarithm
    torch.sigmoid,  # sigmoid
    torch.abs,  # absolute value
    torch.sin,  # sine
    torch.tanh,  # hyperbolic tangent
    lambda x: torch.argsort(torch.argsort(x, dim=-1), dim=-1).float(),  # rank operation
    torch.square,  # squaring
    lambda x: torch.sign(x) * torch.abs(x) ** float(np.exp(np.random.uniform(np.log(0.1), np.log(10)))),  # power functions, NO RANGE, taken from nanotabicl
    torch.nn.functional.softplus,  # smooth relu
    lambda x: (x > 0).float(),  # step function
    lambda x: x - torch.floor(x),  # modulo operation, NO RANGE
]


def neural(x):  # tabpfnv2 paper computational edge mappings subsection
    d = x.shape[-1]
    weight = torch.empty(d, d)
    torch.nn.init.xavier_uniform_(weight)
    bias = torch.randn(d)  # tabpfnv2 paper computational edge mappings subsection, NO RANGE
    return activations[np.random.randint(len(activations))](x @ weight.T + bias)


def init(n, d, mechanism, scale):  # tabpfnv2 paper initialization data sampling subsection
    if mechanism == "mixed":
        mechanism = str(np.random.choice(["normal", "uniform"]))
    if mechanism == "normal":
        return scale * torch.randn(n, d)
    return scale * (2 * torch.rand(n, d) - 1)


def mix(x, prototypes, temperature):  # tabpfnv2 paper initialization data sampling subsection
    if prototypes == 0.0:
        return x
    m = max(int(round(prototypes * len(x))), 1)
    chosen = x[torch.randperm(len(x))[:m]]
    weights = torch.from_numpy(np.random.dirichlet(np.full(m, temperature), size=len(x))).float()
    return weights @ chosen


def categorical(x):  # tabpfnv2 paper computational edge mappings subsection
    categories = min(int(round(np.random.gamma(2.0, 1.0))) + 2, len(x))  # NO RANGE for gamma
    prototypes = x[torch.randperm(len(x))[:categories]]  # NO RANGE, prototypes are random datapoints like nanotabicl
    classes = torch.cdist(x, prototypes).argmin(dim=-1)
    embeddings = torch.randn(categories, x.shape[-1])  # NO RANGE
    return classes, embeddings[classes]


def tree(x):  # tabpfnv2 paper computational edge mappings subsection
    depth = int(np.random.randint(1, 5))  # NO RANGE
    dims = torch.randint(x.shape[-1], (depth,))
    thresholds = x[torch.randint(len(x), (depth,)), dims]
    sides = (x[:, dims] > thresholds).long()
    leaves = (sides * (2 ** torch.arange(depth))).sum(-1)
    values = torch.randn(2 ** depth, x.shape[-1])  # NO RANGE
    return values[leaves]


def noise(x, sigma):  # tabpfnv2 paper computational edge mappings subsection
    edge = abs(float(np.random.normal(0.0, sigma)))  # per node std half normal scaled by dataset sigma like tabpfn_prior
    return x + edge * torch.randn_like(x)


def graph(nodes, redirection, subgraphs):  # tabpfnv2 paper graph structure sampling subsection
    parents = []
    for part in np.array_split(np.arange(nodes), subgraphs):
        offset = len(parents)
        parents.extend([[parent + offset for parent in ps] for ps in gnr(len(part), redirection)])
    return parents


def propagate(parents, n, dimension, mechanism, scale, sigma, prototypes, temperature):  # tabpfnv2 paper details on the causal generative process section
    values = []
    for ps in parents:
        if not ps:
            values.append(mix(init(n, dimension, mechanism, scale), prototypes, temperature))
            continue
        edges = []
        for p in ps:
            module = str(np.random.choice(["neural", "tree", "categorical"], p=[0.7, 0.15, 0.15]))  # NO RANGE, neural vs rest like tabicl mix_probs
            if module == "neural":
                out = neural(values[p])
            elif module == "tree":
                out = tree(values[p])
            else:
                out = categorical(values[p])[1]
            edges.append(noise(out, sigma))
        value = torch.stack(edges).mean(0)
        values.append((value - value.mean(0)) / (value.std(0) + 1e-6))  # not in tabpfnv2 paper, stabilisation like nanotabicl
    return values


def observe(values, features, categoricals):  # tabpfnv2 paper details on the causal generative process section
    slots = [(node, column) for node in range(len(values)) for column in range(values[node].shape[-1])]
    order = [slots[i] for i in np.random.permutation(len(slots))]
    columns = []
    kinds = []
    used = 0
    for _ in range(features):
        if np.random.rand() < categoricals:
            columns.append(categorical(values[np.random.randint(len(values))])[0].float())
            kinds.append("categorical")
        else:
            node, column = order[used % len(order)]
            used += 1
            columns.append(values[node][:, column])
            kinds.append("continuous")
    return torch.stack(columns, -1), kinds


def warp(x):  # tabpfnv2 paper post-processing subsection
    a = float(np.exp(np.random.uniform(np.log(0.2), np.log(5.0))))  # NO RANGE, taken from nanotabicl
    b = float(np.exp(np.random.uniform(np.log(0.2), np.log(5.0))))  # NO RANGE, taken from nanotabicl
    low = x.min(dim=0).values
    high = x.max(dim=0).values
    x = torch.clamp((x - low) / (high - low + 1e-30), 0.0, 1.0)
    return 1.0 - (1.0 - x**a) ** b


def quantize(x):  # tabpfnv2 paper post-processing subsection
    buckets = int(round(np.random.gamma(2.0, 1.0))) + 2  # NO RANGE for gamma
    columns = []
    for j in range(x.shape[-1]):
        edges = x[torch.randint(len(x), (buckets + 1,)), j].sort().values
        columns.append(torch.bucketize(x[:, j].contiguous(), edges[1:-1]).float())
    return torch.stack(columns, -1)


def missing(x):  # tabpfnv2 paper post-processing subsection
    rho = float(np.random.uniform(0.0, 0.3))  # NO RANGE
    return torch.where(torch.rand_like(x) < rho, torch.nan, x)


def get_batch(batch_size, num_datapoints_max, num_features, problem="classification"):
    assert num_datapoints_max >= 129, f"num_datapoints_max must be at least 129 to fit the fixed 128 validation rows plus one training row, got {num_datapoints_max}"
    while True:
        h = hyperparameters()
        if h["categoricals"] == (0.0 if problem == "classification" else 1.0):
            continue
        rows = min(h["rows"], num_datapoints_max - 128)
        features = min(h["features"], num_features)
        xs = []
        ys = []
        for _ in range(batch_size):
            for _ in range(10):  # NO RANGE
                g = graph(h["nodes"], h["redirection"], h["subgraphs"])
                values = propagate(g, rows + 128, h["dimension"], h["mechanism"], h["scale"], h["sigma"], h["prototypes"], h["temperature"])
                observed, kinds = observe(values, features + 1, h["categoricals"])
                x, y, kept = target(observed, kinds, problem)
                if y is not None:
                    xs.append(postprocess(x, kept))
                    ys.append(y)
                    break
            else:
                break
        if len(xs) == batch_size:
            return dict(x=torch.stack(xs), y=torch.stack(ys), target_y=torch.stack(ys), train_test_split_index=rows)


def postprocess(x, kinds):  # tabpfnv2 paper post-processing subsection
    x = x.clone()
    chosen = [np.random.rand() < 0.5, np.random.rand() < 0.5, False]  # NO RANGE, missing off until the model handles nan
    if not any(chosen):
        chosen[np.random.randint(2)] = True
    continuous = [j for j, kind in enumerate(kinds) if kind == "continuous"]
    for transform, on in zip([warp, quantize], chosen[:2]):
        for j in continuous:
            if on and np.random.rand() < 0.3:  # NO RANGE
                x[:, j : j + 1] = transform(x[:, j : j + 1])
    if chosen[2]:
        x = missing(x)
    return x


def target(x, kinds, problem):  # tabpfnv2 paper target generation subsection
    if problem == "regression":
        candidates = [j for j, kind in enumerate(kinds) if kind == "continuous"]
    else:
        candidates = [
            j for j, kind in enumerate(kinds) if kind == "categorical" and len(x[:, j].unique()) <= MAX_NUM_CLASSES
        ]
    if not candidates:
        return None, None, None
    j = int(np.random.choice(candidates))
    y = x[:, j] if problem == "regression" else x[:, j].unique(return_inverse=True)[1].float()
    return torch.cat([x[:, :j], x[:, j + 1 :]], -1), y, kinds[:j] + kinds[j + 1 :]
