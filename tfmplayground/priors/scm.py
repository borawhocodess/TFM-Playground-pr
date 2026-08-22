"""Our own prior, the tabpfnv2 structural causal model implemented from scratch."""

import numpy as np
import torch

from tfmplayground.priors.base import MAX_NUM_CLASSES, Batch, Prior
from tfmplayground.utils import get_default_device


class SCMPrior(Prior):
    """
    Our own structural causal model prior, sampled on the fly, nothing to download.

    Args:
        num_datapoints_max (int): max sequence length per table, at least 129.
        num_features (int): number of input features.
        problem (str): "classification" or "regression".
        device (torch.device): device the batches end up on, defaults to the best available one.
    """

    def __init__(
        self,
        num_datapoints_max: int = 160,
        num_features: int = 8,
        problem: str = "classification",
        device: torch.device = None,
    ):
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.device = device if device is not None else get_default_device()
        self.problem_type = problem
        self.max_num_classes = MAX_NUM_CLASSES if problem == "classification" else None

    def batch(self, batch_size: int) -> Batch:
        return get_batch(batch_size, self.num_datapoints_max, self.num_features, problem=self.problem_type)


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
            x = torch.stack(xs)
            y = torch.stack(ys)
            return x[:, :rows], y[:, :rows], x[:, rows:], y[:, rows:]


def postprocess(x, kinds):  # tabpfnv2 paper post-processing subsection
    x = x.clone()
    chosen = [np.random.rand() < 0.5, np.random.rand() < 0.5, False]  # NO RANGE, missing off until the model handles nan
    if not any(chosen):
        chosen[np.random.randint(2)] = True
    continuous = [j for j, kind in enumerate(kinds) if kind == "continuous"]
    for transform, on in zip([warp, quantize], chosen[:2], strict=True):
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
