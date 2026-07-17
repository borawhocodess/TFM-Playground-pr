# sbopriornotes

source:
Hollmann et al.,
"Accurate predictions on small data with a tabular foundation model",
Nature 637 (2025),
doi:10.1038/s41586-024-08328-6

extracted via pdftotext from `~/repos/fm4sd/papers/tabpfnv2/tabpfnv2.pdf`

## Details on the causal generative process

An SCM G ≔ (Z, ϵ) consists of a collection Z ≔ (z1, …, zk) of structural assignments (called mechanisms): zi = fi(zPAG(i), ϵi),
where PAG(i) is the set of parents of node i (its direct causes) in the underlying directed acyclic graph (DAG) G (the causal graph),
fi is a (potentially nonlinear) deterministic function and ϵi is a noise variable.

Causal relationships in G are represented by edges pointing from causes to effects.

As our prior is a sampling procedure, we can make a lot of choices on, for example, the graph size or complexity.

By defining a probability distribution over these hyperparameters in the prior,
the posterior predictive distribution approximated by TabPFN at inference time implicitly represents a Bayesian ensemble,
jointly integrating over a weighted hyperparameter space.

The specific hyperparameter ranges and sampling strategies are chosen to cover a diverse set of scenarios that we expect to encounter in real-world tabular data.

**Graph structure sampling.**

The structural causal models underlying each dataset are based on a DAG G.

We sample these graphs using the growing network with redirection sampling method,
a preferential attachment process that generates random scale-free networks.

We either sample a single connected component or merge multiple disjoint subgraphs.

Disjoint subgraphs lead to features that are marginally independent of the target if they are not connected to the target node,
reflecting real-world scenarios with uninformative predictors.

To control the complexity of the sampled DAGs, we use two hyperparameters: the number of nodes N and the redirection probability P.

N is sampled from a log-uniform distribution, logN ~ U(a, b), where a and b are hyperparameters controlling the range of the graph size.

The redirection probability P is sampled from a gamma distribution, P ~ Γ(α, β), where α and β are shape and rate parameters, respectively.

Larger values of N yield graphs with more nodes, whereas smaller values of P lead to denser graphs with more edges on average.

**Computational edge mappings.**

In our implementation, each SCM node and sample is represented as a vector in Rd.

When propagating data through the SCM,
the deterministic functions fi at each edge map the input vectors to an output vector using four types of computational modules:

1. Small neural networks:
here we initialize weight matrices W ∈ Rd×d using Xavier initialization and
apply a linear transformation Wx + b to the input vectors x ∈ Rd, where b ∈ Rd is a bias vector.

After the linear projection,
we apply element-wise nonlinear activation functions σ : Rd → Rd, randomly sampled from a set,
including identity, logarithm, sigmoid, absolute value, sine, hyperbolic tangent,
rank operation, squaring, power functions, smooth ReLU, step function and modulo operation.

2. Categorical feature discretization:
to generate categorical features from the numerical vectors at each node,
we map the vector to the index of the nearest neighbour in a set of per node randomly sampled vectors {p1, …, pK} for a feature with K categories.

This discrete index will be observed in the feature set as a categorical feature.

We sample the number of categories K from a rounded gamma distribution with an offset of 2 to yield a minimum number of classes of 2.

To further use these discrete class assignments in the computational graph, they need to be embedded as continuous values.

We sample a second set of embedding vectors {p1′, …, pK′} for each class and transform the classes to these embeddings.

3. Decision trees: to incorporate structured, rule-based dependencies, we implement decision trees in the SCMs.
At certain edges, we select a subset of features and apply decision boundaries on their values to determine the output.
The decision tree parameters (feature splits, thresholds) are randomly sampled per edge.

4. Noise injection: at each edge, we add random normal noise from the normal distribution N(0, σ²I).

**Initialization data sampling.**

For each to-be-generated sample,
we randomly generate initialization data ϵ that is inserted at the DAG root nodes and then propagated through the computational graph.

The noise variables ϵ are generated according to one of three sampling mechanisms:

1. Normal: ϵ ~ N(0, σϵ²), where σϵ² is a hyperparameter.
2. Uniform: ϵ ~ U(−a, a), where a is a hyperparameter.
3. Mixed: for each root node, we randomly select either a normal or uniform distribution to sample the initialization noise ϵ from.

Furthermore, we sample input data with varying degrees of non-independence for some datasets.

Here we first sample a random fraction ρ of samples to serve as prototypes x1*, …, xM*, where M = ρn and n is the dataset size.

Then, for each input vector xi to be sampled,
we assign weights αij to the prototypes and linearly mix the final input as xi = Σ(j=1..M) αij xj*, (1) where Σj αij = 1.

The weights αij are sampled from a multinomial distribution,
αi ~ Multinomial(β), where β is a temperature hyperparameter controlling the degree of non-independence:
larger β yields more uniform weights, whereas smaller β concentrates the weights on fewer prototypes per sample.

**Post-processing.**

Each dataset is post-processed randomly with one or more of the following post-processings:

(1) For some datasets, we use the Kumaraswamy feature warping, introducing nonlinear distortions to features as done in ref. 61.

(2) We quantize some continuous features into buckets of randomly sampled cardinality K,
mimicking binned or discretized features commonly encountered in datasets.

We map a feature value x to the index of the bucket it falls into,
determined by K + 1 bin edges sampled from the set of values this feature takes.

(3) To introduce scenarios for dynamic imputation and handling of incomplete datasets, a common challenge in data science,
we randomly designate a fraction ρmiss of the data as missing according to the missing completely at random strategy.

Each value is masked as missing with probability ρmiss, independently of the data values.

**Target generation.**

To generate target labels for regression tasks, we select a randomly chosen continuous feature without post-processing.

For classification labels, we select a random categorical feature that contains up to 10 classes.

Thus, natively our method is limited to predicting at most 10 classes.

This number can be increased by pre-training on datasets with a larger number of classes or by using approaches such as
building a one-vs-one classifier, one-vs-rest classifier or building on approaches such as error-correcting output codes (ECOC).

## from "Training details" (the rows/features/cells sentences our step 0 also leans on)

We trained our final models for approximately 2,000,000 steps with a batch size of 64 datasets.

That means the models used for TabPFN are trained on around 130,000,000 synthetically generated datasets each.

One training run requires around 2 weeks on one node with eight Nvidia RTX 2080 Ti GPUs.

We sample the number of training samples for each dataset uniformly up to 2,048 and use a fixed validation set size of 128.

We sample the number of features using a beta distribution (k = 0.95, b = 8.0) that we linearly scale to the range 1–160.

To avoid peaks in memory usage,
the total size of each table was restricted to be below 75,000 cells by decreasing the number of samples for large numbers of features.
