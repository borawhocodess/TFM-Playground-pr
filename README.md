# TFM-Playground

The purpose of this repository is to provide a fully open source playground for tabular foundation models.
It contains much smaller and simpler implementations of several TFM architectures (nanoTabPFN, nanoTabICL, nanoTabDPT, moddedNanoTabPFN, nanoTabFM) as well as a training loop, multiple interfaces to load prior data and an evaluation pipeline. We are planning to rapidly extend the repository with more features, prior interfaces and architectures.
It is supposed to be a good starting point for students and researchers that are interested in learning about how Tabular foundation models work under the hood.

Clone the repository, afterwards install dependencies via:
```
pip install -e .
```

### Pretrain a TFM in one call

```python
from tfmplayground import pretrainTFM

model = pretrainTFM()
```

Everything is optional: with no arguments this downloads [100k pre-generated classification datasets](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_3_100k_classification.h5) with 50 datapoints and 3 features each on first use, builds a small nanoTabPFN sized to fit them, picks a loss criterion and pretrains on the best available device, logging the loss to the console. This should take a couple of minutes on a modern NVIDIA GPU (longer on a laptop).

The same goes for regression:

```python
model = pretrainTFM(problem="regression")
```

which instead downloads [1.28M pre-generated regression datasets](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_1280k_regression.h5) (~1GB) and fits a bar distribution over 100 buckets as the criterion.

The trained model plugs straight into our scikit-learn like interface:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tfmplayground import TabularClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = TabularClassifier(model)
clf.fit(X_train, y_train)

prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

### Swapping the parts out

Every part of the call can be replaced:

```python
from tfmplayground import pretrainTFM
from tfmplayground.evaluation import TOY_TASKS_CLASSIFICATION, OpenMLEvaluationCallback
from tfmplayground.external_priors import PriorDumpDataLoader
from tfmplayground.models import NanoTabPFNModel

model = pretrainTFM(
    model=NanoTabPFNModel(
        num_attention_heads=6,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=10,
    ),
    prior=PriorDumpDataLoader("50x3_3_100k_classification.h5", num_steps=25, batch_size=50),
    eval=OpenMLEvaluationCallback(TOY_TASKS_CLASSIFICATION),
    epochs=80,
)
```

`model` takes any architecture from `tfmplayground/models/`, they all share the same forward contract. `prior` takes any of our dataloaders, `eval` takes a callback or list of callbacks run at the end of each epoch. The loss criterion is inferred from the prior (cross entropy for classification, a bar distribution fitted on the dump for regression) unless you pass one via `criterion`.

For regression we offer a pre-generated dataset containing 1.28M tables with 50 datapoints and 3 features each [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_1280k_regression.h5).

### Our Code

`tfmplayground/models/` contains the architectures, each implemented in a single file. `tfmplayground/train.py` implements `pretrainTFM` and a simple training loop and `tfmplayground/external_priors/` provides an interface to publicly available priors from other repositories as well as a dataloader for loading HDF5 dumps.
We will release multiple dumps of different scales soon. We also offer an interface where you can provide your own get\_batch function.

### Creating your own datasets
Check out [tfmplayground.external_priors](https://github.com/automl/TFM-Playground/tree/main/tfmplayground/external_priors) to create your own data using publicly available priors.

You can use `tfmplayground.external_priors` as a command-line-tool to pre-generate data from a prior, e.g. via
```
python -m tfmplayground.external_priors --lib tabicl \
       --prior_type mix_scm \
       --num_batches 1000 --batch_size 4 \
       --min_features 3 --max_features 3 \
       --max_seq_len 50 --max_classes 3 \
       --save_path tabicl_4k_50x3.h5
```
which can afterwards be loaded via
```python
from tfmplayground.external_priors import PriorDumpDataLoader
prior = PriorDumpDataLoader('tabicl_4k_50x3.h5', num_steps=20, batch_size=4)
```
You can also just let it create the data on-the-fly via:
```python
from tfmplayground.external_priors import TabICLPriorDataLoader
prior = TabICLPriorDataLoader(
    num_steps=20,
    batch_size=4,
    num_datapoints_min=50,
    num_datapoints_max=50,
    min_features=3,
    max_features=3,
    max_num_classes=3,
)
```
You can check out `next(iter(prior))` if you want to see an example batch.

Check out `prior_visualization.ipynb` for some more examples.

### Supported Priors

- [TabICL](https://github.com/soda-inria/tabicl) (Classification)
- [TICL](https://github.com/microsoft/ticl) (Regression, Classification)
