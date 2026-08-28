# TFM-Playground

The purpose of this repository is to provide a fully open source playground for tabular foundation models.

```
todo: clone and uv instructions
```

### Pretrain a TFM 

```python
from tfmplayground import pretrainTFM

model = pretrainTFM(problem="classification")
# or = pretrainTFM(problem="regression")
```

it is really that easy!

later use it as:

```python
todo: eval example 
interface(model)
```

### Pretrain a TFM with specific configurations

```python
modelconfig = ...
priorconfig = ...
evalconfig = ...
trainconfig = ...

model = pretrainTFM(
    problem="classification",
    model=NanoTabPFNModel(config=modelconfig),
    prior=TabICLPrior(config=priorconfig, device=device),
    eval=evalconfig,
    training=trainconfig,
)
```

### model list

### prior list 

### eval stuff 

### train stuff 
