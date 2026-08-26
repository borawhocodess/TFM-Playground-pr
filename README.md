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

### Pretrain a TFM with spesific configurations

```python
modelconfig = ...
priorconfig = ...
evalconfig = ...
trainconfig = ...

model = pretrainTFM(
    model=NanoTabPFNModel(config=modelconfig),
    prior=Prior(config=priorconfig),
    eval=tabarenasubsampled(config=evalconfig),
    training=trainer(config=trainconfig),
)
```

### model list

### prior list 

### eval stuff 

### train stuff 
