# %% imports
from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import ModdedNanoTabPFNClassifierConfig, TabFMRegressorConfig
from tfmplayground.configs.priors import NanoTabICLClassificationPriorConfig, TabICLRegressionPriorConfig
from tfmplayground.configs.training import (
    ClassificationExperimentConfig,
    ClassificationTrainingConfig,
    RegressionExperimentConfig,
    RegressionTrainingConfig,
)
from tfmplayground.models.moddednanotabpfn import ModdedNanoTabPFNModel
from tfmplayground.models.tabfm import TabFMModel
from tfmplayground.priors import NanoTabICLPrior, TabICLPrior
from tfmplayground.training.pretrain import pretrainTFM
from tfmplayground.utils import get_default_device

device = get_default_device()

# %% classification example

modelconfig = ModdedNanoTabPFNClassifierConfig(l=10, o=10)
priorconfig = NanoTabICLClassificationPriorConfig()
evalconfig = EvaluationConfig(tasks="toy")
trainconfig = ClassificationTrainingConfig(seed=11, batch_size=4, epochs=10)
experimentconfig = ClassificationExperimentConfig(name="example")

model = pretrainTFM(
    problem="classification",
    model=ModdedNanoTabPFNModel(config=modelconfig),
    prior=NanoTabICLPrior(config=priorconfig, device=device),
    eval=evalconfig,
    training=trainconfig,
    experiment=experimentconfig,
)

# %% regression example

modelconfig = TabFMRegressorConfig()
priorconfig = TabICLRegressionPriorConfig(num_datapoints_max=256, num_features_max=4)
evalconfig = EvaluationConfig(tasks="tabarena", max_n_samples=800)
trainconfig = RegressionTrainingConfig(seed=11, batch_size=4, epochs=10)
experimentconfig = RegressionExperimentConfig(name="example")

model = pretrainTFM(
    problem="regression",
    model=TabFMModel(config=modelconfig),
    prior=TabICLPrior(config=priorconfig, device=device),
    eval=evalconfig,
    training=trainconfig,
    experiment=experimentconfig,
)
