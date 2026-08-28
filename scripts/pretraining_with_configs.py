from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import ModdedNanoTabPFNClassifierConfig, TabFMRegressorConfig
from tfmplayground.configs.priors import NanoTabICLClassificationPriorConfig, TabICLRegressionPriorConfig
from tfmplayground.configs.training import ClassificationTrainingConfig, RegressionTrainingConfig
from tfmplayground.models.moddednanotabpfn import ModdedNanoTabPFNModel
from tfmplayground.models.tabfm import TabFMModel
from tfmplayground.priors import NanoTabICLPrior, TabICLPrior
from tfmplayground.training.pretrain import pretrainTFM
from tfmplayground.utils import get_default_device

device = get_default_device()

modelconfig = ModdedNanoTabPFNClassifierConfig(l=10, o=10)
priorconfig = NanoTabICLClassificationPriorConfig()
evalconfig = EvaluationConfig(tasks="toy")
trainconfig = ClassificationTrainingConfig(seed=11, batch_size=4, epochs=10)

model = pretrainTFM(
    problem="classification",
    model=ModdedNanoTabPFNModel(config=modelconfig),
    prior=NanoTabICLPrior(config=priorconfig, device=device),
    eval=evalconfig,
    training=trainconfig,
)

modelconfig = TabFMRegressorConfig()
priorconfig = TabICLRegressionPriorConfig(num_datapoints_max=256, num_features_max=4)
evalconfig = EvaluationConfig(tasks="tabarena", max_n_samples=800)
trainconfig = RegressionTrainingConfig(seed=11, batch_size=4, epochs=10)

model = pretrainTFM(
    problem="regression",
    model=TabFMModel(config=modelconfig),
    prior=TabICLPrior(config=priorconfig, device=device),
    eval=evalconfig,
    training=trainconfig,
)
