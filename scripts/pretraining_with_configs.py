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

model = pretrainTFM(
    problem="classification",
    model=ModdedNanoTabPFNModel(config=ModdedNanoTabPFNClassifierConfig(l=10, o=10)),
    prior=NanoTabICLPrior(config=NanoTabICLClassificationPriorConfig(), device=device),
    eval=EvaluationConfig(tasks="toy"),
    training=ClassificationTrainingConfig(seed=11, batch_size=4, epochs=10),
)

model = pretrainTFM(
    problem="regression",
    model=TabFMModel(config=TabFMRegressorConfig()),
    prior=TabICLPrior(config=TabICLRegressionPriorConfig(num_datapoints_max=256, num_features_max=4), device=device),
    eval=EvaluationConfig(tasks="tabarena", max_n_samples=800),
    training=RegressionTrainingConfig(seed=11, batch_size=4, epochs=10),
)
