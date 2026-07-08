from types import SimpleNamespace

from tfmplayground import pretrainTFM
from tfmplayground.evaluation import TOY_TASKS_REGRESSION, OpenMLEvaluationCallback
from tfmplayground.external_priors import PriorDumpDataLoader
from tfmplayground.models import NanoTabPFNModel
from tfmplayground.utils import get_default_device, load_config, set_randomness_seed

args = SimpleNamespace(**load_config("nanotabpfn_regressor"))

set_randomness_seed(2402)

device = get_default_device()

prior = PriorDumpDataLoader(
    filename=args.priordump,
    num_steps=args.steps,
    batch_size=args.batchsize,
    device=device,
)

model = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=args.n_buckets,
)

trained_model = pretrainTFM(
    model=model,
    prior=prior,
    eval=OpenMLEvaluationCallback(TOY_TASKS_REGRESSION, classification=False, device=device),
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
)
