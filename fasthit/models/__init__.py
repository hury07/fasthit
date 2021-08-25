### sklearn models
from fasthit.models.sklearn_models import SklearnModel
from fasthit.models.sklearn_models import (
    LinearRegression,
    RandomForestRegressor,
)
### pytorch models
from fasthit.models.torch_model import TorchModel
from fasthit.models.mlp import MLP
from fasthit.models.cnn import CNN
from fasthit.models.finetune_model import Finetune
from fasthit.models.finetune_model_half import FinetuneHalf
from fasthit.models.finetune_model_cat import FinetuneCat
### uncertainty models
from fasthit.models.ensemble import Ensemble
from fasthit.models.gpr_blo import GPRegressor
from fasthit.models.rio import RIO