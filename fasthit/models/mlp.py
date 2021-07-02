"""Define a MLP Model."""
from typing import Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .torch_model import TorchModel
from skorch.callbacks import EarlyStopping


class MLP(TorchModel):
    def __init__(
        self,
        seq_length: int,
        input_features: int,
        n_layers: int,
        hidden_size: Union[Sequence[int], int], # defaults: 64, 128
        name: str = None,
        nogpu: bool = True,
        **new_fit_params,
    ):
        """Create the MLPModel."""
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size for _ in range(n_layers)]
        assert len(hidden_size) == n_layers

        model = MLPModel(
            input_features * seq_length,
            hidden_size,
        )
        ### set fit parameters
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu'
        )
        fit_params = {
            "max_epochs": 20, # default:20
            "batch_size": 32, # default:32
            "lr": 1e-3,
            "criterion": nn.MSELoss,
            "optimizer": optim.Adam,
            "warm_start": False,
            "device": device,
            "callbacks": [EarlyStopping().initialize()],
            #"train_split": None, # default: 5-fold cv
        }
        fit_params.update(new_fit_params)

        if name is None:
            name = f"MLP_hidden_size_{hidden_size}"

        super().__init__(
            name,
            model,
            **fit_params,
        )

class MLPModel(nn.Module):
    """A MLP model with 3 dense layers."""
    def __init__(
        self,
        n_features: int,
        hidden_size: Sequence[int],
    ):
        super(MLPModel, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(n_features, hidden_size[0])])
        for i in range(1, len(hidden_size)):
            self.layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.layers.append(nn.Linear(hidden_size[-1], 1))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
