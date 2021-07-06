"""Define a MLP Model."""
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Sequence, Union

from .torch_model import TorchModel

class MLP(TorchModel):
    def __init__(
        self,
        seq_length: int,
        input_features: int,
        name: str = None,
        nogpu: bool = False,
        n_layers: int = 3,
        hidden_size: Union[Sequence[int], int] = 80, # defaults: 80, 64
        activiation_func: str = "relu",
        drop_prob: float = 0., # defaults: 0, 0.1
        **train_params,
    ):
        ### set model parameters
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size for _ in range(n_layers)]
        assert len(hidden_size) == n_layers
        ###
        activiation_funcs = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
        }
        model = MLPModel(
            input_features * seq_length,
            hidden_size,
            activiation_funcs[activiation_func],
            drop_prob,
        )
        ### set fit parameters
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu'
        )
        fit_params = {
            "lr": 1e-3,
            "max_epochs": 20, # default: 20
            "batch_size": 64, # default: 64?
            "train_split": 5, # default: 5-fold cv
            "callbacks": ["early_stop"],
            "criterion": nn.MSELoss,
            "optimizer": optim.Adam,
            "device": device,
        }
        fit_params.update(train_params)

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
        activiation_func: nn.Module = nn.ReLU(),
        drop_prob: float = 0.,
    ):
        super(MLPModel, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(n_features, hidden_size[0])])
        for i in range(1, len(hidden_size)):
            self.layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.activiation_func = activiation_func
        self.dropout = nn.Dropout(drop_prob)
        self.output = nn.Linear(hidden_size[-1], 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = self.dropout(self.activiation_func(layer(x)))
        x = self.output(x)
        return x
