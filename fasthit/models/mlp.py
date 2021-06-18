"""Define a MLP Model."""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .torch_model import TorchModel


class MLP(TorchModel):
    def __init__(
        self,
        seq_length: int,
        input_features: int,
        hidden_size: int,
        name: str = None,
        **new_fit_params,
    ):
        """Create the MLPModel."""

        model = MLPModel(
            input_features * seq_length,
            hidden_size,
        )
        ### set fit parameters
        fit_params = {
            "max_epochs": 20,
            "batch_size": 256,
            "lr": 1e-3,
            "criterion": nn.MSELoss,
            "optimizer": optim.Adam,
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
        hidden_size: int,
    ):
        super(MLPModel, self).__init__()

        self.linear_1 = nn.Linear(n_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = self.output(x)
        return x
