"""Define a CNN Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .torch_model import TorchModel


class CNN(TorchModel):
    def __init__(
        self,
        input_features: int,
        num_filters: int,
        hidden_size: int,
        kernel_size: int,
        name: str = None,
        nogpu: bool = True,
        **new_fit_params,
    ):
        """Create the CNNModel."""

        model = CNNModel(
            input_features,
            num_filters,
            hidden_size,
            kernel_size,
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
            "callbacks": ["early_stop"],
            "train_split": 5, # default: 5-fold cv
        }
        fit_params.update(new_fit_params)

        if name is None:
            name = f"CNN_hidden_size_{hidden_size}_num_filters_{num_filters}"

        super().__init__(
            name,
            model,
            **fit_params,
        )

class CNNModel(nn.Module):
    """A CNN model with 3 conv layers and 2 dense layers."""
    def __init__(
        self,
        n_features: int,
        num_filters: int,
        hidden_size: int,
        kernel_size: int,
    ):
        super(CNNModel, self).__init__()
        
        self.conv1d_1 = nn.Conv1d(
                            n_features,
                            num_filters,
                            kernel_size=kernel_size,
                            padding="same",
        )
        self.conv1d_2 = nn.Conv1d(
                            num_filters,
                            num_filters,
                            kernel_size=kernel_size,
                            padding="same",
        )
        self.conv1d_3 = nn.Conv1d(
                            num_filters,
                            num_filters,
                            kernel_size=n_features - 1,
                            padding="same"
        )
        self.linear_1 = nn.Linear(num_filters, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1d_1(x))
        x = F.relu(self.conv1d_2(x))
        x = F.relu(self.conv1d_3(x))

        x = F.max_pool1d(x, x.shape[-1]).squeeze(dim=-1)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x
