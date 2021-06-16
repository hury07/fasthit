"""Define a CNN Model."""
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import fasthit
from fasthit import encoders


from .torch_model import TorchModel


class CNN(TorchModel):
    def __init__(
        self,
        seq_len: int,
        num_filters: int,
        hidden_size: int,
        kernel_size: int,
        alphabet: str,
        encoding: str = "onehot",
        landscape: Optional[fasthit.Landscape] = None,
        name: str = None,
        **new_fit_params,
    ):
        """Create the CNNModel."""
        n_features = len(alphabet)
        if encoding == "georgiev":
            n_features = encoders.Georgiev.n_features
        elif encoding in ["transformer", "unirep", "trrosetta"]:
            n_features = encoders.TAPE.encodings.loc[encoding, "n_features"].item()
        elif encoding in ["esm-1b", "esm-msa-1"]:
            n_features = encoders.ESM.encodings.loc[encoding, "n_features"].item()
        else:
            pass

        model = CNNModel(
            n_features,
            num_filters,
            hidden_size,
            kernel_size,
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
            name = f"CNN_hidden_size_{hidden_size}_num_filters_{num_filters}"

        super().__init__(
            name,
            model,
            alphabet,
            encoding=encoding,
            landscape=landscape,
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
