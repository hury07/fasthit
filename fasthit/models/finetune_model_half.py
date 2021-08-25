"""Define a Finetune Model."""
import os
from fasthit.models.mlp import MLPModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from esm import pretrained

from typing import Sequence, Union

from .torch_model import TorchModel

from fasthit.encoders.esm import encodings

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class FinetuneHalf(TorchModel):
    def __init__(
        self,
        pretrained_model_dir: str,
        pretrained_model_name: str,
        pretrained_model_exprs: Sequence[int],
        target_python_idxs: Sequence[int],
        name: str = None,
        nogpu: bool = False,
        ddp: bool = False,
        n_layers: int = 2,
        hidden_size_merge: int = 100,
        hidden_size: Union[Sequence[int], int] = 100,  # defaults: 80, 64
        activiation_func: str = "relu",
        drop_prob: float = 0.,  # defaults: 0, 0.1
        **train_params,
    ):
        # set model parameters
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size for _ in range(n_layers)]
        assert len(hidden_size) == n_layers
        assert pretrained_model_name in ["esm-1b", "esm-1v"]
        pretrained_model_name = encodings.loc[pretrained_model_name]["model"]
        ###
        activiation_funcs = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
        }
        model = FinetuneModel(
            pretrained_model_dir,
            pretrained_model_name,
            pretrained_model_exprs,
            target_python_idxs,
            hidden_size_merge,
            hidden_size,
            activiation_funcs[activiation_func],
            drop_prob,
        )
        # set fit parameters
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu'
        )
        # ddp configure
        # TODO: finish ddp if nessassary
        if ddp:
            raise NotImplementedError
        if not nogpu and ddp:
            local_rank = os.environ['local_rank']
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
            device = torch.device("cuda", local_rank)
            model = model.to(device)
            model = DDP(model, device_ids=[
                        local_rank], output_device=local_rank)

        fit_params = {
            "lr": 1e-3,
            "max_epochs": 20,  # default: 20
            "batch_size": 128,  # default: 64?
            "train_split": 5,  # default: 5-fold cv
            "callbacks": ["early_stop"],
            "criterion": nn.MSELoss,
            "optimizer": optim.Adam,
            "device": device,
        }
        fit_params.update(train_params)

        if name is None:
            name = f"finetune_{hidden_size}"

        super().__init__(
            name,
            model,
            **fit_params,
        )


class MergeModel(nn.Module):
    "Use MLP to merge several features"

    def __init__(
        self,
        pretrained_model_exprs: Sequence[int],
        target_protein_idxs: Sequence[int],
        input_size: int,
        hidden_size: int,
        activiation_func: nn.Module = nn.ReLU(),
        drop_prob: float = 0.,
    ):
        super().__init__()
        self.feature_num = len(pretrained_model_exprs)
        self.layer_index = pretrained_model_exprs
        self.target_protein_idxs = target_protein_idxs

        self.layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) for _ in range(self.feature_num)])
        self.activiation_func = activiation_func
        self.dropout = nn.Dropout(drop_prob)
        self.output = nn.LayerNorm(hidden_size, hidden_size)

    def forward(self, features: torch.Tensor):
        assert features.dim() == 4
        feature_outputs = [self.dropout(self.activiation_func(
            layer(features[self.layer_index[i]][:, self.target_protein_idxs, :, i]))) for i, layer in enumerate(self.layers)]
        x = torch.stack(feature_outputs, dim=0).sum(dim=0)
        x = self.output(x)
        return x


class FinetuneModel(nn.Module):
    """A Finetune model with several features."""

    def __init__(
        self,
        pretrained_model_dir: str,
        pretrained_model_name: str,
        pretrained_model_exprs: Sequence[int],
        target_protein_idxs: Sequence[int],
        hidden_size_merge: int,
        hidden_size: Sequence[int],
        activiation_func: nn.Module = nn.ReLU(),
        drop_prob: float = 0.,
    ):
        super().__init__()
        self.pretrained_model_exprs = pretrained_model_exprs
        self.target_protein_idxs = [idx + 1 for idx in target_protein_idxs]
        self.target_len = len(target_protein_idxs)

        self.feature_merge = MergeModel(
            self.pretrained_model_exprs,
            self.target_protein_idxs,
            self.pretrained_model.args.embed_dim,
            hidden_size_merge,
            activiation_func,
            drop_prob,
        )
        self.finetune_head = MLPModel(
            hidden_size_merge * self.target_len,
            hidden_size,
            activiation_func,
            drop_prob,
        )

    def forward(self, x):
        x = self.feature_merge(x)
        x = self.finetune_head(x)
        return x
