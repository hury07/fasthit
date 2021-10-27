"""Define a Finetune Model."""
from typing import Sequence, Union
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from esm import pretrained

from fasthit.encoders.esm import encodings
from fasthit.models.mlp import MLPModel

from .torch_model import TorchModel


class Finetune(TorchModel):
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
            "batch_size": 32,  # default: 64?
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

class FinetuneModel(nn.Module):
    """A Finetune model with several features."""

    def __init__(
        self,
        pretrained_model_dir: str,
        pretrained_model_name: str,
        pretrained_model_exprs: Sequence[int],
        target_python_idxs: Sequence[int],
        hidden_size_merge: int,
        hidden_size: Sequence[int],
        activiation_func: nn.Module = nn.ReLU(),
        drop_prob: float = 0.,
    ):
        super().__init__()
        self.pretrained_model_exprs = pretrained_model_exprs
        self.target_protein_idxs = [idx + 1 for idx in target_python_idxs]
        self.target_len = len(target_python_idxs)

        self.pretrained_model, self.esm_alphabet = pretrained.load_model_and_alphabet(
            pretrained_model_dir + pretrained_model_name + ".pt"
        )
        self.pretrained_model.train()

        self.finetune_head = MLPModel(
            1280 * self.target_len * len(self.pretrained_model_exprs),
            hidden_size,
            activiation_func,
            drop_prob,
        )

    def forward(self, x):
        out = self.pretrained_model(
                x, repr_layers=self.pretrained_model_exprs, return_contacts=False)
        embeddings = out["representations"]
        embeddings = torch.stack([v for _, v in embeddings.items()], dim=-1)
        embeddings = embeddings[:, self.target_protein_idxs, :, :]
        x = self.finetune_head(embeddings)
        return x
