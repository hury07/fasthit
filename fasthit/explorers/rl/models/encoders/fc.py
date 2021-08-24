from typing import Optional
import torch
import torch.nn as nn

from ding.torch_utils import ResFCBlock, ResBlock
from ding.utils import SequenceType

class FCEncoder(nn.Module):
    r"""
    Overview:
        The ``FCEncoder`` used in models. Used to encoder raw 1-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            obs_shape: int,
            hidden_size_list: SequenceType,
            res_block: bool = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the FC Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`int`): Observation shape
            - hidden_size_list (:obj:`SequenceType`): The collection of ``hidden_size``
            - res_block (:obj:`bool`): Whether use ``res_block``.
            - activation (:obj:`nn.Module`):
                The type of activation to use in the ``ResFCBlock``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.ResFCBlock`` for more details
        """
        super(FCEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.init = nn.Linear(obs_shape, hidden_size_list[0])

        if res_block:
            assert len(set(hidden_size_list)) == 1, "Please indicate the same hidden size for res block parts"
            if len(hidden_size_list) == 1:
                self.main = ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type)
            else:
                layers = []
                for i in range(len(hidden_size_list)):
                    layers.append(ResFCBlock(hidden_size_list[0], activation=self.act, norm_type=norm_type))
                self.main = nn.Sequential(*layers)
        else:
            layers = []
            for i in range(len(hidden_size_list) - 1):
                layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
                layers.append(self.act)
            self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation
        Returns:
            - outputs (:obj:`torch.Tensor`): Embedding tensor
        """
        x = self.act(self.init(x))
        x = self.main(x)
        return x