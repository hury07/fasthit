from copy import copy
from typing import Sequence, Tuple, Union
from typing_extensions import Self
import torch

from torch.utils.data import Dataset


def collate_batch(batch_list: Sequence[Union[Tuple, Sequence[Tuple]]]):
    return batch_list


class SequenceData(Dataset):
    def __init__(self, residue_chars: Sequence[str], target_python_idxs: Sequence[int], wild_type: Sequence[str]) -> None:
        super().__init__()
        self.residue_chars = residue_chars
        self.mut_seqs = tuple(self.format(
            residue_chars, target_python_idxs, wild_type))

    @staticmethod
    def format(residue_chars, target_python_idxs, wild_type) -> Tuple[str]:
        ret = []
        for chars in residue_chars:
            seq_list = copy(wild_type)
            for idx, c in zip(target_python_idxs, chars):
                seq_list[idx] = c
            ret.append(''.join(seq_list))
        return tuple(ret)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.residue_chars[index], self.mut_seqs[index]
    
    def __len__(self) -> int:
        return len(self.residue_chars)
