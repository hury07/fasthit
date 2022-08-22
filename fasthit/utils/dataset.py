from copy import copy
from typing import Sequence, Tuple
from typing_extensions import Self
import torch

from torch.utils.data import Dataset


def collate_batch(batch_list):
    assert type(batch_list) == list, f"Error"
    batch_size = len(batch_list)
    combo_batch = torch.cat([item[0]
                            for item in batch_list]).reshape(batch_size, -1)
    temp_seqs = torch.cat([item[1] for item in batch_list]
                          ).reshape(batch_size, -1)
    return combo_batch, temp_seqs


class SequenceData(Dataset):
    def __init__(self, residue_chars: Sequence[str], target_python_idxs: Sequence[int], wild_type: Sequence[str]) -> None:
        super().__init__()
        self.residue_chars = residue_chars
        self.mut_seqs = tuple(self.format(
            residue_chars, target_python_idxs, wild_type))

    @staticmethod
    def format(residue_chars, target_python_idxs, wild_type) -> Tuple[str]:
        ret = ['' for _ in range(len(residue_chars))]
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
