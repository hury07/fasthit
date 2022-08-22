###
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from fasthit import Encoder
from fasthit.error import EsmImportError
from fasthit.utils.dataset import SequenceData
from torch.utils.data import DataLoader

from .esm import encodings


class ESM_Tokenizer(Encoder):
    def __init__(
        self,
        encoding: str,
        wt_seq: Sequence[str],
        target_python_idxs: Sequence[int],
        pretrained_model_dir: str
    ):
        try:
            from esm import pretrained
        except ImportError:
            raise EsmImportError

        assert encoding in ["esm-1b", "esm-1v"]
        name = f"ESM_{encoding}"

        pretrained_model_name = encodings.loc[encoding]["model"]
        _, esm_alphabet = pretrained.load_model_and_alphabet(
            pretrained_model_dir + pretrained_model_name + ".pt"
        )
        self._wt_list = list(wt_seq)
        self._target_python_idxs = target_python_idxs
        self.batch_converter = esm_alphabet.get_batch_converter()
        super().__init__(name, esm_alphabet.all_toks, len(esm_alphabet))

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        # [bsz, seq_length]
        dataset = SequenceData(
            sequences, self._target_python_idxs, self._wt_list)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)

        outs = [None for _ in range(len(dataloader))]
        for i, data in enumerate(dataloader):
            _, _, out = self.batch_converter(data)
            outs[i] = out
        return torch.vstack(outs)
