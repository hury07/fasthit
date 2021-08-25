###
import math
from typing import Sequence, Union

import numpy as np
import torch
from esm import pretrained
from fasthit.encoders.esm import encodings

import fasthit


class Extend(fasthit.Encoder):
    def __init__(self,
                 encoding: str,
                 wt_seq: Sequence[str],
                 target_python_idxs: Sequence[int],
                 pretrained_model_dir: str):

        assert encoding in ["esm-1b", "esm-1v"]
        pretrained_model_name = encodings.loc[encoding]["model"]
        _, esm_alphabet = pretrained.load_model_and_alphabet(
            pretrained_model_dir + pretrained_model_name + ".pt"
        )
        self.wt_list = list(wt_seq)
        self.target_python_idxs = target_python_idxs
        self.batch_converter = esm_alphabet.get_batch_converter()
        super().__init__("onehot", esm_alphabet.all_toks, len(esm_alphabet))

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        ### [bsz, seq_length]
        temp_seqs = [None for _ in range(len(sequences))]
        
        outs = []
        n_batches = math.ceil(len(sequences) / self.batch_size)
        for i, seq_batch in enumerate(np.array_split(sequences, n_batches)):
            for i, combo in enumerate(seq_batch):
                temp_seq = self.wt_list.copy()
                # Loop over the target python indices and set new amino acids
                for aa, ind in zip(combo, self.target_python_idxs):
                    temp_seq[ind] = aa
                temp_seqs[i] = "".join(temp_seq)
            data = [(label, seq) for label, seq in zip(seq_batch, temp_seqs)]
            _, _, out = self.batch_converter(data)
            outs.append(out)
        return torch.vstack(outs)
