###
from typing import List

import numpy as np

import fasthit
from fasthit.utils import sequence_utils as s_utils

class OneHot(fasthit.Encoder):
    def __init__(self, alphabet: str):
        super().__init__("onehot", alphabet, len(alphabet))

    def encode(self, sequences: List[str]) -> np.array:
        ### [bsz, seq_length, n_features]
        return np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences],
            dtype=np.float32,
        )