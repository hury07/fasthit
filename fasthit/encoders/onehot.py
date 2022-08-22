###
from typing import Sequence

import numpy as np
from fasthit import Encoder
from fasthit.utils import sequence_utils as s_utils


class OneHot(Encoder):
    def __init__(self, alphabet: str):
        super().__init__("onehot", alphabet, len(alphabet))

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        ### [bsz, seq_length, n_features]
        encoding_stack = list(map(lambda seq: s_utils.string_to_one_hot(
            seq, self.alphabet), sequences))

        return np.stack(encoding_stack, axis=0).astype(np.float32)
