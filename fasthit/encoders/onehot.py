###
import numpy as np
from typing import List, Optional

from fasthit.utils import sequence_utils as s_utils

class OneHot(object):
    def __init__(self, alphabet: str) -> None:
        super().__init__()
        self.alphabet = alphabet

    def encode(
        self,
        sequences: List[str],
        batch_size: Optional[int] = None
    ) -> np.array:
        ### [bsz, seq_length, n_features]
        return np.array(
            [s_utils.string_to_one_hot(seq, self.alphabet) for seq in sequences],
            dtype=np.float32,
        )