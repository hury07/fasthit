### Define abstract encoder class: Encoder
import abc
from typing import List

import numpy as np


class Encoder(abc.ABC):
    def __init__(
        self,
        name: str,
        alphabet: str,
        n_features: int,
        batch_size: int = 1,
    ):
        super().__init__()

        self.name = name
        self.alphabet = alphabet
        self.n_features = n_features
        self.batch_size = batch_size

    @abc.abstractmethod
    def encode(self, sequences: List[str]) -> np.array:
        pass