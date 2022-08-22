# Define abstract encoder class: Encoder
import abc
from dataclasses import dataclass
from typing import Sequence

import numpy as np


class Encoder(abc.ABC):
    def __init__(
        self,
        name,
        alphabet: str,
        n_features: int,
        batch_size: int = 1,
    ):
        super().__init__()

        self._name = name
        self._alphabet = alphabet
        self._n_features = n_features
        self._batch_size = batch_size

    @abc.abstractmethod
    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        pass

    @property
    def name(self):
        return self._name

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def n_features(self):
        return self._n_features

    @property
    def batch_size(self):
        return self._batch_size
