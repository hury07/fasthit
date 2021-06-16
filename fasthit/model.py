"""Defines base Model class."""
import abc
from typing import Any, List

import numpy as np

import fasthit
from fasthit.types import SEQUENCES_TYPE


class Model(fasthit.Landscape, abc.ABC):
    """
    Base model class. Inherits from `fasthit.Landscape` and adds an additional
    `train` method.

    """

    @abc.abstractmethod
    def train(self, sequences: SEQUENCES_TYPE, labels: List[Any]):
        """
        Train model.

        This function is called whenever you would want your model to update itself
        based on the set of sequences it has measurements for.

        """
        pass


class LandscapeAsModel(Model):
    """
    This simple class wraps a `fasthit.Landscape` in a `fasthit.Model` to allow running
    experiments against a perfect model.

    This class's `_fitness_function` simply calls the landscape's `_fitness_function`.
    """

    def __init__(self, landscape: fasthit.Landscape):
        """
        Create a `fasthit.Model` out of a `fasthit.Landscape`.

        Args:
            landscape: Landscape to wrap in a model.

        """
        super().__init__(f"LandscapeAsModel={landscape.name}")
        self.landscape = landscape

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        return self.landscape._fitness_function(sequences)

    def train(self, sequences: SEQUENCES_TYPE, labels: List[Any]):
        """No-op."""
        pass
