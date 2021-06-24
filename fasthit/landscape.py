"""Defines the Landscape class."""
import abc

import numpy as np
from typing import List, Union


class Landscape(abc.ABC):
    """
    Base class for all landscapes and for `fasthit.Model`.

    Attributes:
        cost (int): Number of sequences whose fitness has been evaluated.
        name (str): A human-readable name for the landscape (often contains
            parameter values in the name) which is used when logging explorer runs.

    """
    def __init__(self, name: str):
        """Create Landscape, setting `name` and setting `cost` to zero."""
        self._cost = 0
        self._name = name

    @abc.abstractmethod
    def _fitness_function(self, sequences: Union[List[str], np.ndarray]) -> np.ndarray:
        pass

    def get_fitness(self, sequences: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Score a list/numpy array of sequences.

        This public method should not be overriden – new landscapes should
        override the private `_fitness_function` method instead. This method
        increments `self.cost` and then calls and returns `_fitness_function`.

        Args:
            sequences: A list/numpy array of sequence strings to be scored.

        Returns:
            Scores for each sequence.

        """
        self._cost += len(sequences)
        return self._fitness_function(sequences)
    
    @property
    def name(self):
        return self._name
    
    @property
    def cost(self):
        return self._cost
    
    @cost.setter
    def cost(self, cost):
        self._cost = cost
