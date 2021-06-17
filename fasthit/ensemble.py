"""Defines the Ensemble class."""
from typing import Callable, List

import numpy as np

import fasthit


class Ensemble(fasthit.Model):
    """
    Class to ensemble models or landscapes together.

    Attributes:
        models (List[fasthit.Landscape]): List of landscapes/models being ensembled.
        combine_with (Callable[[np.ndarray], np.ndarray]): Function to combine ensemble
            predictions.

    """

    def __init__(
        self,
        models: List[fasthit.Landscape],
        combine_with: Callable[[np.ndarray], np.ndarray] = lambda x: np.mean(x, axis=1),
    ):
        """
        Create ensemble.

        Args:
            models: List of landscapes/models to ensemble.
            combine_with: A function that takes in a matrix of scores
                (num_seqs, num_models) and combines ensembled model scores into an
                array (num_seqs,).

        """
        name = f"Ens({'|'.join(model.name for model in models)})"
        super().__init__(name)

        self.models = models
        self.combine_with = combine_with

    def train(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Train each model in `self.models`.

        Args:
            sequences: Training sequences
            labels: Training labels

        """
        for model in self.models:
            model.train(sequences, labels)

    def _fitness_function(self, sequences):
        scores = np.stack(
            [model.get_fitness(sequences) for model in self.models], axis=1
        )

        return self.combine_with(scores)
