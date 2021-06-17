"""Define the base PytorchModel class."""
#from typing import Callable

import numpy as np
import torch.nn as nn
from skorch import NeuralNet

import fasthit


class TorchModel(fasthit.Model):
    """A wrapper around Pytorch models."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        #custom_train_function: Callable[[torch.Tensor, torch.Tensor], None] = None,
        #custom_predict_function: Callable[[torch.Tensor], np.ndarray] = None,
        **fit_params,
    ):
        """
        Wrap a Pytorch model.

        Args:
            model: A callable and fittable keras model.
            alphabet: Alphabet string.
            name: Human readable description of model (used for logging).
            custom_train_function: A function that receives a tensor of one-hot
                sequences and labels and trains `model`.
            custom_predict_function: A function that receives a tensor of one-hot
                sequences and predictions.
            fit_params: Parameters for model training

        """
        super().__init__(name)

        self.model = NeuralNet(
            model,
            train_split=None,
            warm_start=True,
            **fit_params
        ).initialize()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
    ):
        """Train Pytorch model."""
        y = np.expand_dims(y, axis=1).astype(np.float32)

        self.model.set_params(verbose=verbose)
        self.model.fit(X, y)

    def _fitness_function(self, X: np.ndarray):
        return np.nan_to_num(
            self.model.predict(X).squeeze(axis=1)
        )
