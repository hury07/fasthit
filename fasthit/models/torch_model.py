"""Define the base PytorchModel class."""
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

        self._model = NeuralNet(
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

        self._model.set_params(verbose=verbose)
        self._model.fit(X, y)

    def _fitness_function(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num(
            self._model.predict(X).squeeze(axis=1)
        )
    
    @property
    def model(self):
        return self._model
