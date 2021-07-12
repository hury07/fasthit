import numpy as np
import torch.nn as nn
from skorch import NeuralNet
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping

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
        ###
        callbacks = []
        cbs = fit_params.pop("callbacks", [])
        for cb in cbs:
            if cb == "early_stop":
                callbacks.append(EarlyStopping().initialize())
            else:
                pass
        ###
        self._train_split = fit_params.pop("train_split", 0)
        if self._train_split == 0:
            train_split = None
            callbacks = []
        else:
            train_split = CVSplit(cv=self._train_split)
        warm_start = fit_params.pop("warm_start", False)
        ###
        self._model = NeuralNet(
            module=model,
            train_split=train_split,
            callbacks=callbacks,
            warm_start=warm_start, # default: False
            **fit_params
        ).initialize()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
    ):
        """Train Pytorch model."""
        if X.shape[0] < self._train_split:
            X = np.resize(X, (self._train_split * X.shape[0], *X.shape[1:]))
            y = np.resize(y, (self._train_split * y.shape[0], *y.shape[1:]))

        self._model.set_params(verbose=verbose)
        self._model.fit(X, y)

    def _fitness_function(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num(
            self._model.predict(X).squeeze(axis=1)
        )
    
    @property
    def model(self):
        return self._model
