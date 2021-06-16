"""Define the base PytorchModel class."""
from typing import Optional#, Callable

import numpy as np
import torch.nn as nn
from skorch import NeuralNet

import fasthit
from fasthit import encoders
from fasthit.types import SEQUENCES_TYPE
from fasthit.utils import sequence_utils as s_utils


class TorchModel(fasthit.Model):
    """A wrapper around Pytorch models."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        alphabet: str,
        encoding: str = "onehot",
        landscape: Optional[fasthit.Landscape] = None,
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

        self.alphabet = alphabet
        self.batch_size = fit_params["batch_size"]
        # Encoding strategy
        if encoding == "onehot":
            self.encoder = encoders.OneHot(self.alphabet)
        elif encoding == "georgiev":
            self.encoder = encoders.Georgiev()
        elif encoding in ["transformer", "unirep", "trrosetta"]:
            self.encoder = encoders.TAPE(
                landscape.name, self.alphabet, encoding, landscape.wt, landscape.combo_python_inds
            )
        elif encoding in ["esm-1b", "esm-msa-1"]:
            self.encoder = encoders.ESM(
                landscape.name, self.alphabet, encoding, landscape.wt, landscape.combo_python_inds
            )
        else:
            raise NotImplementedError(f"Unknown encoding strategy: {encoding}.")

        self.model = NeuralNet(
            model,
            train_split=None,
            warm_start=True,
            **fit_params
        ).initialize()

    def train(
        self, sequences: SEQUENCES_TYPE, labels: np.ndarray, verbose: bool = False
    ):
        """Train Pytorch model."""
        encodings = self.encoder.encode(sequences, batch_size=self.batch_size)
        labels = np.expand_dims(labels, axis=1).astype(np.float32)

        self.model.set_params(verbose=verbose)
        self.model.fit(encodings, labels)

    def _fitness_function(self, sequences):
        encodings = self.encoder.encode(sequences, batch_size=self.batch_size)
        return np.nan_to_num(
            self.model.predict(encodings).squeeze(axis=1)
        )
