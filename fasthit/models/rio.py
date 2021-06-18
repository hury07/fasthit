from copy import deepcopy
from typing import Union

import numpy as np

import fasthit
from fasthit.models import TorchModel, SklearnModel, GPRegressor

class RIO(fasthit.Model):
    def __init__(
        self,
        mean_module: Union[TorchModel, SklearnModel],
        uncertainty_module: GPRegressor,
        name: str = None,
    ):
        if name is None:
            name = f"{mean_module.name}_{uncertainty_module.name}"
        super().__init__(name)

        self._mean_module = deepcopy(mean_module)
        self._uncertainty_module = deepcopy(uncertainty_module)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        self._mean_module.train(X, y, verbose=verbose)
        residual = y - self._mean_module.get_fitness(X)
        self._uncertainty_module.train(X, residual, verbose=verbose)

    def _fitness_function(self, X: np.ndarray) -> np.ndarray:
        y_hat = self._mean_module.get_fitness(X)
        residual = self._uncertainty_module.get_fitness(X)
        return np.nan_to_num(
            y_hat + residual
        )
    
    @property
    def mean_module(self):
        return self._mean_module
    
    @property
    def uncertainty_module(self):
        return self._uncertainty_module
    
    @property
    def uncertainties(self):
        return self._uncertainty_module.uncertainties