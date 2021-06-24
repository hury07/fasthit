"""Define scikit-learn model wrappers as well a few convenient pre-wrapped models."""
import abc

import numpy as np
from sklearn import linear_model, ensemble

import fasthit


class SklearnModel(fasthit.Model, abc.ABC):
    """Base sklearn model wrapper."""
    def __init__(self, name, model):
        """
        Args:
            model: sklearn model to wrap.
            alphabet: Alphabet string.
            name: Human-readable short model descriptipon (for logging).

        """
        super().__init__(name)

        self._model = model

    def train(self, X: np.ndarray, labels: np.ndarray):
        """Flatten one-hot sequences and train model using `model.fit`."""
        flattened = X.reshape(X.shape[0], -1)
        self._model.fit(flattened, labels)
    
    def _fitness_function(self, X: np.ndarray) -> np.ndarray:
        flattened = X.reshape(X.shape[0], -1)
        return self._model.predict(flattened)
    
    @property
    def model(self):
        return self._model


class LinearRegression(SklearnModel):
    """Sklearn linear regression."""
    def __init__(self, **kwargs):
        """Create linear regression model."""
        model = linear_model.LinearRegression(**kwargs)
        super().__init__("linear_regression", model)


class RandomForestRegressor(SklearnModel):
    """Sklearn random forest regressor."""
    def __init__(self, **kwargs):
        """Create random forest regressor."""
        model = ensemble.RandomForestRegressor(**kwargs)
        super().__init__("random_forest", model)