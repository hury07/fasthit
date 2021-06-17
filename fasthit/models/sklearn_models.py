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

        self.model = model

    def train(self, X: np.ndarray, labels: np.ndarray):
        """Flatten one-hot sequences and train model using `model.fit`."""
        flattened = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        self.model.fit(flattened, labels)

###
#   Regressors
###
class SklearnRegressor(SklearnModel, abc.ABC):
    """Class for sklearn regressors (uses `model.predict`)."""
    def _fitness_function(self, X: np.ndarray):
        flattened = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        return self.model.predict(flattened)


class LinearRegression(SklearnRegressor):
    """Sklearn linear regression."""
    def __init__(self, **kwargs):
        """Create linear regression model."""
        model = linear_model.LinearRegression(**kwargs)
        super().__init__("linear_regression", model)


class RandomForestRegressor(SklearnRegressor):
    """Sklearn random forest regressor."""
    def __init__(self, **kwargs):
        """Create random forest regressor."""
        model = ensemble.RandomForestRegressor(**kwargs)
        super().__init__("random_forest", model)


###
#   Classifiers
###
class SklearnClassifier(SklearnModel, abc.ABC):
    """Class for sklearn classifiers (uses `model.predict_proba(...)[:, 1]`)."""
    def _fitness_function(self, X: np.ndarray):
        flattened = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        return self.model.predict_proba(flattened)[:, 1]


class LogisticClassifier(SklearnClassifier):
    """Sklearn logistic regression."""
    def __init__(self, **kwargs):
        """Create logistic regression model."""
        model = linear_model.LogisticRegression(**kwargs)
        super().__init__("logistic_regression", model)