"""Extreme Learning Machine classifier (scikit-learn compatible).

Single hidden layer feed-forward network:
    H = g(X W + b)          W, b: random, fixed (never trained)
    beta = argmin ||H beta - T||^2 + alpha ||beta||^2
         = (H^T H + alpha I)^-1 H^T T     (ridge-regularised Moore-Penrose solution)

Training is a single linear solve, so there is no backpropagation and no epochs.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted, validate_data

_ACTIVATIONS = {
    "sigm": lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))),
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0),
}


class ELMClassifier(ClassifierMixin, BaseEstimator):
    """Ridge-regularised ELM for multi-class classification.

    Parameters
    ----------
    n_neurons : int
        Number of hidden neurons.
    ufunc : {"sigm", "tanh", "relu"}
        Hidden layer activation.
    alpha : float
        L2 regularisation strength on the output weights.
    random_state : int | None
        Seed for the random hidden layer.
    """

    def __init__(self, n_neurons: int = 2048, ufunc: str = "sigm", alpha: float = 1e-3,
                 random_state: int | None = None) -> None:
        self.n_neurons = n_neurons
        self.ufunc = ufunc
        self.alpha = alpha
        self.random_state = random_state

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        return _ACTIVATIONS[self.ufunc](X @ self.input_weights_ + self.biases_)

    def fit(self, X, y):
        if self.ufunc not in _ACTIVATIONS:
            raise ValueError(f"ufunc must be one of {sorted(_ACTIVATIONS)}, got {self.ufunc!r}")
        X, y = validate_data(self, X, y, dtype=np.float64)

        self._binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        T = self._binarizer.fit_transform(y)
        if T.shape[1] == 1:  # binary problem -> two output columns
            T = np.hstack([-T, T])
        self.classes_ = self._binarizer.classes_

        rng = np.random.default_rng(self.random_state)
        # Scaled init keeps pre-activations in a sensible range for standardised inputs.
        self.input_weights_ = rng.normal(0.0, 1.0 / np.sqrt(X.shape[1]), (X.shape[1], self.n_neurons))
        self.biases_ = rng.normal(0.0, 1.0, self.n_neurons)

        H = self._hidden(X)
        gram = H.T @ H
        gram[np.diag_indices_from(gram)] += self.alpha
        self.output_weights_ = np.linalg.solve(gram, H.T @ T)
        return self

    def decision_function(self, X) -> np.ndarray:
        check_is_fitted(self, "output_weights_")
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return self._hidden(X) @ self.output_weights_

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
