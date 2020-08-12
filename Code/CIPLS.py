"""Covariance-free Partial Least Squares"""

# Author: Artur Jordao <arturjlcorreia[at]gmail.com>
#         Artur Jordao

import numpy as np
from scipy import linalg

from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
import copy

class CIPLS(BaseEstimator):
    """Covariance-free Partial Least Squares (CIPLS).

    Parameters
    ----------
    n_components : int or None, (default=None)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    References
    Covariance-free Partial Least Squares: An Incremental Dimensionality Reduction Method
    """

    def __init__(self, n_components=10, copy=True):
        self.__name__ = 'Covariance-free Partial Least Squares'
        self.n_components = n_components
        self.n = 0
        self.copy = copy
        self.sum_x = None
        self.sum_y = None
        self.n_features = None
        self.x_rotations = None
        self.x_loadings = None
        self.y_loadings = None
        self.eign_values = None
        self.x_mean = None
        self.p = []

    def normalize(self, x):
        return normalize(x[:, np.newaxis], axis=0).ravel()

    def fit(self, X, Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy, ensure_2d=False)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if np.unique(Y).shape[0] == 2:
            Y[np.where(Y == 0)[0]] = -1

        n_samples, n_features = X.shape

        if self.n == 0:
            self.x_rotations = np.zeros((self.n_components, n_features))
            self.x_loadings = np.zeros((n_features, self.n_components))
            self.y_loadings = np.zeros((Y.shape[1], self.n_components))
            self.n_features = n_features
            self.eign_values = np.zeros((self.n_components))
            self.p = [0] * self.n_components

        for j in range(0, n_samples):
            self.n = self.n + 1
            u = X[j]
            l = Y[j]

            if self.n == 1:
                self.sum_x = u
                self.sum_y = l
            else:
                old_mean = 1 / (self.n - 1) * self.sum_x
                self.sum_x = self.sum_x + u
                mean_x = 1 / self.n * self.sum_x
                u = u - mean_x
                delta_x = mean_x - old_mean
                self.x_rotations[0] = self.x_rotations[0] - delta_x * self.sum_y
                self.x_rotations[0] = self.x_rotations[0] + (u * l)
                self.sum_y = self.sum_y + l

                t = np.dot(u, self.normalize(self.x_rotations[0].T))

                self.x_loadings[:, 0] = self.x_loadings[:, 0] + (u * t)
                self.y_loadings[:, 0] = self.y_loadings[:, 0] + (l * t)

                for c in range(1, self.n_components):
                    u -= np.dot(t, self.x_loadings[:, c - 1])
                    l -= np.dot(t, self.y_loadings[:, c - 1])

                    self.x_rotations[c] = self.x_rotations[c] + (u * l)
                    self.x_loadings[:, c] = self.x_loadings[:, c] + (u * t)
                    self.y_loadings[:, c] = self.y_loadings[:, c] + (l * t)
                    t = np.dot(u, self.normalize(self.x_rotations[c].T))

        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        mean = 1 / self.n * self.sum_x
        X -= mean
        w_rotation = np.zeros(self.x_rotations.shape)

        for c in range(0, self.n_components):
            w_rotation[c] = self.normalize(self.x_rotations[c])

        return np.dot(X, w_rotation.T)
