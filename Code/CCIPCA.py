"""Candid Covariance-Free Incremental PCA (CCIPCA)."""

import numpy as np
from scipy import linalg

from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
import copy

class CCIPCA(BaseEstimator):
    """Candid Covariance-Free Incremental PCA (CCIPCA).

    Parameters
    ----------
    n_components : int or None, (default=None)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    References
    Candid Covariance-free Incremental Principal Component Analysis
    """

    def __init__(self, n_components=10, amnesic=2, copy=True):
        self.__name__ = 'Incremental Projection on Latent Space (IPLS)'
        self.n_components = n_components
        self.amnesic = amnesic
        self.n = 0
        self.copy = copy

        self.x_rotations = None
        self.sum_x = None
        self.n_features = None
        self.eign_values = None
        self.x_mean = None

    def normalize(self, x):
        return normalize(x[:, np.newaxis], axis=0).ravel()

    def fit(self, X, Y=None):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)

        n_samples, n_features = X.shape

        if self.n == 0:
            self.n_features = n_features
            self.x_rotations = np.zeros((n_features, self.n_components))
            self.eign_values = np.zeros((self.n_components))
            self.incremental_mean = 1


        for j in range(0, n_samples):
            self.n = self.n + 1
            u = X[j]

            old_mean = (self.n-1)/self.n*self.incremental_mean
            new_mean = 1/self.n*u
            self.incremental_mean = old_mean+new_mean

            if self.n == 1:
                self.x_rotations[:, 0] = u
                self.sum_x = u
            else:
                u = u - self.incremental_mean
                self.sum_x = self.sum_x + u

            k = min(self.n, self.n_components)
            for i in range(1, k+1):
                if i == self.n:
                    self.x_rotations[:, i - 1] = u
                else:
                    w1, w2 = (self.n-1-self.amnesic)/self.n, (self.n+self.amnesic)/self.n
                    v_norm = self.normalize(self.x_rotations[:, i-1])
                    v_norm = np.expand_dims(v_norm, axis=1)
                    self.x_rotations[:, i - 1] = w1 * self.x_rotations[:, i - 1] + w2*u*np.dot(u.T, v_norm)[0]

                    v_norm = self.normalize(self.x_rotations[:, i-1])
                    v_norm = np.expand_dims(v_norm, axis=1)
                    u = u - (np.dot(u.T, v_norm)*v_norm)[:, 0]
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        X -= self.incremental_mean

        w_rotation = np.zeros(self.x_rotations.shape)

        for c in range(0, self.n_components):
            w_rotation[:, c] = self.normalize(self.x_rotations[:, c])

        return np.dot(X, w_rotation)