"""Incremental partial least squares (IPLS)"""

import numpy as np
from scipy import linalg

from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
import copy

class IPLS(BaseEstimator):
    """Incremental Partial Least Squares (IPLS).

    Parameters
    ----------
    n_components : int or None, (default=None)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    L : Number of principal components

    amnesic :  Value to dynamically determine the retaining rate of
    the old and new data.

    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    References
    Incremental partial least squares analysis of big streaming data
    """

    def __init__(self, n_components=10, L=20, amnesic=2, copy=True):
        self.__name__ = 'Incremental Projection on Latent Space (IPLS)'
        self.n_components = n_components
        self.L = L
        self.amnesic = amnesic
        self.n = 0
        self.pcv = np.zeros((L, 1))
        self.copy = copy

        self.v = None
        self.sum_x = None
        self.sum_y = None
        self.n_features = None
        self.x_rotations = None
        self.eign_values = None
        self.x_mean = None

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
            self.v = np.zeros((self.n_components, n_features))
            self.pcv = np.zeros((self.L, n_features))
            self.n_features = n_features
            self.x_rotations = np.zeros((self.n_components, n_features))
            self.eign_values = np.zeros((self.n_components))

        for j in range(0, n_samples):
            self.n = self.n + 1
            u = X[j]
            l = Y[j]

            if self.n == 1:
                self.sum_x = u
                self.sum_y = l
                self.pcv[0] = u
            else:
                # IPLS
                old_mean = 1/(self.n-1)*self.sum_x
                self.sum_x = self.sum_x+u
                mean_x = 1/self.n*self.sum_x
                u = u - mean_x
                delta_x = mean_x-old_mean

                self.v[0] = self.v[0]-delta_x*self.sum_y
                self.v[0] = self.v[0] + (u*l)
                self.sum_y = self.sum_y+l

                #CCIPCA
                if self.n_components > 1:
                    k = min(self.n, self.L)
                    for i in range(1, k+1):
                        if i == self.n:
                            self.pcv[i-1] = u
                        else:
                            v = self.pcv[i-1]
                            w = (self.n-1-self.amnesic)/self.n
                            t2 = u * (w*np.inner(v, u)/ (np.linalg.norm(v)+np.finfo(float).eps))
                            v = v*(1-w)+t2
                            self.pcv[i-1] = v
                            v_product = np.inner(v, v)
                            if v_product != 0:
                                u = u - v*(np.inner(v, u)/v_product)
        return self

    def all_components(self):
        self.x_mean = 1 / self.n * self.sum_x
        xy = self.normalize(self.v[0])
        self.x_rotations[0] = self.normalize(self.v[0])
        pcv = copy.deepcopy(self.pcv)

        if self.n_components > 1:
            pnorms = np.zeros((self.L, 1))
            sum_norm = 0
            for j in range(0, self.L):
                pnorms[j] = np.linalg.norm(pcv[j])
                sum_norm = sum_norm + pnorms[j]
                pcv[j] = self.normalize(pcv[j])

            for j in range(0, self.L):
                pnorms[j] /= sum_norm*0.05

            for i in range(1, self.n_components):
                v_cur = np.zeros(self.n_features)

                for j in range(0, self.L):
                    v_cur = v_cur + pcv[j]*(np.inner(pcv[j], xy)*pnorms[j])

                xy = self.normalize(v_cur)
                for j in range(0, i):
                    tmp_v = copy.deepcopy(self.x_rotations[j])
                    v_cur = v_cur - tmp_v*np.inner(tmp_v, v_cur)
                    ti = np.inner(v_cur, v_cur)
                    if ti == 0:
                        v_cur = np.zeros((self.n_features, 1))
                        break
                    elif ti < 1e-1 or ti > 1e10:
                        v_cur = v_cur*1/np.sqrt(ti)

                if np.inner(v_cur, v_cur) == 0:
                    break

                self.v[i] = v_cur
                self.x_rotations[i] = self.normalize(self.v[i])
                self.eign_values[i] = np.linalg.norm(self.v[i])

        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        #Find the n-1 components and normalize the first component
        self.all_components()

        mean = 1 / self.n * self.sum_x
        X -= mean
        return np.dot(X, self.x_rotations.T)
