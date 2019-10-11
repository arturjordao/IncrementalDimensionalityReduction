"""Online Partial Least Squares (OLPLS)."""

# Author: Artur Jordao <arturjlcorreia@gmail.com>
#         Artur Jordao

import numpy as np
from scipy import linalg

from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
import copy

class OLPLS(BaseEstimator):
    """Online Partial Least Squares (OLPLS).

    Parameters
    ----------
    n_components : int or None, (default=None)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    References
    An online NIPALS algorithm for Partial Least Squares (OLPLS)
    """

    def __init__(self, n_components=10, copy=True, amnesic=0.9999, mu=1e-4):
        self.__name__ = 'Online Partial Least Squares (OLPLS)'
        self.n_components = n_components
        self.amnesic = amnesic
        self.mu = mu
        self.copy = copy
        self.W = None
        self.P = None
        self.C = None
        self.S = None
        self.n = 0

    def fit(self, X, Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy, ensure_2d=False)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if np.unique(Y).shape[0] == 2:
            Y[np.where(Y == 0)[0]] = -1

        n_samples, n_features = X.shape

        if self.n == 0:
            x = np.expand_dims(X[0], 0)
            y = np.expand_dims(Y[0], 0)

            self.W = np.zeros((n_features, self.n_components))
            self.P = np.zeros((n_features, self.n_components))
            self.C = np.zeros((1, self.n_components))
            self.S = np.zeros((n_features, self.n_components))

            for c in range(0, self.n_components):
                self.S[:, c] = (x.T * y)[:, 0]
                Suse = self.S[:, c]
                wtemp = Suse
                wtemp = wtemp / np.linalg.norm(wtemp)
                self.W[:, c] = wtemp

                wtemp = wtemp + self.mu * (np.dot(np.dot(np.dot(np.dot(-wtemp.T, Suse), Suse.T), wtemp), wtemp)) / np.dot(wtemp.T, wtemp)**2 + np.dot(np.dot(Suse, Suse.T), wtemp)/np.dot(wtemp.T, wtemp)
                wtemp = wtemp / np.linalg.norm(wtemp)
                self.W[:, c] = wtemp

                t = np.dot(x, self.W[:, c])

                pn = np.dot(x.T, t)/np.dot(t.T, t)
                self.P[:, c] = 0.5*pn

                cn = np.dot(y.T, t)/np.dot(t.T, t)
                self.C[:, c] = 0.5*cn

                x = x - t*self.P[:, c]
                y = y - t*self.C[:, c]

                self.P[:, c] = 0.6 * pn
                self.C[:, c] = 0.6 * cn

        if self.n == 0:
            begin = 1
        else:
            begin = 0

        for i in range(begin, n_samples):

            x = np.expand_dims(X[i], 0)
            y = np.expand_dims(Y[i], 0)

            self.S[:, 1] = self.amnesic*self.S[:, 1] + (1-self.amnesic) * (x.T * y)[:, 0]
            Suse = self.S[:, 0]
            wtemp = self.W[:, 0]

            eigval = np.dot(np.dot(np.dot(wtemp.T, Suse), Suse.T), wtemp) / np.dot(wtemp.T, wtemp)
            lagrange = (eigval - eigval/np.linalg.norm(wtemp))

            a = np.dot(np.dot(np.dot(np.dot(-wtemp.T, Suse), Suse.T), wtemp), wtemp)
            a = a / (np.dot(wtemp, wtemp)**2)
            b = np.dot(np.dot(Suse, Suse.T), wtemp)
            b = b / (np.dot(wtemp.T, wtemp))
            self.W[:, 0] = wtemp + self.mu * (a + b - lagrange*wtemp)

            t = np.dot(x, self.W[:, 0])

            xhat = t*self.P[:, 0]
            ep1 = x-xhat
            self.P[:, 0] = self.P[:, 0] + self.mu * ep1 * t

            yhat = t*self.C[:, 0]
            ec1 = y - yhat
            self.C[:, 0] = self.C[:, 0] + self.mu * ec1 * t

            x = x - t*self.P[:, 0]
            y = y - t*self.C[:, 0]

            for c in range(1, self.n_components):
                self.S[:, c] = self.amnesic*self.S[:, c] + (1-self.amnesic)*(x.T * y)[:, 0]
                Suse = self.S[:, c]
                wtemp = self.W[:, c]

                eigval = np.dot(np.dot(np.dot(wtemp.T, Suse), Suse.T), wtemp) / (np.dot(wtemp.T, wtemp))
                lagrange = (eigval - eigval/np.linalg.norm(wtemp))
                a = np.dot(np.dot(np.dot(np.dot(-wtemp.T, Suse), Suse.T), wtemp), wtemp)
                a = a / (np.dot(wtemp, wtemp) ** 2)
                b = np.dot(np.dot(Suse, Suse.T), wtemp)
                b = b / (np.dot(wtemp.T, wtemp))
                self.W[:, c] = wtemp + self.mu*(a+b-lagrange*wtemp)

                t = np.dot(x, self.W[:, c])

                xhat = t*self.P[:, c]
                ep1 = x-xhat
                self.P[:, c] = self.P[:, c] + self.mu * ep1 * t

                yhat = t*self.C[:, c]
                ec1 = y-yhat
                self.C[:, c] = self.C[:, c] + self.mu * ec1 * t

                x = x - t*self.P[:, c]
                y = y - t*self.C[:, c]


            self.n = self.n + 1

        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        return np.dot(X, self.W)