from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics.classification import accuracy_score
from sklearn.utils import gen_batches
import numpy as np
import h5py

from CCIPCA import CCIPCA
from SGDPLS import SGDPLS
from IPLS import IPLS
from CIPLS import CIPLS
from OLPLS import OLPLS

def projection_method(method='CIPLS', n_components=2):
    if method == 'CCIPCA':
        return CCIPCA(n_components=n_components)

    if method == 'IPLS':
        return IPLS(n_components=n_components)

    if method == 'SGDPLS':
        return SGDPLS(n_components=n_components)

    if method == 'CIPLS':
        return CIPLS(n_components=n_components)

    if method == 'OLPLS':
        return OLPLS(n_components=n_components)

if __name__ == '__main__':
    np.random.seed(12227)

    method = 'CIPLS' # Options are: 'CCIPCA', 'IPLS', 'SGDPLS', 'CIPLS' or 'OLPLS'
    n_components = 2
    memory_restricted = True
    batch_size = 128 #Required only if memory_restricted=True

    if memory_restricted:
        print('Method:[{}] #Components[{}] Mode:[Memory Restricted]'.format(method, n_components))
    else:
        print('Method:[{}] #Components[{}] Mode:[Normal]'.format(method, n_components))

    tmp = h5py.File('multiclass_data.h5')
    X_train, y_train = tmp['X_train'], tmp['y_train']
    X_test, y_test = tmp['X_test'], tmp['y_test']

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    categories = y_train.shape[1]

    if method == 'CCIPCA':
        categories = 1

    y_train, y_test = y_train[0:n_train], y_test[0:n_test]

    #If memory_restricted=True the samples will be load in batch.
    #Otherwise, we load all samples into memory
    if memory_restricted == False:
        X_train, y_train = X_train[0:n_train], y_train[0:n_train]
        X_test, y_test = X_test[0:n_test], y_test[0:n_test]
        batch_size = max(n_train, n_test)
        tmp.close()

    X_train_latent = np.zeros((n_train, n_components*categories))
    X_test_latent = np.zeros((n_test, n_components*categories))

    for category in range(0, categories):

        dm = projection_method(method=method, n_components=n_components)

        y_tmp = np.zeros((n_train, 1))
        y_tmp[np.argmax(y_train, axis=1) == category] = 1
        y_tmp[np.argmax(y_train, axis=1) != category] = -1

        train_latent = np.zeros((n_train, n_components))
        test_latent = np.zeros((n_test, n_components))

        for batch in gen_batches(n_train, batch_size):
            dm.fit(X_train[batch], y_tmp[batch])

        for batch in gen_batches(n_train, batch_size):
            train_latent[batch] = dm.transform(X_train[batch])

        for batch in gen_batches(n_test, batch_size):
            test_latent[batch] = dm.transform(X_test[batch])

        idx = category*n_components
        X_train_latent[:, idx:idx+n_components] = train_latent
        X_test_latent[:, idx:idx + n_components] = test_latent

        print('Projection for category [{}] done'.format(category))

    X_train, X_test = X_train_latent, X_test_latent

    model = LinearSVC(random_state=0)
    model = OneVsRestClassifier(model).fit(X_train, y_train)
    y_pred = model.decision_function(X_test)


    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print('Accuracy [{:.4f}]'.format(acc, n_components))