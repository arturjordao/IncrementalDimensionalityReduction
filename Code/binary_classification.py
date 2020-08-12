from sklearn.metrics.classification import accuracy_score
from sklearn.utils import gen_batches
import numpy as np
import h5py
from CIPLS import CIPLS
from sklearn import svm

if __name__ == '__main__':
    np.random.seed(12227)

    classifier = svm.LinearSVC(C=0.1, random_state=1227)
    method = 'CIPLS'
    n_components = 2
    memory_restricted = False
    batch_size = 128  # Required only if memory_restricted=True

    if memory_restricted:
        print('Method:[{}] #Components[{}] Mode:[Memory Restricted]'.format(method, n_components))
    else:
        print('Method:[{}] #Components[{}] Mode:[Normal]'.format(method, n_components))

    dm = CIPLS(n_components=n_components)

    tmp = h5py.File('binary_data.h5', 'r')
    X_train, y_train = tmp['X_train'], tmp['y_train']
    X_test, y_test = tmp['X_test'], tmp['y_test']

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    y_train, y_test = y_train[0:n_train], y_test[0:n_test]

    # If memory_sensitive=True the samples will be load in batch.
    # Otherwise, we load all samples into memory
    if memory_restricted == False:
        X_train, y_train = X_train[0:n_train], y_train[0:n_train]
        X_test, y_test = X_test[0:n_test], y_test[0:n_test]
        batch_size = max(n_train, n_test)
        tmp.close()

    X_train_latent = np.zeros((n_train, n_components))
    X_test_latent = np.zeros((n_test, n_components))

    for batch in gen_batches(n_train, batch_size):
        dm.fit(X_train[batch], y_train[batch])

    for batch in gen_batches(n_train, batch_size):
        X_train_latent[batch] = dm.transform(X_train[batch])

    for batch in gen_batches(n_test, batch_size):
        X_test_latent[batch] = dm.transform(X_test[batch])

    classifier.fit(X_train_latent, y_train[0:n_train])
    y_pred = classifier.predict(X_test_latent)

    acc = accuracy_score(y_test[0:n_test], y_pred)
    print('Accuracy [{:.4f}]'.format(acc, n_components))