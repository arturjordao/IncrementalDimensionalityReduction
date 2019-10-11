import numpy as np
from sklearn.utils import gen_batches
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import keras

if __name__ == '__main__':
    np.random.seed(12227)

    X, y = make_classification(n_samples=1500, n_features=400, n_classes=3, n_clusters_per_class=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    import h5py
    h5f = h5py.File('multiclass_data.h5', 'w')
    #h5f.create_group('data')
    h5f['X_train'] = X_train
    h5f['y_train'] = y_train
    h5f['X_test'] = X_test
    h5f['y_test'] = y_test
    h5f.close()