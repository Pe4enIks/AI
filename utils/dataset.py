from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
import numpy as np


def get_dataset(dataset, test_size, random_state):
    if dataset == 'wine':
        data = load_wine()
    elif dataset == 'iris':
        data = load_iris()
    else:
        raise ValueError(f'no dataset with name {dataset}')

    X, y = data.data, data.target
    y = (y > 0).astype(np.int8) * 2 - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return (X_train, y_train), (X_test, y_test)
