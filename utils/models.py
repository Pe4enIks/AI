import numpy as np
from torch import sigmoid
from tqdm import tqdm


class SVM():
    def __init__(self, lr, alpha, epochs):
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.W = None
        self.train_loss = None
        self.test_loss = None

    def fit(self, train, test):
        X_train, y_train = train[0], train[1]
        X_test, y_test = test[0], test[1]
        X_train = self.add_bias_feature(X_train)
        X_test = self.add_bias_feature(X_test)

        limit = np.sqrt(6.0 / X_train.shape[1])
        self.W = np.random.uniform(-limit, limit, size=X_train.shape[1])

        train_loss_epoch = []
        test_loss_epoch = []

        tqdm_iter = tqdm(range(self.epochs))

        for _ in tqdm_iter:
            train_loss = 0
            test_loss = 0

            for x, y in zip(X_train, y_train):
                margin = y * np.dot(self.W, x)
                if margin >= 1:
                    self.W = self.W + self.lr * self.alpha * self.W
                    train_loss += self.soft_margin_loss(x, y)
                else:
                    self.W = self.W + self.lr * (y * x - self.alpha * self.W)
                    train_loss += self.soft_margin_loss(x, y)

            for x, y in zip(X_test, y_test):
                test_loss += self.soft_margin_loss(x, y)

            train_loss /= len(y_train)
            test_loss /= len(y_test)

            train_loss_epoch.append(train_loss)
            test_loss_epoch.append(test_loss)

            tqdm_iter.set_postfix({'train loss:': train_loss, 'test loss': test_loss}, refresh=True)
            tqdm_iter.refresh()

        self.train_loss = np.array(train_loss_epoch)
        self.test_loss = np.array(test_loss_epoch)

    def predict(self, X):
        y_pred = []
        X_extended = self.add_bias_feature(X)
        for x in X_extended:
            y_pred.append(np.sign(np.dot(self.W, x)))
        return np.array(y_pred)

    def hinge_loss(self, X, y):
        return max(0, 1 - y * np.dot(X, self.W))

    def soft_margin_loss(self, X, y):
        return self.hinge_loss(X, y) + self.alpha * np.dot(self.W, self.W)

    def add_bias_feature(self, X):
        X_extended = np.zeros((X.shape[0], X.shape[1] + 1))
        X_extended[:, :-1] = X
        X_extended[:, -1] = int(1)
        return X_extended


class LogisticRegression():
    def __init__(self, lr, epochs):
        self.epochs = epochs
        self.lr = lr
        self.W = None
        self.train_loss = None
        self.test_loss = None

    def fit(self, train, test):
        X_train, y_train = train[0], train[1]
        X_test, y_test = test[0], test[1]
        X_train = self.add_bias_feature(X_train)
        X_test = self.add_bias_feature(X_test)

        y_train = (y_train + 1.0) / 2.0
        y_test = (y_test + 1.0) / 2.0

        limit = np.sqrt(6.0 / X_train.shape[1])
        self.W = np.random.uniform(-limit, limit, size=X_train.shape[1])

        train_loss_epoch = []
        test_loss_epoch = []

        tqdm_iter = tqdm(range(self.epochs))

        for _ in tqdm_iter:
            train_loss = 0
            test_loss = 0

            for x, y in zip(X_train, y_train):
                reg = np.dot(self.W, x)
                diff = reg - y
                self.W = self.W - self.lr * x * diff
                train_loss += self.logloss(reg, y)

            for x, y in zip(X_test, y_test):
                reg = np.dot(self.W, x)
                test_loss += self.logloss(reg, y)

            train_loss /= len(y_train)
            test_loss /= len(y_test)

            train_loss_epoch.append(train_loss)
            test_loss_epoch.append(test_loss)

            tqdm_iter.set_postfix({'train loss:': train_loss, 'test loss': test_loss}, refresh=True)
            tqdm_iter.refresh()

        self.train_loss = np.array(train_loss_epoch)
        self.test_loss = np.array(test_loss_epoch)

    def predict(self, X):
        y_pred = []
        X_extended = self.add_bias_feature(X)
        for x in X_extended:
            reg = np.dot(self.W, x)
            pred = (self.sigmoid(reg) > 0.5).astype(np.int8) * 2 - 1
            y_pred.append(pred)
        return np.array(y_pred)

    def sigmoid(self, x):
        return (1.0 / (1.0 + np.exp(-x)))

    def logloss(self, reg, y):
        log1 = np.log(self.sigmoid(reg) + 1e-9)
        log2 = np.log(1.0 - self.sigmoid(reg) + 1e-9)
        return -y * log1 - (1.0 - y) * log2

    def add_bias_feature(self, X):
        X_extended = np.zeros((X.shape[0], X.shape[1] + 1))
        X_extended[:, :-1] = X
        X_extended[:, -1] = int(1)
        return X_extended
