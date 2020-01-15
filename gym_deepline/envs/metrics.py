from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score
from sklearn.model_selection import KFold
import numpy as np


class Metric:
    def __init__(self, name):
        self.name = name
        self.decription = ""

    def evaluate(self, Y, Y_hat):
        pass


class Accuracy(Metric):
    def __init__(self, balanced=False):
        super(Accuracy, self).__init__(name="accuracy")
        self.balanced = balanced

    def evaluate(self,  Y, Y_hat):
        import pandas as pd
        if isinstance(Y_hat, pd.Series):
            Y_hat = Y_hat.values
        Y_hat = Y_hat.astype(Y.dtype)
        if not self.balanced:
            score = accuracy_score(Y, Y_hat)
        else:
            score = balanced_accuracy_score(Y, Y_hat)

        return score

    def cv_evaluate(self, X, Y, cls):
        kf = KFold(n_splits=5, shuffle=False, random_state=42)
        scores = []
        for train_index, test_index in kf.split(X):
            try:
                X_train, X_test = X.copy(deep=True).loc[train_index].reset_index(inplace=False), X.copy(deep=True).loc[test_index].reset_index(inplace=False)
                y_train, y_test = Y[train_index], Y[test_index]
                cls.fit(X_train, y_train)
                cls.produce(X_test)
                y_hat = cls.produce_outputs['predictions']
                scores.append(self.evaluate(y_test, y_hat))
            except Exception as e:
                continue
        return np.mean(np.array(scores))


class MSE(Metric):
    def __init__(self):
        super(MSE, self).__init__(name="MSE")

    def evaluate(self,  Y, Y_hat):
        score = mean_squared_error(Y, Y_hat)

        return score



