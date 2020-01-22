from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, SCORERS
import pandas as pd
import numpy as np


def balanced_accuracy(y_true, y_pred):
    """Default scoring function: balanced accuracy.

    Balanced accuracy computes each class' accuracy on a per-class basis using a
    one-vs-rest encoding, then computes an unweighted average of the class accuracies.

    Parameters
    ----------
    y_true: numpy.ndarray {n_samples}
        True class labels
    y_pred: numpy.ndarray {n_samples}
        Predicted class labels by the estimator

    Returns
    -------
    fitness: float
        Returns a float value indicating the individual's balanced accuracy
        0.5 is as good as chance, and 1.0 is perfect predictive accuracy
    """
    all_classes = list(set(np.append(y_true, y_pred)))
    all_class_accuracies = []
    for this_class in all_classes:
        this_class_sensitivity = 0.
        this_class_specificity = 0.
        if sum(y_true == this_class) != 0:
            this_class_sensitivity = \
                float(sum((y_pred == this_class) & (y_true == this_class))) /\
                float(sum((y_true == this_class)))

            this_class_specificity = \
                float(sum((y_pred != this_class) & (y_true != this_class))) /\
                float(sum((y_true != this_class)))

        this_class_accuracy = (this_class_sensitivity + this_class_specificity) / 2.
        all_class_accuracies.append(this_class_accuracy)

    return np.mean(all_class_accuracies)


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
        if isinstance(Y_hat, pd.Series):
            Y_hat = Y_hat.values

        # Y_hat = Y_hat.astype(Y.dtype)
        if not self.balanced:
            score = accuracy_score(Y, Y_hat)
        else:
            score = balanced_accuracy(Y, Y_hat)

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



