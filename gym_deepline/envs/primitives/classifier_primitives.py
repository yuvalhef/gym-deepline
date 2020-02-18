from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from gym_deepline.envs.Primitives import primitive
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,\
GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model.logistic import LogisticRegression, LogisticRegressionCV
from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import RUSBoostClassifier
from lightgbm.sklearn import LGBMClassifier
from copy import deepcopy
import pandas as pd
import numpy as np
np.random.seed(1)


def replace_bad_str(str_list):
    for i in range(len(str_list)):
        col = str_list[i]
        col = col.replace('[', 'abaqa')
        col = col.replace(']', 'bebab')
        col = col.replace('<', 'cfckc')
        col = col.replace('>', 'dmdad')
        str_list[i] = col
    return str_list


def handle_data(data):
    if len(data) == 1:
        new_data = deepcopy(data[0])
    else:
        concatenated_df = pd.DataFrame()
        for d_input in data.values():
            df2 = d_input['X'][d_input['X'].columns.difference(concatenated_df.columns)].copy()
            concatenated_df = pd.concat([concatenated_df.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        new_data = deepcopy(data[0])
        new_data['X'] = concatenated_df.infer_objects()

    cols = list(new_data['X'].columns)
    new_data['X'].columns = replace_bad_str(cols)
    new_data['X'] = new_data['X'].loc[:, ~new_data['X'].columns.duplicated()]
    return new_data


class RandomForestClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(RandomForestClassifierPrim, self).__init__(name='RF_classifier')
        self.id = 55
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "A random forest classifier. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default)."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=5)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class AdaBoostClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(AdaBoostClassifierPrim, self).__init__(name='AdaBoostClassifier')
        self.id = 56
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "An AdaBoost classifier. An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases. This class implements the algorithm known as AdaBoost-SAMME."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = AdaBoostClassifier(random_state=self.random_state)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class BaggingClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(BaggingClassifierPrim, self).__init__(name='BaggingClassifier')
        self.id = 57
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "A Bagging classifier. A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it. This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting. If samples are drawn with replacement, then the method is known as Bagging. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = BaggingClassifier(random_state=self.random_state, n_jobs=5)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')


    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class BernoulliNBClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(BernoulliNBClassifierPrim, self).__init__(name='BernoulliNBClassifier')
        self.id = 58
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Naive Bayes classifier for multivariate Bernoulli models. Like MultinomialNB, this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = BernoulliNB()
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class ComplementNBClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(ComplementNBClassifierPrim, self).__init__(name='ComplementNBClassifier')
        self.id = 59
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "The Complement Naive Bayes classifier described in Rennie et al. (2003). The Complement Naive Bayes classifier was designed to correct the “severe assumptions” made by the standard Multinomial Naive Bayes classifier. It is particularly suited for imbalanced data sets."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = ComplementNB()
        self.accept_type = 'd'

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class DecisionTreeClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(DecisionTreeClassifierPrim, self).__init__(name='DecisionTreeClassifier')
        self.id = 60
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "A decision tree classifier."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = DecisionTreeClassifier(random_state=random_state)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class ExtraTreesClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(ExtraTreesClassifierPrim, self).__init__(name='ExtraTreesClassifier')
        self.id = 61
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "An extra-trees classifier. This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = ExtraTreesClassifier(random_state=random_state, n_jobs=5)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class GaussianNBClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(GaussianNBClassifierPrim, self).__init__(name='GaussianNBClassifier')
        self.id = 62
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Gaussian Naive Bayes (GaussianNB). Can perform online updates to model parameters via partial_fit method."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = GaussianNB()
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class GaussianProcessClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(GaussianProcessClassifierPrim, self).__init__(name='GaussianProcessClassifierPrim')
        self.id = 63
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Gaussian process classification (GPC) based on Laplace approximation. The implementation is based on Algorithm 3.1, 3.2, and 5.1 of Gaussian Processes for Machine Learning (GPML) by Rasmussen and Williams. Internally, the Laplace approximation is used for approximating the non-Gaussian posterior by a Gaussian. Currently, the implementation is restricted to using the logistic link function. For multi-class classification, several binary one-versus rest classifiers are fitted. Note that this class thus does not implement a true multi-class Laplace approximation."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = GaussianProcessClassifier()
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class GradientBoostingClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(GradientBoostingClassifierPrim, self).__init__(name='GradientBoostingClassifier')
        self.id = 64
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Gradient Boosting for classification. GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = GradientBoostingClassifier(random_state=random_state)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class KNeighborsClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(KNeighborsClassifierPrim, self).__init__(name='KNeighborsClassifierPrim')
        self.id = 65
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Classifier implementing the k-nearest neighbors vote."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = KNeighborsClassifier(n_jobs=5)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class LinearDiscriminantAnalysisPrim(primitive):
    def __init__(self, random_state=0):
        super(LinearDiscriminantAnalysisPrim, self).__init__(name='LinearDiscriminantAnalysisPrim')
        self.id = 66
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Linear Discriminant Analysis. A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix. The fitted model can also be used to reduce the dimensionality of the input by projecting it to the most discriminative directions."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = LinearDiscriminantAnalysis()
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class LinearSVCPrim(primitive):
    def __init__(self, random_state=0):
        super(LinearSVCPrim, self).__init__(name='LinearSVC')
        self.id = 67
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Linear Support Vector Classification. Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples. This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = LinearSVC(random_state=random_state)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        classes = self.model.classes_.tolist()
        dummy_predictions = pd.get_dummies(output['predictions'])
        if not classes == list(dummy_predictions.columns):
            diff_classes = list(set(classes) - set(list(dummy_predictions.columns)))
            for cls in diff_classes:
                idx = classes.index(cls)
                if idx < dummy_predictions.shape[1] - 1:
                    dummy_predictions.insert(idx, cls, np.zeros(dummy_predictions.shape[0]))
                else:
                    dummy_predictions[cls] = np.zeros(dummy_predictions.shape[0])
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(dummy_predictions.values, columns=cols).reset_index(drop=True).infer_objects()
        output['proba_predictions'] = pd.DataFrame(dummy_predictions.values, columns=classes).reset_index(drop=True).infer_objects()
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class LogisticRegressionPrim(primitive):
    def __init__(self, random_state=0):
        super(LogisticRegressionPrim, self).__init__(name='LogisticRegression')
        self.id = 68
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Logistic Regression (aka logit, MaxEnt) classifier. In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross- entropy loss if the ‘multi_class’ option is set to ‘multinomial’. (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’ and ‘newton-cg’ solvers.) This class implements regularized logistic regression using the ‘liblinear’ library, ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers. It can handle both dense and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance; any other input format will be converted (and copied). The ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers support only L2 regularization with primal formulation. The ‘liblinear’ solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = LogisticRegression(random_state=random_state, n_jobs=5, multi_class='auto')
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class LogisticRegressionCVPrim(primitive):
    def __init__(self, random_state=0):
        super(LogisticRegressionCVPrim, self).__init__(name='LogisticRegressionCV')
        self.id = 69
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Logistic Regression CV (aka logit, MaxEnt) classifier. See glossary entry for cross-validation estimator. This class implements logistic regression using liblinear, newton-cg, sag of lbfgs optimizer. The newton-cg, sag and lbfgs solvers support only L2 regularization with primal formulation. The liblinear solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty. For the grid of Cs values (that are set by default to be ten values in a logarithmic scale between 1e-4 and 1e4), the best hyperparameter is selected by the cross-validator StratifiedKFold, but it can be changed using the cv parameter. In the case of newton-cg and lbfgs solvers, we warm start along the path i.e guess the initial coefficients of the present fit to be the coefficients got after convergence in the previous fit, so it is supposed to be faster for high-dimensional dense data. For a multiclass problem, the hyperparameters for each class are computed using the best scores got by doing a one-vs-rest in parallel across all folds and classes. Hence this is not the true multinomial loss."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = LogisticRegressionCV(random_state=random_state, n_jobs=5, multi_class='auto')
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class MultinomialNBPrim(primitive):
    def __init__(self, random_state=0):
        super(MultinomialNBPrim, self).__init__(name='MultinomialNB')
        self.id = 70
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Naive Bayes classifier for multinomial models. The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = MultinomialNB()
        self.accept_type = 'd'

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class NearestCentroidPrim(primitive):
    def __init__(self, random_state=0):
        super(NearestCentroidPrim, self).__init__(name='NearestCentroid')
        self.id = 71
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Nearest centroid classifier. Each class is represented by its centroid, with test samples classified to the class with the nearest centroid."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = NearestCentroid()
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        classes = self.model.classes_.tolist()
        dummy_predictions = pd.get_dummies(output['predictions'])
        if not classes == list(dummy_predictions.columns):
            diff_classes = list(set(classes) - set(list(dummy_predictions.columns)))
            for cls in diff_classes:
                idx = classes.index(cls)
                if idx < dummy_predictions.shape[1] - 1:
                    dummy_predictions.insert(idx, cls, np.zeros(dummy_predictions.shape[0]))
                else:
                    dummy_predictions[cls] = np.zeros(dummy_predictions.shape[0])
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(dummy_predictions.values, columns=cols).reset_index(drop=True).infer_objects()
        output['proba_predictions'] = pd.DataFrame(dummy_predictions.values, columns=classes).reset_index(drop=True).infer_objects()
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


# class NuSVCPrim(primitive):
#     def __init__(self, random_state=0):
#         super(NuSVCPrim, self).__init__(name='NuSVCPrim')
#         self.hyperparams = []
#         self.type = 'Classifier'
#         self.description = "Nu-Support Vector Classification. Similar to SVC but uses a parameter to control the number of support vectors. The implementation is based on libsvm."
#         self.hyperparams_run = {'default': True}
#         self.random_state = random_state
#         self.model = NuSVC(random_state=random_state)
#
#     def can_accept(self, data):
#         # data = handle_data(data)
#         if data['X'].empty:
#             return False
#         cols = data['X']
#         num_cols = data['X']._get_numeric_data().columns
#         cat_cols = list(set(cols) - set(num_cols))
#         if not data['learning_job'].task == 'Classification' or data['X'].isnull().any().any():
#             return False
#         elif not len(cat_cols) == 0:
#             return False
#         return True
#
#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True
#
#     def fit(self, data):
#         data = handle_data(data)
#         self.model.fit(data['X'], data['Y'])
#
#     def produce(self, data):
#         output = handle_data(data)
#         output['predictions'] = self.model.predict(output['X'])
#         classes = self.model.classes_.tolist()
#         dummy_predictions = pd.get_dummies(output['predictions'])
#         if not classes == list(dummy_predictions.columns):
#             diff_classes = list(set(classes) - set(list(dummy_predictions.columns)))
#             for cls in diff_classes:
#                 idx = classes.index(cls)
#                 if idx < dummy_predictions.shape[1] - 1:
#                     dummy_predictions.insert(idx, cls, np.zeros(dummy_predictions.shape[0]))
#                 else:
#                     dummy_predictions[cls] = np.zeros(dummy_predictions.shape[0])
#         cols = ["{}_{}Pred".format(c, self.name) for c in classes]
#         output['X'] = pd.DataFrame(dummy_predictions.values, columns=cols)
#         output['Y'] = output['Y']
#         final_output = {0: output}
#         return final_output


class PassiveAggressiveClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(PassiveAggressiveClassifierPrim, self).__init__(name='PassiveAggressiveClassifier')
        self.id = 72
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Passive Aggressive Classifier"
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = PassiveAggressiveClassifier(random_state=random_state, n_jobs=5)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        classes = self.model.classes_.tolist()
        dummy_predictions = pd.get_dummies(output['predictions'])
        if not classes == list(dummy_predictions.columns):
            diff_classes = list(set(classes) - set(list(dummy_predictions.columns)))
            for cls in diff_classes:
                idx = classes.index(cls)
                if idx < dummy_predictions.shape[1] - 1:
                    dummy_predictions.insert(idx, cls, np.zeros(dummy_predictions.shape[0]))
                else:
                    dummy_predictions[cls] = np.zeros(dummy_predictions.shape[0])
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(dummy_predictions.values, columns=cols).reset_index(drop=True).infer_objects()
        output['proba_predictions'] = pd.DataFrame(dummy_predictions.values, columns=classes).reset_index(drop=True).infer_objects()
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class QuadraticDiscriminantAnalysisPrim(primitive):
    def __init__(self, random_state=0):
        super(QuadraticDiscriminantAnalysisPrim, self).__init__(name='QuadraticDiscriminantAnalysis')
        self.id = 73
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Quadratic Discriminant Analysis. A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule. The model fits a Gaussian density to each class."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = QuadraticDiscriminantAnalysis()
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


# class RadiusNeighborsClassifierPrim(primitive):
#     def __init__(self, random_state=0):
#         super(RadiusNeighborsClassifierPrim, self).__init__(name='RadiusNeighborsClassifier')
#         self.hyperparams = []
#         self.type = 'Classifier'
#         self.description = "Classifier implementing a vote among neighbors within a given radius"
#         self.hyperparams_run = {'default': True}
#         self.random_state = random_state
#         self.model = RadiusNeighborsClassifier(n_jobs=5)
#
#     def can_accept(self, data):
#         # data = handle_data(data)
#         if data['X'].empty:
#             return False
#         cols = data['X']
#         num_cols = data['X']._get_numeric_data().columns
#         cat_cols = list(set(cols) - set(num_cols))
#         if not data['learning_job'].task == 'Classification' or data['X'].isnull().any().any():
#             return False
#         elif not len(cat_cols) == 0:
#             return False
#         return True
#
#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True
#
#     def fit(self, data):
#         data = handle_data(data)
#         self.model.fit(data['X'], data['Y'])
#
#     def produce(self, data):
#         output = handle_data(data)
#         output['predictions'] = self.model.predict(output['X'])
#         classes = self.model.classes_.tolist()
#         dummy_predictions = pd.get_dummies(output['predictions'])
#         if not classes == list(dummy_predictions.columns):
#             diff_classes = list(set(classes) - set(list(dummy_predictions.columns)))
#             for cls in diff_classes:
#                 idx = classes.index(cls)
#                 if idx < dummy_predictions.shape[1] - 1:
#                     dummy_predictions.insert(idx, cls, np.zeros(dummy_predictions.shape[0]))
#                 else:
#                     dummy_predictions[cls] = np.zeros(dummy_predictions.shape[0])
#         cols = ["{}_{}Pred".format(c, self.name) for c in classes]
#         output['X'] = pd.DataFrame(dummy_predictions.values, columns=cols)
#         output['Y'] = output['Y']
#         final_output = {0: output}
#         return final_output


class RidgeClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(RidgeClassifierPrim, self).__init__(name='RidgeClassifier')
        self.id = 74
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Classifier using Ridge regression."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = RidgeClassifier(random_state=random_state)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        classes = self.model.classes_.tolist()
        dummy_predictions = pd.get_dummies(output['predictions'])
        if not classes == list(dummy_predictions.columns):
            diff_classes = list(set(classes) - set(list(dummy_predictions.columns)))
            for cls in diff_classes:
                idx = classes.index(cls)
                if idx < dummy_predictions.shape[1] - 1:
                    dummy_predictions.insert(idx, cls, np.zeros(dummy_predictions.shape[0]))
                else:
                    dummy_predictions[cls] = np.zeros(dummy_predictions.shape[0])
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(dummy_predictions.values, columns=cols).reset_index(drop=True).infer_objects()
        output['proba_predictions'] = pd.DataFrame(dummy_predictions.values, columns=classes).reset_index(drop=True).infer_objects()
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class RidgeClassifierCVPrim(primitive):
    def __init__(self, random_state=0):
        super(RidgeClassifierCVPrim, self).__init__(name='RidgeClassifierCV')
        self.id = 75
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Ridge classifier with built-in cross-validation. By default, it performs Generalized Cross-Validation, which is a form of efficient Leave-One-Out cross-validation."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = RidgeClassifierCV()
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        classes = self.model.classes_.tolist()
        dummy_predictions = pd.get_dummies(output['predictions'])
        if not classes == list(dummy_predictions.columns):
            diff_classes = list(set(classes) - set(list(dummy_predictions.columns)))
            for cls in diff_classes:
                idx = classes.index(cls)
                if idx < dummy_predictions.shape[1] - 1:
                    dummy_predictions.insert(idx, cls, np.zeros(dummy_predictions.shape[0]))
                else:
                    dummy_predictions[cls] = np.zeros(dummy_predictions.shape[0])
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(dummy_predictions.values, columns=cols).reset_index(drop=True).infer_objects()
        output['proba_predictions'] = pd.DataFrame(dummy_predictions.values, columns=classes).reset_index(drop=True).infer_objects()
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class SGDClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(SGDClassifierPrim, self).__init__(name='SGDClassifier')
        self.id = 76
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Linear classifiers (SVM, logistic regression, a.o.) with SGD training. This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning, see the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance. This implementation works with data represented as dense or sparse arrays of floating point values for the features. The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM). The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = SGDClassifier(random_state=random_state, n_jobs=5, loss='log')
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class SVCPrim(primitive):
    def __init__(self, random_state=0):
        super(SVCPrim, self).__init__(name='SVC')
        self.id = 77
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "C-Support Vector Classification. The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples. The multiclass support is handled according to a one-vs-one scheme. For details on the precise mathematical formulation of the provided kernel functions and how gamma, coef0 and degree affect each other, see the corresponding section in the narrative documentation: Kernel functions."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = SVC(random_state=random_state, probability=True)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class XGBClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(XGBClassifierPrim, self).__init__(name='XGBClassifier')
        self.id = 78
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = XGBClassifier(random_state=random_state, n_jobs=5)
        self.accept_type = 'xgb'

    def can_accept(self, data):
        # data = handle_data(data)
        if data['X'].empty:
            return False
        cols = data['X']
        num_cols = data['X']._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if not data['learning_job'].task == 'Classification':
            return False
        elif not len(cat_cols) == 0:
            return False
        return True

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class BalancedRandomForestClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(BalancedRandomForestClassifierPrim, self).__init__(name='BalancedRandomForestClassifier')
        self.id = 79
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "A balanced random forest classifier. A balanced random forest randomly under-samples each boostrap sample to balance it."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = BalancedRandomForestClassifier(random_state=random_state, n_jobs=5)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class EasyEnsembleClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(EasyEnsembleClassifierPrim, self).__init__(name='EasyEnsembleClassifier')
        self.id = 80
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Bag of balanced boosted learners also known as EasyEnsemble. This algorithm is known as EasyEnsemble [1]. The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = EasyEnsembleClassifier(random_state=random_state, n_jobs=5)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class RUSBoostClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(RUSBoostClassifierPrim, self).__init__(name='RUSBoostClassifier')
        self.id = 81
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "Random under-sampling integrating in the learning of an AdaBoost classifier. During learning, the problem of class balancing is alleviated by random under-sampling the sample at each iteration of the boosting algorithm."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = RUSBoostClassifier(random_state=random_state)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


class LGBMClassifierPrim(primitive):
    def __init__(self, random_state=0):
        super(LGBMClassifierPrim, self).__init__(name='LGBMClassifier')
        self.id = 82
        self.hyperparams = []
        self.type = 'Classifier'
        self.description = "LightGBM is a gradient boosting framework that uses tree based learning algorithms."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = LGBMClassifier(random_state=random_state, n_jobs=5)
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['proba_predictions'] = self.model.predict_proba(output['X'])
        classes = list(self.model.classes_)
        cols = ["{}_{}Pred".format(c, self.name) for c in classes]
        output['X'] = pd.DataFrame(output['proba_predictions'], columns=cols)
        output['proba_predictions'] = pd.DataFrame(output['proba_predictions'], columns=classes)
        output['Y'] = output['Y']
        final_output = {0: output}
        return final_output


# class CalibratedClassifierCVPrim(primitive):
#     def __init__(self, random_state=0):
#         super(CalibratedClassifierCVPrim, self).__init__(name='CalibratedClassifierCV')
#         self.hyperparams = []
#         self.type = 'Classifier'
#         self.description = "Naive Bayes classifier for multivariate Bernoulli models. Like MultinomialNB, this classifier is suitable for discrete data. The difference is that while MultinomialNB works with occurrence counts, BernoulliNB is designed for binary/boolean features."
#         self.hyperparams_run = {'default': True}
#         self.random_state = random_state
#         self.model = CalibratedClassifierCV()
#
#     def can_accept(self, data):
#         # data = handle_data(data)
#         cols = data['X']
#         num_cols = data['X']._get_numeric_data().columns
#         cat_cols = list(set(cols) - set(num_cols))
#         if not data['learning_job'].task == 'Classification' or data['X'].isnull().any().any():
#             return False
#         elif not len(cat_cols) == 0:
#             return False
#         return True
#
#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True
#
#     def fit(self, data):
#         data = handle_data(data)
#         self.model.fit(data['X'], data['Y'])
#
#     def produce(self, data):
#         output = handle_data(data)
#         output['predictions'] = self.model.predict(output['X'])
#         output['proba_predictions'] = self.model.predict_proba(output['X'])
#         output['Y'] = data[0]['Y']
#         return output


# Semi-supervised? LabelPropagation, LabelSpreading
# NN? MLP Classifier, Perceptron