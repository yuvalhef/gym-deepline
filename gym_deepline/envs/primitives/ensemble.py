from gym_deepline.envs.Primitives import primitive
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestRegressor, \
    ExtraTreesClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
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
    if len(data) == 0:
        return None
    elif len(data) == 1:
        new_data = deepcopy(data[0])
    else:
        concatenated_df = pd.DataFrame()
        for d_input in data.values():
            if any('Pred' in s for s in list(d_input['X'].columns)):
                df2 = deepcopy(d_input['X'][d_input['X'].columns.difference(concatenated_df.columns)])
            else:
                df2 = deepcopy(d_input['X'])
            concatenated_df = pd.concat([concatenated_df.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        new_data = deepcopy(data[0])
        new_data['X'] = concatenated_df.infer_objects()
        cols = list(new_data['X'].columns)
        new_data['X'].columns = replace_bad_str(cols)
    return new_data


class MajorityVotingPrim(primitive):
    def __init__(self, random_state=0):
        super(MajorityVotingPrim, self).__init__(name='MajorityVoting')
        self.id = 83
        self.hyperparams = []
        self.type = 'ensemble'
        self.description = "Ensemble method taking multiple classifier probability predictions and outputs a prediction by a majority voting."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.labels = None
        self.good_labels = None
        self.labels_dict = {}
        self.accept_type = 'c_majority'

    def can_accept(self, data):
        cols = data['X'].columns
        if not all('Pred' in s for s in list(cols)):
            return False
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        output = handle_data(data)
        self.labels = list(map(str, list(pd.unique(output['Y']))))
        self.good_labels = replace_bad_str(deepcopy(self.labels))
        self.labels_dict = dict(zip(self.labels, list(pd.unique(output['Y']))))

    def produce(self, data):
        output = handle_data(data)
        proba_predictions = pd.DataFrame(np.zeros((output['X'].shape[0], len(self.labels))), columns=self.labels)
        for i in range(len(self.good_labels)):
           cols = [s for s in list(output['X'].columns) if self.good_labels[i] in s]
           proba_predictions[self.labels[i]] = output['X'][cols].values.mean(1)
        predictions = proba_predictions.idxmax(axis=1)
        output['predictions'] = np.array([self.labels_dict[x] for x in predictions])
        output['proba_predictions'] = proba_predictions
        final_output = {0: output}
        return final_output


class RandomForestMetaPrim(primitive):
    def __init__(self, random_state=0):
        super(RandomForestMetaPrim, self).__init__(name='RandomForestMeta')
        self.id = 84
        self.hyperparams = []
        self.type = 'ensemble'
        self.description = "A random forest classifier. A meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default)."
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
        try:
            self.model.fit(data['X'], data['Y'])
        except Exception as e:
            print(e)

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


class AdaBoostClassifierMetaPrim(primitive):
    def __init__(self, random_state=0):
        super(AdaBoostClassifierMetaPrim, self).__init__(name='AdaBoostClassifierMeta')
        self.id = 56
        self.hyperparams = []
        self.type = 'ensemble'
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


class ExtraTreesClassifierMetaPrim(primitive):
    def __init__(self, random_state=0):
        super(ExtraTreesClassifierMetaPrim, self).__init__(name='ExtraTreesMetaClassifier')
        self.id = 61
        self.hyperparams = []
        self.type = 'ensemble'
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


class GradientBoostingClassifierMetaPrim(primitive):
    def __init__(self, random_state=0):
        super(GradientBoostingClassifierMetaPrim, self).__init__(name='GradientBoostingClassifierMeta')
        self.id = 64
        self.hyperparams = []
        self.type = 'ensemble'
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


class XGBClassifierMetaPrim(primitive):
    def __init__(self, random_state=0):
        super(XGBClassifierMetaPrim, self).__init__(name='XGBClassifierMeta')
        self.id = 78
        self.hyperparams = []
        self.type = 'ensemble'
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


class RandomForestRegressorMetaPrim(primitive):
    def __init__(self, random_state=0):
        super(RandomForestRegressorMetaPrim, self).__init__(name='RandomForestRegressorMeta')
        self.hyperparams = []
        self.type = 'ensemble'
        self.description = "A random forest regressor. A meta estimator that fits a number of decision tree regressors on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default)."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = RandomForestRegressor(random_state=self.random_state, n_jobs=5)
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output