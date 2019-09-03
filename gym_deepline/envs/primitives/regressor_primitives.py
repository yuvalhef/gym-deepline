from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import ARDRegression
from gym_deepline.envs.Primitives import primitive
from copy import deepcopy
import pandas as pd
import numpy as np
np.random.seed(1)

def handle_data(data):
    new_data = {}
    if len(data) == 1:
        new_data = deepcopy(data[0])
        # new_data['X'].columns = list(map(str, list(range(new_data['X'].shape[1]))))
    else:
        concatenated_df = pd.DataFrame()
        for d_input in data.values():
            df2 = deepcopy(d_input['X'][d_input['X'].columns.difference(concatenated_df.columns)])
            concatenated_df = pd.concat([concatenated_df.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        # concatenated_df = concatenated_df.T.drop_duplicates().T
        new_data = deepcopy(data[0])
        new_data['X'] = concatenated_df.infer_objects()
        # new_data['X'].columns = list(map(str, list(range(new_data['X'].shape[1]))))
    cols = list(new_data['X'].columns)
    for i in range(len(cols)):
        col = cols[i]
        col = col.replace('[', 'abaqa')
        col = col.replace(']', 'bebab')
        col = col.replace('<', 'cfckc')
        col = col.replace('>', 'dmdad')
        cols[i] = col
    new_data['X'].columns = cols
    new_data['X'] = new_data['X'].loc[:, ~new_data['X'].columns.duplicated()]

    return new_data


class ARDRegressionPrim(primitive):
    def __init__(self, random_state=0):
        super(ARDRegressionPrim, self).__init__(name='ARDRegression')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "Bayesian ARD regression. Fit the weights of a regression model, using an ARD prior. The weights of the regression model are assumed to be in Gaussian distributions. Also estimate the parameters lambda (precisions of the distributions of the weights) and alpha (precision of the distribution of the noise). The estimation is done by an iterative procedures (Evidence Maximization)"
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = ARDRegression()
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


class AdaBoostRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(AdaBoostRegressorPrim, self).__init__(name='AdaBoostRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "An AdaBoost regressor. An AdaBoost [1] regressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction. As such, subsequent regressors focus more on difficult cases."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = AdaBoostRegressor(random_state=random_state)
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


class BaggingRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(BaggingRegressorPrim, self).__init__(name='BaggingRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "A Bagging regressor. A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it. This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [1]. If samples are drawn with replacement, then the method is known as Bagging [2]. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [3]. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [4]."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = BaggingRegressor(random_state=random_state, n_jobs=5)
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
