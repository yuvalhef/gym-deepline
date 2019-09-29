from gym_deepline.envs.Primitives import primitive
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, chi2, SelectKBest, f_classif,\
    mutual_info_classif, f_regression, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
import pandas as pd
from copy import deepcopy
from itertools import compress
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

    if not len(pd.unique(new_data['X'].columns)) == len(new_data['X'].columns):
        print('debug')
    return new_data


class VarianceThresholdPrim(primitive):
    def __init__(self, random_state=0):
        super(VarianceThresholdPrim, self).__init__(name='VarianceThreshold')
        self.id = 16
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Feature selector that removes all low-variance features."
        self.hyperparams_run = {'default': True}
        self.selector = VarianceThreshold()
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class UnivariateSelectChiKbestPrim(primitive):
    def __init__(self, random_state=0):
        super(UnivariateSelectChiKbestPrim, self).__init__(name='UnivariateSelectChiKbest')
        self.id = 17
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to the k highest scores with Chi-square"
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'd'

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        k = 10
        if self.hyperparams_run['default']:
            if data['X'].shape[1] < k: k = 'all'
        self.selector = SelectKBest(chi2, k=k)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class f_classifKbestPrim(primitive):
    def __init__(self, random_state=0):
        super(f_classifKbestPrim, self).__init__(name='f_classifKbest')
        self.id = 18
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to the k highest scores with ANOVA F-value between label/feature for classification tasks."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        k = 10
        if self.hyperparams_run['default']:
            if data['X'].shape[1] < k: k = 'all'
        self.selector = SelectKBest(f_classif, k=k)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class mutual_info_classifKbestPrim(primitive):
    def __init__(self, random_state=0):
        super(mutual_info_classifKbestPrim, self).__init__(name='mutual_info_classifKbest')
        self.id = 19
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to the k highest scores with Mutual information for a discrete target."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        k = 10
        if self.hyperparams_run['default']:
            if data['X'].shape[1] < k: k = 'all'
        self.selector = SelectKBest(mutual_info_classif, k=k)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class f_regressionKbestPrim(primitive):
    def __init__(self, random_state=0):
        super(f_regressionKbestPrim, self).__init__(name='f_regressionKbest')
        self.id = 20
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to the k highest scores with F-value between label/feature for regression tasks."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        k = 10
        if self.hyperparams_run['default']:
            if data['X'].shape[1] < k: k = 'all'
        self.selector = SelectKBest(f_regression, k=k)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class mutual_info_regressionKbestPrim(primitive):
    def __init__(self, random_state=0):
        super(mutual_info_regressionKbestPrim, self).__init__(name='mutual_info_regressionKbest')
        self.id = 21
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to the k highest scores with mutual information for a continuous target."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        k = 10
        if self.hyperparams_run['default']:
            if data['X'].shape[1] < k: k = 'all'
        self.selector = SelectKBest(mutual_info_regression, k=k)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class UnivariateSelectChiPercentilePrim(primitive):
    def __init__(self, random_state=0):
        super(UnivariateSelectChiPercentilePrim, self).__init__(name='UnivariateSelectChiPercentile')
        self.id = 22
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to a percentile of the highest scores with Chi-square"
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'd'

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectPercentile(chi2)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class f_classifPercentilePrim(primitive):
    def __init__(self, random_state=0):
        super(f_classifPercentilePrim, self).__init__(name='f_classifPercentile')
        self.id = 23
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to a percentile of the highest scores with ANOVA F-value between label/feature for classification tasks."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectPercentile(f_classif)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class mutual_info_classifPercentilePrim(primitive):
    def __init__(self, random_state=0):
        super(mutual_info_classifPercentilePrim, self).__init__(name='mutual_info_classifPercentile')
        self.id = 24
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to a percentile of the highest scores with Mutual information for a discrete target."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectPercentile(mutual_info_classif)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class f_regressionPercentilePrim(primitive):
    def __init__(self, random_state=0):
        super(f_regressionPercentilePrim, self).__init__(name='f_regressionPercentile')
        self.id = 25
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to a percentile of the highest scores with F-value between label/feature for regression tasks."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectPercentile(f_regression)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class mutual_info_regressionPercentilePrim(primitive):
    def __init__(self, random_state=0):
        super(mutual_info_regressionPercentilePrim, self).__init__(name='mutual_info_regressionPercentile')
        self.id = 26
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select features according to a percentile of the highest scores with mutual information for a continuous target."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectPercentile(mutual_info_regression)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class UnivariateSelectChiFPRPrim(primitive):
    def __init__(self, random_state=0):
        super(UnivariateSelectChiFPRPrim, self).__init__(name='UnivariateSelectChiFPR')
        self.id = 27
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Filter: Select the pvalues below alpha based on a FPR test with Chi-square. FPR test stands for False Positive Rate test. It controls the total amount of false detections."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'd'

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectFpr(chi2, alpha=0.05)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        try:
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        except Exception as e:
            print(e)
        final_output = {0: output}
        return final_output


class f_classifFPRPrim(primitive):
    def __init__(self, random_state=0):
        super(f_classifFPRPrim, self).__init__(name='f_classifFPR')
        self.id = 28
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Filter: Select the pvalues below alpha based on a FPR test with ANOVA F-value between label/feature for classification tasks. FPR test stands for False Positive Rate test. It controls the total amount of false detections."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectFpr(f_classif, alpha=0.05)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        try:
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        except Exception as e:
            print(e)
        final_output = {0: output}
        return final_output


# class mutual_info_classifFPRPrim(primitive):
#     def __init__(self, random_state=0):
#         super(mutual_info_classifFPRPrim, self).__init__(name='mutual_info_classifFPR')
#         self.id = 29
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature selection'
#         self.description = "Filter: Select the pvalues below alpha based on a FPR test with Mutual information for a discrete target. FPR test stands for False Positive Rate test. It controls the total amount of false detections."
#         self.hyperparams_run = {'default': True}
#         self.selector = None
#
#     def can_accept(self, data):
#         # data = handle_data(data))
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
#         self.selector = SelectFpr(mutual_info_classif, alpha=0.4)
#         self.selector.fit(data['X'], data['Y'])
#
#     def produce(self, data):
#         output = handle_data(data)
#         cols = list(output['X'].columns)
#         mask = self.selector.get_support(indices=False)
#         final_cols = list(compress(cols, mask))
#         output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
#         final_output = {0: output}
#         return final_output


class f_regressionFPRPrim(primitive):
    def __init__(self, random_state=0):
        super(f_regressionFPRPrim, self).__init__(name='f_regressionFPR')
        self.id = 29
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Filter: Select the pvalues below alpha based on a FPR test with F-value between label/feature for regression tasks. FPR test stands for False Positive Rate test. It controls the total amount of false detections."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectFpr(f_regression)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class UnivariateSelectChiFDRPrim(primitive):
    def __init__(self, random_state=0):
        super(UnivariateSelectChiFDRPrim, self).__init__(name='UnivariateSelectChiFDR')
        self.id = 31
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Filter: Select the p-values for an estimated false discovery rate with Chi-square. This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'd'

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectFdr(chi2, alpha=0.05)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        try:
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        except Exception as e:
            print(e)
        final_output = {0: output}
        return final_output


class f_classifFDRPrim(primitive):
    def __init__(self, random_state=0):
        super(f_classifFDRPrim, self).__init__(name='f_classifFDR')
        self.id = 32
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Filter: Select the p-values for an estimated false discovery rate with ANOVA F-value between label/feature for classification tasks. This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectFdr(f_classif)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        try:
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        except Exception as e:
            print(e)
        final_output = {0: output}
        return final_output


class f_regressionFDRPrim(primitive):
    def __init__(self, random_state=0):
        super(f_regressionFDRPrim, self).__init__(name='f_regressionFDR')
        self.id = 34
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Filter: Select the p-values for an estimated false discovery rate with F-value between label/feature for regression tasks. This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectFdr(f_regression)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class UnivariateSelectChiFWEPrim(primitive):
    def __init__(self, random_state=0):
        super(UnivariateSelectChiFWEPrim, self).__init__(name='UnivariateSelectChiFWE')
        self.id = 36
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select the p-values corresponding to Family-wise error rate with Chi-square."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'd'

    def can_accept(self, data):
        return self.can_accept_d(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectFwe(chi2, alpha=0.05)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        try:
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        except Exception as e:
            print(e)
        final_output = {0: output}
        return final_output


class f_classifFWEPrim(primitive):
    def __init__(self, random_state=0):
        super(f_classifFWEPrim, self).__init__(name='f_classifFWE')
        self.id = 37
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select the p-values corresponding to Family-wise error rate with ANOVA F-value between label/feature for classification tasks."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectFwe(f_classif, alpha=0.05)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        try:
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        except Exception as e:
            print(e)
        final_output = {0: output}
        return final_output


class f_regressionFWEPrim(primitive):
    def __init__(self, random_state=0):
        super(f_regressionFWEPrim, self).__init__(name='f_regressionFWE')
        self.id = 39
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Select the p-values corresponding to Family-wise error rate with F-value between label/feature for regression tasks."
        self.hyperparams_run = {'default': True}
        self.selector = None
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector = SelectFwe(f_regression, alpha=0.05)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        try:
            mask = self.selector.get_support(indices=False)
            final_cols = list(compress(cols, mask))
            output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        except Exception as e:
            print(e)
        final_output = {0: output}
        return final_output


class RFE_RandomForestPrim(primitive):
    def __init__(self, random_state=0):
        super(RFE_RandomForestPrim, self).__init__(name='RFE_RandomForest')
        self.id = 41
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Feature ranking with recursive feature elimination with Random-Forest classifier. Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.selector = RFE(RandomForestClassifier(random_state=self.random_state))
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector.fit(data['X'], data['Y'])


    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class RFE_GradientBoostingPrim(primitive):
    def __init__(self, random_state=0):
        super(RFE_GradientBoostingPrim, self).__init__(name='RFE_GradientBoosting')
        self.id = 42
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Feature ranking with recursive feature elimination with Gradient-Boosting classifier. Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached."
        self.hyperparams_run = {'default': True}
        self.selector = RFE(GradientBoostingClassifier(n_estimators=20))
        self.accept_type = 'c'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Classification')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class RFE_SVRPrim(primitive):
    def __init__(self, random_state=0):
        super(RFE_SVRPrim, self).__init__(name='RFE_SVR')
        self.id = 43
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Feature ranking with recursive feature elimination with SVR regressor. Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached."
        self.hyperparams_run = {'default': True}
        self.selector = RFE(SVR(kernel="linear"))
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output


class RFE_RandomForestRegPrim(primitive):
    def __init__(self, random_state=0):
        super(RFE_RandomForestRegPrim, self).__init__(name='RFE_RandomForestReg')
        self.id = 44
        self.PCA_LAPACK_Prim = []
        self.type = 'feature selection'
        self.description = "Feature ranking with recursive feature elimination with Random-Forest regressor. Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached."
        self.hyperparams_run = {'default': True}
        self.selector = RFE(RandomForestRegressor())
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        if data['X'].shape[1] < 3:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        self.selector.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        mask = self.selector.get_support(indices=False)
        final_cols = list(compress(cols, mask))
        output['X'] = pd.DataFrame(self.selector.transform(output['X']), columns=final_cols)
        final_output = {0: output}
        return final_output

# Add more RFE primitives!

# Add  SelectFromModel primitives!
