from gym_deepline.envs.Primitives import primitive
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer, Normalizer, KBinsDiscretizer, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from copy import deepcopy
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


class MinMaxScalerPrim(primitive):
    def __init__(self, random_state=0):
        super(MinMaxScalerPrim, self).__init__(name='MinMaxScaler')
        self.id = 7
        self.hyperparams = []
        self.type = 'feature preprocess'
        self.description = "Transforms features by scaling each feature to a given range. This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one."
        self.hyperparams_run = {'default': True}
        self.scaler = MinMaxScaler()
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.scaler.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        for i in range(len(cols)):
            if not 'one_hot' in cols[i]:
                cols[i] = "{}_mnmxscale".format(cols[i])
        output['X'] = pd.DataFrame(self.scaler.transform(output['X']), columns=cols)
        final_output = {0: output}
        return final_output


class MaxAbsScalerPrim(primitive):
    def __init__(self, random_state=0):
        super(MaxAbsScalerPrim, self).__init__(name='MaxAbsScaler')
        self.id = 8
        self.hyperparams = []
        self.type = 'feature preprocess'
        self.description = "Scale each feature by its maximum absolute value. his estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity. This scaler can also be applied to sparse CSR or CSC matrices."
        self.hyperparams_run = {'default': True}
        self.scaler = MaxAbsScaler()
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        # Update
        return True

    def fit(self, data):
        data = handle_data(data)
        self.scaler.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        for i in range(len(cols)):
            if not 'one_hot' in cols[i]:
                cols[i] = "{}_mxabsscale".format(cols[i])
        output['X'] = pd.DataFrame(self.scaler.transform(output['X']), columns=cols)
        final_output = {0: output}
        return final_output


class RobustScalerPrim(primitive):
    def __init__(self, random_state=0):
        super(RobustScalerPrim, self).__init__(name='RobustScaler')
        self.id = 9
        self.hyperparams = []
        self.type = 'feature preprocess'
        self.description = "Scale features using statistics that are robust to outliers. This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile). Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Median and interquartile range are then stored to be used on later data using the transform method. Standardization of a dataset is a common requirement for many machine learning estimators. Typically this is done by removing the mean and scaling to unit variance. However, outliers can often influence the sample mean / variance in a negative way. In such cases, the median and the interquartile range often give better results."
        self.hyperparams_run = {'default': True}
        self.scaler = RobustScaler()
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        # Update
        return True

    def fit(self, data):
        data = handle_data(data)
        self.scaler.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_rbstscale".format(x) for x in cols]
        output['X'] = pd.DataFrame(self.scaler.transform(output['X']), columns=cols)
        final_output = {0: output}
        return final_output


class StandardScalerPrim(primitive):
    def __init__(self, random_state=0):
        super(StandardScalerPrim, self).__init__(name='StandardScaler')
        self.id = 10
        self.hyperparams = []
        self.type = 'feature preprocess'
        self.description = "Standardize features by removing the mean and scaling to unit variance"
        self.hyperparams_run = {'default': True}
        self.scaler = StandardScaler()
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        # Update
        return True

    def fit(self, data):
        data = handle_data(data)
        self.scaler.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_stndrdscale".format(x) for x in cols]
        output['X'] = pd.DataFrame(self.scaler.transform(output['X']), columns=cols)
        final_output = {0: output}
        return final_output


class QuantileTransformerPrim(primitive):
    def __init__(self, random_state=0):
        super(QuantileTransformerPrim, self).__init__(name='QuantileTransformer')
        self.id = 11
        self.hyperparams = []
        self.type = 'feature preprocess'
        self.description = "Transform features using quantiles information. This method transforms the features to follow a uniform or a normal distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values. It also reduces the impact of (marginal) outliers: this is therefore a robust preprocessing scheme. The transformation is applied on each feature independently. The cumulative distribution function of a feature is used to project the original values. Features values of new/unseen data that fall below or above the fitted range will be mapped to the bounds of the output distribution. Note that this transform is non-linear. It may distort linear correlations between variables measured at the same scale but renders variables measured at different scales more directly comparable."
        self.hyperparams_run = {'default': True}
        self.scaler = QuantileTransformer()
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        # Update
        return True

    def fit(self, data):
        data = handle_data(data)
        self.scaler.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_qntl".format(x) for x in cols]
        output['X'] = pd.DataFrame(self.scaler.transform(output['X']), columns=cols)
        final_output = {0: output}
        return final_output


class PowerTransformerPrim(primitive):
    def __init__(self, random_state=0):
        super(PowerTransformerPrim, self).__init__(name='PowerTransformer')
        self.id = 12
        self.hyperparams = []
        self.type = 'feature preprocess'
        self.description = "Apply a power transform featurewise to make data more Gaussian-like. Power transforms are a family of parametric, monotonic transformations that are applied to make data more Gaussian-like. This is useful for modeling issues related to heteroscedasticity (non-constant variance), or other situations where normality is desired. Currently, PowerTransformer supports the Box-Cox transform and the Yeo-Johnson transform. The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood. Box-Cox requires input data to be strictly positive, while Yeo-Johnson supports both positive or negative data. By default, zero-mean, unit-variance normalization is applied to the transformed data."
        self.hyperparams_run = {'default': True}
        self.scaler = PowerTransformer()
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        # Update
        return True

    def fit(self, data):
        data = handle_data(data)
        self.scaler.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_pwrtrnsfrm".format(x) for x in cols]
        output['X'] = pd.DataFrame(self.scaler.transform(output['X']), columns=cols)
        final_output = {0: output}
        return final_output


class NormalizerPrim(primitive):
    def __init__(self, random_state=0):
        super(NormalizerPrim, self).__init__(name='Normalizer')
        self.id = 13
        self.hyperparams = []
        self.type = 'feature preprocess'
        self.description = "Normalize samples individually to unit norm. Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of other samples so that its norm (l1 or l2) equals one. This transformer is able to work both with dense numpy arrays and scipy.sparse matrix (use CSR format if you want to avoid the burden of a copy / conversion). Scaling inputs to unit norms is a common operation for text classification or clustering for instance. For instance the dot product of two l2-normalized TF-IDF vectors is the cosine similarity of the vectors and is the base similarity metric for the Vector Space Model commonly used by the Information Retrieval community."
        self.hyperparams_run = {'default': True}
        self.scaler = Normalizer()
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        # Update
        return True

    def fit(self, data):
        data = handle_data(data)
        self.scaler.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_nrmlzr".format(x) for x in cols]
        output['X'] = pd.DataFrame(self.scaler.transform(output['X']), columns=cols)
        final_output = {0: output}
        return final_output


class KBinsDiscretizerOrdinalPrim(primitive):
    # can handle missing values. turns nans to extra category
    def __init__(self, random_state=0):
        super(KBinsDiscretizerOrdinalPrim, self).__init__(name='KBinsDiscretizerOrdinal')
        self.id = 14
        self.hyperparams = []
        self.type = 'feature preprocess'
        self.description = "Bin continuous data into intervals. Ordinal."
        self.hyperparams_run = {'default': True}
        self.preprocess = None
        self.accept_type = 'c_t_kbins'

    def can_accept(self, data):
        if not self.can_accept_c(data):
            return False
        for c in data['X'].columns:
            if np.unique(data['X'][c]).shape[0] < 10:
                return False
        return True


    def is_needed(self, data):
        # data = handle_data(data)
        # Update
        return True

    def fit(self, data):
        data = handle_data(data)
        # data['X'].columns = data['X'].columns.astype(str)
        num_cols = data['X']._get_numeric_data().columns
        self.preprocess = ColumnTransformer([("discrit", KBinsDiscretizer(encode='ordinal'), list(set(num_cols)))])
        self.preprocess.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_ordnldscrtzr".format(x) for x in cols]
        output['X'] = pd.DataFrame(self.preprocess.transform(output['X']), columns=cols)
        final_output = {0: output}
        return final_output


class KBinsDiscretizerOneHotPrim(primitive):
    # can handle missing values. turns nans to extra category
    def __init__(self, random_state=0):
        super(KBinsDiscretizerOneHotPrim, self).__init__(name='KBinsDiscretizerOneHot')
        self.id = 15
        self.hyperparams = []
        self.type = 'feature preprocess'
        self.description = "Bin continuous data into intervals. One-Hot"
        self.hyperparams_run = {'default': True}
        self.preprocess = None
        self.accept_type = 'kbinsohe'

    def can_accept(self, data):
        # data = handle_data(data)
        if data['X'].empty:
            return False
        cols = data['X'].columns
        if any('one_hot' in x for x in list(cols)):
            return False
        num_cols = data['X']._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if data['X'].isnull().any().any():
            return False
        elif not len(cat_cols) == 0:
            return False
        for c in data['X'].columns:
            if np.unique(data['X'][c]).shape[0] < 10:
                return False
        return True


    def is_needed(self, data):
        # data = handle_data(data)
        # Update
        return True

    def fit(self, data):
        data = handle_data(data)
        num_cols = data['X']._get_numeric_data().columns
        self.preprocess = ColumnTransformer([("discrit", KBinsDiscretizer(), list(set(num_cols)))])
        self.preprocess.fit(data['X'])


    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        final_cols = []
        for i in range(output['X'].shape[1]):
            for j in range(5):
                final_cols.append("{}_onhtbnrzr{}".format(cols[i], str(j)))
        result = self.preprocess.transform(output['X'])
        if isinstance(result, csr_matrix):
            result = result.toarray()
        output['X'] = pd.DataFrame(result, columns=final_cols[:result.shape[1]]).infer_objects()
        # output['X'] = pd.DataFrame(self.preprocess.transform(output['X']).toarray())
        final_output = {0: output}
        return final_output
