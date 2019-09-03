from scipy.sparse import csr_matrix
from ..Primitives import primitive
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
import pandas as pd
from copy import deepcopy
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


class NumericDataPrim(primitive):
    def __init__(self, random_state=0):
        super(NumericDataPrim, self).__init__(name='NumericData')
        self.id = 0
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Extracts only numeric data columns from input."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.num_cols = None
        self.accept_type = 'a'

    def can_accept(self, data):
        return self.can_accept_a(data)

    def is_needed(self, data):
        # data = handle_data(data)
        cols = data['X'].columns
        num_cols = data['X']._get_numeric_data().columns
        if not len(cols) == len(num_cols):
            return True
        return False

    def fit(self, data):
        data = handle_data(data)
        self.num_cols = data['X']._get_numeric_data().columns

    def produce(self, data):
        output = handle_data(data)
        output['X'] = output['X'][self.num_cols]
        final_output = {0: output}
        return final_output


class Imputer(primitive):
    def __init__(self, random_state=0):
        super(Imputer, self).__init__(name='imputer')
        self.id = 1
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Imputation transformer for completing missing values by mean."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.imp = SimpleImputer()
        self.num_cols = None
        self.imp_cols = None
        self.accept_type = 'a'

    def can_accept(self, data):
        return self.can_accept_a(data)

    def is_needed(self, data):
        # data = handle_data(data)
        if data['X'].isnull().any().any():
            return True
        return False

    def fit(self, data):
        data = handle_data(data)
        self.num_cols = data['X']._get_numeric_data().columns
        self.imp.fit(data['X'][self.num_cols])
        self.imp_cols = data['X'][self.num_cols].columns[data['X'][self.num_cols].isnull().any()].tolist()

    def produce(self, data):
        output = handle_data(data)
        # self.imp_cols = output['X'][self.num_cols].columns[output['X'][self.num_cols].isnull().any()].tolist()
        cols = self.num_cols.tolist()
        reg_cols = list(set(cols) - set(self.imp_cols))
        # new_cols = ["{}_imp_mean".format(v) for v in list(imp_cols)]
        for i in range(len(cols)):
            if cols[i] in reg_cols:
                continue
            elif cols[i] in self.imp_cols:
                cols[i] = "{}_imp_mean".format(cols[i])

        # try:
        output['X'] = pd.DataFrame(self.imp.transform(output['X'][self.num_cols]), columns=cols).reset_index(
            drop=True).infer_objects()
        output['X'] = output['X'].ix[:, ~output['X'].columns.duplicated()]

        # except Exception as e:
        #     print(e)
        final_output = {0: output}
        return final_output


class ImputerMedian(primitive):
    def __init__(self, random_state=0):
        super(ImputerMedian, self).__init__(name='ImputerMedian')
        self.id = 2
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Imputation transformer for completing missing values by median."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.imp = SimpleImputer(strategy='median')
        self.num_cols = None
        self.imp_cols = None
        self.accept_type = 'a'

    def can_accept(self, data):
        return self.can_accept_a(data)

    def is_needed(self, data):
        # data = handle_data(data)
        if data['X'].isnull().any().any():
            return True
        return False

    def fit(self, data):
        data = handle_data(data)
        self.num_cols = data['X']._get_numeric_data().columns
        self.imp.fit(data['X'][self.num_cols])
        self.imp_cols = data['X'][self.num_cols].columns[data['X'][self.num_cols].isnull().any()].tolist()

    def produce(self, data):
        output = handle_data(data)
        # self.imp_cols = output['X'][self.num_cols].columns[output['X'][self.num_cols].isnull().any()].tolist()
        cols = self.num_cols.tolist()
        reg_cols = list(set(cols)-set(self.imp_cols))
        # new_cols = ["{}_imp_mean".format(v) for v in list(imp_cols)]
        for i in range(len(cols)):
            if cols[i] in reg_cols:
                continue
            elif cols[i] in self.imp_cols:
                cols[i] = "{}_imp_median".format(cols[i])

        # try:
        output['X'] = pd.DataFrame(self.imp.transform(output['X'][self.num_cols]), columns=cols).reset_index(drop=True).infer_objects()
        output['X'] = output['X'].ix[:, ~output['X'].columns.duplicated()]

        # except Exception as e:
        #     print(e)
        final_output = {0: output}
        return final_output


class ImputerIndicatorPrim(primitive):
    def __init__(self, random_state=0):
        super(ImputerIndicatorPrim, self).__init__(name='imputerIndicator')
        self.id = 3
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "All features will be imputed using SimpleImputer, in order to enable classifiers to work with this data. Additionally, it adds the the indicator variables from MissingIndicator."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.imp = FeatureUnion(transformer_list=[('features', SimpleImputer()), ('indicators', MissingIndicator())])
        self.num_cols = None
        self.imp_cols = None
        self.accept_type = 'b'

    def can_accept(self, data):
        return self.can_accept_b(data)

    def is_needed(self, data):
        # data = handle_data(data)
        if data['X'].isnull().any().any():
            return True
        return False

    def fit(self, data):
        data = handle_data(data)
        self.num_cols = data['X']._get_numeric_data().columns
        self.imp.fit(data['X'][self.num_cols])
        self.imp_cols = data['X'][self.num_cols].columns[data['X'][self.num_cols].isnull().any()].tolist()

    def produce(self, data):
        output = handle_data(data)

        cols = self.num_cols.tolist()
        reg_cols = list(set(cols)-set(self.imp_cols))
        # new_cols = ["{}_imp_mean".format(v) for v in list(imp_cols)]
        for i in range(len(cols)):
            if cols[i] in reg_cols:
                continue
            elif cols[i] in self.imp_cols:
                cols[i] = "{}_imp_mean".format(cols[i])
        result = self.imp.transform(output['X'][self.num_cols])
        # extra_cols = list(range(result.shape[1] - len(cols)))
        extra_cols = ["{}_miss_indicator".format(v) for v in self.imp_cols]
        output['X'] = pd.DataFrame(result, columns=cols + extra_cols).reset_index(drop=True).infer_objects()
        output['X'] = output['X'].ix[:,~output['X'].columns.duplicated()]
        final_output = {0: output}
        return final_output


class OneHotEncoderPrim(primitive):
    # can handle missing values. turns nans to extra category
    def __init__(self, random_state=0):
        super(OneHotEncoderPrim, self).__init__(name='OneHotEncoder')
        self.id = 4
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Encode categorical integer features as a one-hot numeric array. The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. The features are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’) encoding scheme. This creates a binary column for each category and returns a sparse matrix or dense array. By default, the encoder derives the categories based on the unique values in each feature. Alternatively, you can also specify the categories manually. The OneHotEncoder previously assumed that the input features take on values in the range [0, max(values)). This behaviour is deprecated. This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels."
        self.hyperparams_run = {'default': True}
        self.preprocess = None
        self.cat_cols = None
        self.accept_type = 'b'

    def can_accept(self, data):
        return self.can_accept_b(data)

    def is_needed(self, data):
        # data = handle_data(data)
        cols = data['X']
        num_cols = data['X']._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) == 0:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        if not self.is_needed(data):
            return
        x = deepcopy(data['X'])
        cols = data['X'].columns
        num_cols = data['X']._get_numeric_data().columns
        self.cat_cols = list(set(cols) - set(num_cols))
        x[self.cat_cols] = x[self.cat_cols].fillna('NaN')
        self.preprocess = ColumnTransformer([("one_hot", OneHotEncoder(handle_unknown='ignore'), self.cat_cols)])
        x[self.cat_cols] = x[self.cat_cols].astype(str)
        self.preprocess.fit(x)  # .astype(str)

    def produce(self, data):
        output = handle_data(data)
        if not self.is_needed(output):
            final_output = {0: output}
            return final_output
        output['X'][self.cat_cols] = output['X'][self.cat_cols].fillna('NaN')
        result = self.preprocess.transform(output['X'])
        if isinstance(result, csr_matrix):
            result = result.toarray()
        output['X'] = pd.DataFrame(result, columns=self.preprocess.get_feature_names()).infer_objects()
        output['X'] = output['X'].ix[:,~output['X'].columns.duplicated()]
        final_output = {0: output}
        return final_output


class LabelEncoderPrim(primitive):
    # can handle missing values. Operates on all categorical features.
    def __init__(self, random_state=0):
        super(LabelEncoderPrim, self).__init__(name='LabelEncoder')
        self.id = 5
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Encode labels with value between 0 and n_classes-1."
        self.hyperparams_run = {'default': True}
        self.cat_cols = None
        self.preprocess = {}
        self.accept_type = 'b'

    def can_accept(self, data):
        return self.can_accept_b(data)

    def is_needed(self, data):
        # data = handle_data(data)
        cols = data['X']
        num_cols = data['X']._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if len(cat_cols) == 0:
            return False
        return True

    def fit(self, data):
        data = handle_data(data)
        if not self.is_needed(data):
            return
        x = deepcopy(data['X'])
        cols = data['X'].columns
        num_cols = data['X']._get_numeric_data().columns
        self.cat_cols = list(set(cols) - set(num_cols))
        x[self.cat_cols] = x[self.cat_cols].fillna('NaN')
        for col in self.cat_cols:
            self.preprocess[col] = LabelEncoder().fit(x[col].astype(str))

    def produce(self, data):
        output = handle_data(data)
        if not self.is_needed(output):
            final_output = {0: output}
            return final_output
        x = output['X']
        x[self.cat_cols] = x[self.cat_cols].fillna('NaN')
        final_cols = []
        for col in self.cat_cols:
            arr = self.preprocess[col].transform(x[col])
            output['X'][col] = arr
            output['X'] = output['X'].rename(index=str, columns={col: "{}_lbl_enc".format(col)})
            final_cols.append("{}_lbl_enc".format(col))
            # output['X'][col].columns = "{}_lbl_enc".format(col)
        # cols = ["{}_lbl_enc".format(v) for v in list(self.cat_cols)]
        output['X'] = output['X'][final_cols].infer_objects()
        output['X'] = output['X'].ix[:,~output['X'].columns.duplicated()]
        final_output = {0: output}
        return final_output


class ImputerEncoderPrim(primitive):
    def __init__(self, random_state=0):
        super(ImputerEncoderPrim, self).__init__(name='ImputerEncoderPrim')
        self.id = 6
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Imputation transformer for completing missing values and encode labels with value between 0 and n_classes-1."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.imp = Imputer()
        self.encoder = LabelEncoderPrim()
        self.accept_type = 'b'

    def can_accept(self, data):
        return self.can_accept_b(data)

    def is_needed(self, data):
        # data = handle_data(data)
        if not self.imp.is_needed(data) or not self.encoder.is_needed(data):
            return False
        return True

    def fit(self, data):
        dt = handle_data(data)
        if self.encoder.is_needed(dt):
            self.needed_enc = True
            self.encoder.fit(data)
            out = self.encoder.produce(data)
        else:
            out = data
        if self.imp.is_needed(handle_data(out)):
            self.needed_imp = True
            self.imp.fit(out)

    def produce(self, data):
        output = data
        if self.needed_enc:
            output = self.encoder.produce(data)[0]
        if self.needed_imp:
            output = self.imp.produce(output)[0]

        if not self.needed_imp and not self.needed_enc:
            output = handle_data(output)
        final_output = {0: output}
        return final_output


class ImputerOneHotEncoderPrim(primitive):
    def __init__(self, random_state=0):
        super(ImputerOneHotEncoderPrim, self).__init__(name='ImputerOneHotEncoderPrim')
        self.id = 6
        self.hyperparams = []
        self.type = 'data preprocess'
        self.description = "Imputation transformer for completing missing values and encode categorical one-hot."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.imp = Imputer()
        self.encoder = OneHotEncoderPrim()
        self.accept_type = 'b'
        self.needed_imp = False
        self.needed_enc = False

    def can_accept(self, data):
        return self.can_accept_b(data)

    def is_needed(self, data):
        # data = handle_data(data)
        if not self.imp.is_needed(data) or not self.encoder.is_needed(data):
            return False
        return True

    def fit(self, data):
        dt = handle_data(data)
        if self.encoder.is_needed(dt):
            self.needed_enc = True
            self.encoder.fit(data)
            out = self.encoder.produce(data)
        else:
            out = data
        if self.imp.is_needed(handle_data(out)):
            self.needed_imp = True
            self.imp.fit(out)

    def produce(self, data):
        output = data
        if self.needed_enc:
            output = self.encoder.produce(data)[0]
        if self.needed_imp:
            output = self.imp.produce(output)[0]

        if not self.needed_imp and not self.needed_enc:
            output = handle_data(output)
        final_output = {0: output}
        return final_output
