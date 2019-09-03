from gym_deepline.envs.Primitives import primitive
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD, FastICA, NMF, LatentDirichletAllocation
from sklearn.ensemble import RandomTreesEmbedding
import pandas as pd
import numpy as np
from copy import deepcopy
import warnings
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


class PolynomialFeaturesPrim(primitive):
    def __init__(self, random_state=0):
        super(PolynomialFeaturesPrim, self).__init__(name='PolynomialFeatures')
        self.id = 45
        self.hyperparams = []
        self.type = 'feature engineering'
        self.description = "Generate polynomial and interaction features. Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]."
        self.hyperparams_run = {'default': True}
        self.scaler = PolynomialFeatures(include_bias=False)
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
        result = self.scaler.transform(output['X'])
        cols = self.scaler.get_feature_names(output['X'].columns)
        output['X'] = pd.DataFrame(result, columns=cols)
        output['X'] = output['X'].loc[:, ~output['X'].columns.duplicated()]
        final_output = {0: output}
        return final_output


class InteractionFeaturesPrim(primitive):
    def __init__(self, random_state=0):
        super(InteractionFeaturesPrim, self).__init__(name='InteractionFeatures')
        self.id = 46
        self.hyperparams = []
        self.type = 'feature engineering'
        self.description = "Generate interaction features."
        self.hyperparams_run = {'default': True}
        self.scaler = PolynomialFeatures(interaction_only=True, include_bias=False)
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
        result = self.scaler.transform(output['X'])
        cols = self.scaler.get_feature_names(output['X'].columns)
        output['X'] = pd.DataFrame(result, columns=cols)
        output['X'] = output['X'].loc[:, ~output['X'].columns.duplicated()]
        final_output = {0: output}
        return final_output


class PCA_LAPACK_Prim(primitive):
    def __init__(self, random_state=0):
        super(PCA_LAPACK_Prim, self).__init__(name='PCA_LAPACK')
        self.id = 47
        self.PCA_LAPACK_Prim = []
        self.type = 'feature engineering'
        self.description = "LAPACK principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.hyperparams_run = {'default': True}
        self.pca = PCA(svd_solver='full')  # n_components=0.9
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.pca.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_pcalpck".format(x) for x in cols]
        result = self.pca.transform(output['X'])
        output['X'] = pd.DataFrame(result, columns=cols[:result.shape[1]])
        final_output = {0: output}
        return final_output


class PCA_ARPACK_Prim(primitive):
    def __init__(self, random_state=0):
        super(PCA_ARPACK_Prim, self).__init__(name='PCA_ARPACK')
        self.id = 48
        self.PCA_LAPACK_Prim = []
        self.type = 'feature engineering'
        self.description = "ARPACK principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.hyperparams_run = {'default': True}
        self.pca = PCA(svd_solver='arpack')
        self.accept_type = 'c_t_arpck'

    def can_accept(self, data):
        return self.can_accept_c(data, larpack=True)

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.pca.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_pcaarpck".format(x) for x in cols]
        result = self.pca.transform(output['X'])
        output['X'] = pd.DataFrame(result, columns=cols[:result.shape[1]])
        final_output = {0: output}
        return final_output


class PCA_Randomized_Prim(primitive):
    def __init__(self, random_state=0):
        super(PCA_Randomized_Prim, self).__init__(name='PCA_Randomized')
        self.id = 49
        self.PCA_LAPACK_Prim = []
        self.type = 'feature engineering'
        self.description = "Randomized SVD principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. It uses the LAPACK implementation of the full SVD or a randomized truncated SVD by the method of Halko et al. 2009, depending on the shape of the input data and the number of components to extract."
        self.hyperparams_run = {'default': True}
        self.pca = PCA(svd_solver='randomized')
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.pca.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_pcarndmzd".format(x) for x in cols]
        result = self.pca.transform(output['X'])
        output['X'] = pd.DataFrame(result, columns=cols[: result.shape[1]])
        final_output = {0: output}
        return final_output


class IncrementalPCA_Prim(primitive):
    def __init__(self, random_state=0):
        super(IncrementalPCA_Prim, self).__init__(name='IncrementalPCA')
        self.id = 50
        self.PCA_LAPACK_Prim = []
        self.type = 'feature engineering'
        self.description = "Incremental principal components analysis (IPCA). Linear dimensionality reduction using Singular Value Decomposition of centered data, keeping only the most significant singular vectors to project the data to a lower dimensional space. Depending on the size of the input data, this algorithm can be much more memory efficient than a PCA. This algorithm has constant memory complexity."
        self.hyperparams_run = {'default': True}
        self.pca = IncrementalPCA()
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.pca.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_pcaincrmnt".format(x) for x in cols]
        result = self.pca.transform(output['X'])
        output['X'] = pd.DataFrame(result, columns=cols[: result.shape[1]])
        final_output = {0: output}
        return final_output


class KernelPCA_Prim(primitive):
    def __init__(self, random_state=0):
        super(KernelPCA_Prim, self).__init__(name='KernelPCA')
        self.id = 51
        self.PCA_LAPACK_Prim = []
        self.type = 'feature engineering'
        self.description = "Kernel Principal component analysis (KPCA). Non-linear dimensionality reduction through the use of kernels"
        self.hyperparams_run = {'default': True}
        self.pca = None  # n_components=5
        self.accept_type = 'c_t_krnl'

    def can_accept(self, data):
        if int(0.5 * data['X'].shape[1]) == 0:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.pca = KernelPCA(n_components=int(0.5 * data['X'].shape[1]))
        self.pca.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_pcakrnl".format(x) for x in cols]
        result = self.pca.transform(output['X'])
        output['X'] = pd.DataFrame(result, columns=cols[: result.shape[1]])
        final_output = {0: output}
        return final_output


# class SparsePCA_Prim(primitive):
#     def __init__(self, random_state=0):
#         super(SparsePCA_Prim, self).__init__(name='SparsePCA')
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature engineering'
#         self.description = "Sparse Principal Components Analysis (SparsePCA) Finds the set of sparse components that can optimally reconstruct the data. The amount of sparseness is controllable by the coefficient of the L1 penalty, given by the parameter alpha."
#         self.hyperparams_run = {'default': True}
#         self.pca = SparsePCA(normalize_components=True)
#
#     def can_accept(self, data):
#         # data = handle_data(data)
#         if data['X'].empty:
#             return False
#         cols = data['X']
#         num_cols = data['X']._get_numeric_data().columns
#         cat_cols = list(set(cols) - set(num_cols))
#         if data['X'].isnull().any().any():
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
#         self.pca.fit(data['X'])
#
#     def produce(self, data):
#         output = handle_data(data)
#         output['X'] = pd.DataFrame(self.pca.transform(output['X']))
#         final_output = {0: output}
#         return final_output


class TruncatedSVD_Prim(primitive):
    def __init__(self, random_state=0):
        super(TruncatedSVD_Prim, self).__init__(name='TruncatedSVD')
        self.id = 52
        self.PCA_LAPACK_Prim = []
        self.type = 'feature engineering'
        self.description = "Dimensionality reduction using truncated SVD (aka LSA). This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with scipy.sparse matrices efficiently. In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA). This estimator supports two algorithms: a fast randomized SVD solver, and a “naive” algorithm that uses ARPACK as an eigensolver on (X * X.T) or (X.T * X), whichever is more efficient."
        self.hyperparams_run = {'default': True}
        self.pca = None
        self.accept_type = 'c_t_krnl'

    def can_accept(self, data):
        if int(0.5 * data['X'].shape[1]) == 0:
            return False
        else:
            return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.pca = TruncatedSVD(n_components=int(0.5*data['X'].shape[1]))
        self.pca.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_pcatrnckd".format(x) for x in cols]
        result = self.pca.transform(output['X'])
        output['X'] = pd.DataFrame(result, columns=cols[: result.shape[1]])
        final_output = {0: output}
        return final_output


class FastICA_Prim(primitive):
    def __init__(self, random_state=0):
        super(FastICA_Prim, self).__init__(name='FastICA')
        self.id = 53
        self.PCA_LAPACK_Prim = []
        self.type = 'feature engineering'
        self.description = "FastICA: a fast algorithm for Independent Component Analysis."
        self.hyperparams_run = {'default': True}
        self.pca = FastICA(random_state=random_state)
        self.accept_type = 'c_t'
        self.sk_error = False

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message='array must not contain infs or NaNs')
            try:
                self.pca.fit(data['X'])
            except ValueError as e:
                if 'array must not contain infs or NaNs' in e.args[0]:
                    print('sklearn error in FactICA (array must not contain infs or NaNs)- skipping the primitive')
                    self.sk_error = True
                else:
                    raise ValueError(e)


    def produce(self, data):
        if self.sk_error:
            return {0: handle_data(data)}
        output = handle_data(data)
        cols = list(output['X'].columns)
        cols = ["{}_fstica".format(x) for x in cols]
        result = self.pca.transform(output['X'])
        output['X'] = pd.DataFrame(result, columns=cols[: result.shape[1]])
        final_output = {0: output}
        return final_output


# class NMF_Prim(primitive):
#     # Only non-negative data
#     def __init__(self, random_state=0):
#         super(NMF_Prim, self).__init__(name='NMF')
#         self.PCA_LAPACK_Prim = []
#         self.type = 'feature engineering'
#         self.description = "Non-Negative Matrix Factorization (NMF) Find two non-negative matrices (W, H) whose product approximates the non- negative matrix X. This factorization can be used for example for dimensionality reduction, source separation or topic extraction."
#         self.hyperparams_run = {'default': True}
#         self.pca = NMF()
#
#     def can_accept(self, data):
#         # data = handle_data(data)
#         if data['X'].empty:
#             return False
#         cols = data['X']
#         num_cols = data['X']._get_numeric_data().columns
#         cat_cols = list(set(cols) - set(num_cols))
#         if not len(cat_cols) == 0:
#             return False
#         elif data['X'].lt(0).sum().sum() > 0:
#             return False
#         elif data['X'].isnull().any().any():
#             return False
#         return True
#
#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True
#
#     def fit(self, data):
#         data = handle_data(data)
#         self.pca.fit(data['X'])
#
#     def produce(self, data):
#         output = handle_data(data)
#         # try:
#         # if not self.can_accept(output):
#         #     return -1
#         cols = list(output['X'].columns)
#         cols = ["{}_nmf".format(x) for x in cols]
#         result = self.pca.transform(output['X'])
#         output['X'] = pd.DataFrame(result, columns=cols[: result.shape[1]])
#         # except Exception as e:
#         #     print(e)
#         final_output = {0: output}
#         return final_output


# class LDA_Prim(primitive):
#     # Only non-negative data
#     def __init__(self, random_state=0):
#         super(LDA_Prim, self).__init__(name='LDA_Prim')
#         self.type = 'feature engineering'
#         self.description = "Latent Dirichlet Allocation with online variational Bayes algorithm."
#         self.hyperparams_run = {'default': True}
#         self.pca = None
#         self.n_topics = None
#
#     def can_accept(self, data):
#         # data = handle_data(data)
#         if data['X'].empty:
#             return False
#         cols = data['X']
#         num_cols = data['X']._get_numeric_data().columns
#         cat_cols = list(set(cols) - set(num_cols))
#         if not len(cat_cols) == 0:
#             return False
#         if data['X'].isnull().any().any() or data['X'].lt(0).sum().sum() > 0:
#             return False
#         return True
#
#     def is_needed(self, data):
#         # data = handle_data(data)
#         return True
#
#     def fit(self, data):
#         data = handle_data(data)
#         self.n_topics = min(10, data['X'].shape[1])
#         self.pca = LatentDirichletAllocation(n_topics=self.n_topics)
#         self.pca.fit(data['X'])
#
#     def produce(self, data):
#         output = handle_data(data)
#         # if not self.can_accept(output):
#         #     return {0: output}
#         # try:
#         cols = list(output['X'].columns)
#         cols = ["{}_lda".format(x) for x in cols]
#         result = self.pca.transform(output['X'])
#         output['X'] = pd.DataFrame(result, columns=cols[: result.shape[1]])
#         # except Exception as e:
#         #     print(e)
#         final_output = {0: output}
#         return final_output


class RandomTreesEmbeddingPrim(primitive):
    def __init__(self, random_state=0):
        super(RandomTreesEmbeddingPrim, self).__init__(name='RandomTreesEmbedding')
        self.id = 54
        self.PCA_LAPACK_Prim = []
        self.type = 'feature engineering'
        self.description = "FastICA: a fast algorithm for Independent Component Analysis."
        self.hyperparams_run = {'default': True}
        self.pca = RandomTreesEmbedding(random_state=random_state)
        self.accept_type = 'c_t'

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.pca.fit(data['X'])

    def produce(self, data):
        output = handle_data(data)
        cols = list(output['X'].columns)
        code = ''.join(word[0] for word in cols)[:10]
        result = self.pca.transform(output['X']).toarray()
        new_cols = list(map(str, list(range(result.shape[1]))))
        cols = ["{}_rfembdng{}".format(x, code) for x in new_cols]
        output['X'] = pd.DataFrame(result, columns=cols)
        final_output = {0: output}
        return final_output
# class LogFeaturesPrim(primitive):
#     def __init__(self, random_state=0):
#         super(LogFeaturesPrim, self).__init__(name='LogFeatures')
#         self.hyperparams = []
#         self.type = 'feature engineering'
#         self.description = "Generate logarithmic features."
#         self.hyperparams_run = {'default': True}
#         self.transformer = FunctionTransformer(np.square, validate=True)
#
#     def fit(self, data):
#         data = handle_data(data)
#         self.transformer.fit(data['X'])
#
#     def produce(self, data):
#         output = handle_data(data)
#         output['X'] = pd.DataFrame(self.transformer.transform(output['X']))
#         return output



