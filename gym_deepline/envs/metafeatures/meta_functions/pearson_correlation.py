from ..meta_functions.base import MetaFunction
from scipy.stats import pearsonr
import numpy as np

class PearsonCorrelation(MetaFunction):

    def get_numerical_arity(self):
        return 2

    def get_categorical_arity(self):
        return 0

    def get_output_type(self):
        return 'numerical'

    def get_matrix_applicable(self):
        return False

    def _calculate(self, input):
        input = input[~np.isnan(input).any(axis=1)]

        return pearsonr(input[:,0], input[:,1])