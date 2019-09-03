import numpy as np
import scipy.stats

from gym_deepline.envs.metafeatures.meta_functions.base import MetaFunction



class Mean(MetaFunction):
    def __init__(self):
        """Computes the mean of a set of values."""
        pass

    def get_numerical_arity(self):
        return 1

    def get_categorical_arity(self):
        return 0

    def get_output_type(self):
        return 'numerical'

    def get_matrix_applicable(self):
        return False

    def _calculate(self, input):
        return np.mean(input[~np.isnan(input)])


class StandardDeviation(MetaFunction):
    def __init__(self):
        """Computes the standard deviation of a set of values."""
        pass

    def get_numerical_arity(self):
        return 1

    def get_categorical_arity(self):
        return 0

    def get_output_type(self):
        return 'numerical'

    def get_matrix_applicable(self):
        return False

    def _calculate(self, input):
        return np.std(input[~np.isnan(input)])


class Kurtosis(MetaFunction):
    def __init__(self):
        """Computes the kurtosis of a set of values."""
        pass

    def get_numerical_arity(self):
        return 1

    def get_categorical_arity(self):
        return 0

    def get_output_type(self):
        return 'numerical'

    def get_matrix_applicable(self):
        return False

    def _calculate(self, input):
        return scipy.stats.kurtosis(input[~np.isnan(input)])


class Skew(MetaFunction):
    def __init__(self):
        """Computes the skew of a set of values."""
        pass

    def get_numerical_arity(self):
        return 1

    def get_categorical_arity(self):
        return 0

    def get_output_type(self):
        return 'numerical'

    def get_matrix_applicable(self):
        return False

    def _calculate(self, input):
        return scipy.stats.skew(input[~np.isnan(input)])


class MissingValues(MetaFunction):
    def __init__(self):
        """Computes the missing values count of a set of values."""
        pass

    def get_numerical_arity(self):
        return 1

    def get_categorical_arity(self):
        return 1

    def get_output_type(self):
        return 'numerical'

    def get_matrix_applicable(self):
        return False

    def _calculate(self, input):
        return np.size(~np.isnan(input[:,0]))#- np.count_nonzero(input[:,0])
        #return np.mean(~np.isnan(input[:,0]))