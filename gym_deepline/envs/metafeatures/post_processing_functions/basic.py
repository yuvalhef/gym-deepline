import numpy as np
import scipy.stats

from ..post_processing_functions.base import PostProcessing

class NonAggregated(PostProcessing):

    def __init__(self):
        """Does nothing!"""
        pass

    def get_input_types(self):
        return 'numerical'

    def get_input_arity(self):
        return 'one'

    def _calculate(self, input):
        return input[0]

class Mean(PostProcessing):
    def __init__(self):
        """Computes the mean of a set of values."""
        pass

    def get_input_types(self):
        return 'numerical'

    def get_input_arity(self):
        return 'n'

    def _calculate(self, input):
        return np.mean(input)


class StandardDeviation(PostProcessing):
    def __init__(self):
        """Computes the standard deviation of a set of values."""
        pass

    def get_input_types(self):
        return 'numerical'

    def get_input_arity(self):
        return 'n'

    def _calculate(self, input):
        return np.std(input)


class Kurtosis(PostProcessing):
    def __init__(self):
        """Computes the kurtosis of a set of values."""
        pass

    def get_input_types(self):
        return 'numerical'

    def get_input_arity(self):
        return 'n'

    def _calculate(self, input):
        return scipy.stats.kurtosis(input)


class Skew(PostProcessing):
    def __init__(self):
        """Computes the skew of a set of values."""
        pass

    def get_input_types(self):
        return 'numerical'

    def get_input_arity(self):
        return 'n'

    def _calculate(self, input):
        return scipy.stats.skew(input)