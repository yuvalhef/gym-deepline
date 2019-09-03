from __future__ import division
import numpy as np
import scipy.stats

from gym_deepline.envs.metafeatures.meta_functions.base import MetaFunction


class Entropy(MetaFunction):

    def get_numerical_arity(self):
        return 0

    def get_categorical_arity(self):
        return 1

    def get_output_type(self):
        return 'numerical'

    def get_matrix_applicable(self):
        return False

    def _calculate(self, input):
        # TODO use scipy.stats.entropy - dont know if its the correct use
        return scipy.stats.entropy(input[:,0], base=2)
        #print(input[:,0])

        #probs = [np.mean(~np.isnan(input[:,0]) == c) for c in set(~np.isnan(input[:,0]))]
        #return np.sum(-p * np.log2(p) for p in probs)
