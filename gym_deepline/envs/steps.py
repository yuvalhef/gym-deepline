import pandas as pd
import numpy as np


class Step:
    def __init__(self, index, input_indices, primitive, metafeatures=None):
        self.index = index
        self.input_indices = input_indices  # [[step#, output#],[step#, output#],[step#, output#]...]
        self.primitive = primitive
        self.hyperparams_run = {'default': True}
        self.metafeatures = metafeatures
        self.vec_representation = None

    def to_vector(self, num_primitives, ind, max_inputs):
        inputs_vec = np.zeros(max_inputs) - 1
        meta_features = self.metafeatures
        primitive_ind = np.array([ind])
        for inpt in range(len(self.input_indices)):
            inputs_vec[inpt] = self.input_indices[inpt][0]  # change in envs with more than 1 output to primitive
        self.vec_representation = [inputs_vec, meta_features, primitive_ind]
