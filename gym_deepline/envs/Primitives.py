import pandas as pd


class primitive:
    def __init__(self, name):
        self.name = name
        self.description = str(name)
        self.hyperparams = []
        self.type = "primitive"
        self.hyperparams_run = {}

    def fit(self, data):
        pass

    def produce(self, data):
        pass

    def can_accept(self, data):
        pass

    def can_accept_a(self, data):
        if data['X'].empty:
            return False
        elif data['X'].shape[1] == 0:
            return False
        num_cols = data['X']._get_numeric_data().columns
        if not len(num_cols) == 0:
            return True
        return False

    def can_accept_b(self, data):
        if data['X'].empty:
            return False
        elif data['X'].shape[1] == 0:
            return False
        return True

    def can_accept_c(self, data, task=None, larpack=False):
        if data['X'].empty:
            return False
        elif data['X'].shape[1] == 0:
            return False
        cols = data['X']
        num_cols = data['X']._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if task:
            if not data['learning_job'].task == task:
                return False
        if larpack:
            if min(data['X'].shape[0], data['X'].shape[1]) - 1 == 0:
                return False
        # if data['X'].isnull().any().any():
        #     return False
        with pd.option_context('mode.use_inf_as_null', True):
            if data['X'].isnull().any().any():
                return False
        if not len(cat_cols) == 0:
            return False
        return True

    def can_accept_d(self, data, task):
        if data['X'].empty:
            return False
        elif data['X'].shape[1] == 0:
            return False
        cols = data['X']
        num_cols = data['X']._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if not len(cat_cols) == 0:
            return False
        if not data['learning_job'].task == task or data['X'].isnull().any().any():
            return False
        elif data['X'].lt(0).sum().sum() > 0:
            return False
        return True

    def can_accept_e(self, data, task):
        if data['X'].empty:
            return False
        elif data['X'].shape[1] == 0:
            return False
        cols = data['X'].columns
        if not all('Pred' in s for s in list(cols)):
            return False
        num_cols = data['X']._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if not data['learning_job'].task == task or data['X'].isnull().any().any():
            return False
        elif not len(cat_cols) == 0:
            return False
        return True


    def is_needed(self, data):
        pass
