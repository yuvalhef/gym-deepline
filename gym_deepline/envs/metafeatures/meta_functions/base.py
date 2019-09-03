import abc
import time
import scipy.sparse


class MetaFunction(object):
    """Abstract meta-function. Each meta-function must have the following variables pre-defined:

    get_numerical_arity: number of numerical type objects that it can take as input
    get_categorical_arity: number of categorical type objects that it can take as input
    get_output_type: "numerical" or "categorical"
    get_matrix_applicable: boolean to define if meta-function can be applied to a matrix (whole dataset). For example,
    count the number of rows (examples).
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_numerical_arity(self):
        pass

    @abc.abstractmethod
    def get_categorical_arity(self):
        pass

    @abc.abstractmethod
    def get_output_type(self):
        pass

    @abc.abstractmethod
    def get_matrix_applicable(self):
        pass

    @abc.abstractmethod
    def _calculate(self, input):
        pass

    def __call__(self, input):
        starttime = time.time()

        try:
            if scipy.sparse.issparse(input) and hasattr(self, "_calculate_sparse"):
                value = self._calculate_sparse(input)
            else:
                value = self._calculate(input)
            comment = ""
        except MemoryError as e:
            value = None
            comment = "Memory Error"

        endtime = time.time()
        calculation_time = endtime - starttime
        return value, calculation_time, comment
