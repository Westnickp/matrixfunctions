import warnings
import numpy
import scipy.linalg
import numbers

__all__ = [
    "MatrixFunction",
    "MatrixExponential",
    "MatrixSin",
    "MatrixCos",
    "MatrixInverse"
]


class MatrixFunction:
    """DOCSTRING?"""
    def __init__(
            self,
            f,
            f_description=None,
            display_error_estimate=False
    ):
        """DOCSTRING?"""
        self.f = numpy.vectorize(f)
        self.f_description = f_description
        self.display_error_estimate = display_error_estimate

    def evaluate_hermitian(self, A, is_positive_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        w, v = scipy.linalg.eigh(A, check_finite=False)  # Assume finite
        if is_positive_semidefinite:
            w = numpy.maximum(w, 0)
        w = self.f(w)
        return (v * w) @ v.conj().T

    def evaluate_general(self, A):
        return scipy.linalg.funm(A,
                                 self.f,
                                 not self.display_error_estimate)

    def get_error_estimate(self, A):
        # Method for error estimation for general method
        # TODO
        pass

    def __call__(self, A, is_hermitian=False, is_positive_semidefinite=False):
        """Compute matrix function of current matrix with defined method"""

        if is_hermitian:
            return self.evaluate_hermitian(A, is_positive_semidefinite)
        else:
            return self.evaluate_general(A)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.f_description:
            return str(type(self).__name__) + "({})".format(self.f_description)
        else:
            return str(type(self).__name__) + "({})".format(str(self.f))

    def __add__(self, g):
        pass

    def __sub__(self):
        pass

    def __mult__(self):
        pass


class MatrixExponential(MatrixFunction):
    pass


class MatrixSin(MatrixFunction):
    pass


class MatrixCos(MatrixFunction):
    pass


class MatrixInverse(MatrixFunction):
    pass
