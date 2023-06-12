import warnings
import numpy
import scipy.linalg

from .. import utils

__all__ = [
    "MatrixFunction",
    "MatrixExponential",
    "MatrixSin",
    "MatrixCos",
    "MatrixRationalFunction",
    "MatrixPolynomial",
    "MatrixInverse"
]


class MatrixFunction:
    """DOCSTRING?"""
    def __init__(
            self,
            f,
            display_error_estimate=False,  # Block diagonal non-sparse? Banded?
    ):
        """DOCSTRING?"""
        self.f = f
        self.display_error_estimate = display_error_estimate

    def __call__(self, A, is_hermitian=False, is_positive_semidefinite=False):
        """Compute matrix function of current matrix with defined method"""
        def _evaluate_hermitian(A):
            """From scipy notes*** NEED TO CITE"""
            w, v = scipy.linalg.eigh(A, check_finite=False)  # Assume finite
            if is_positive_semidefinite:
                w = numpy.maximum(w, 0)
            w = self.f(w)
            return v @ w @ v.conj().T

        def _evaluate_general(A):
            return scipy.linalg.funm(A, self.f, self.display_error_estimate)

        if is_hermitian:
            return _evaluate_hermitian(A)
        else:
            warnings.warn("Using scipy.linalg.funm may yield poor results"
                          + " if matrix has repeated or similar eigenvalues")
            return _evaluate_general(A)

    def __str__(self):
        return str(type(self)) + ": " + str(self.f)

    def __repr__(self):
        return 

    def __add__(self):
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

class MatrixInverse(MatrixRationalFunction):
    pass
