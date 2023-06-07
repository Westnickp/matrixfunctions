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
            A,
            f,
            fA=None,
            sparsity_pattern=None,  # Block diagonal non-sparse? Banded?
            important_entries=None,  # Only care about diagonal entries? f(A)v? 
            normal=False,
            self_adjoint=False,
            positive_semidefinite=False
    ):
        """DOCSTRING?"""
        self.A = A
        self.f = f
        self.sparsity_pattern = sparsity_pattern
        self.important_entries = important_entries
        self.normal = False
        self.self_adjoint = False
        self.positive_semidefinite = False
        self.fA = fA if fA is not None else self._computef()

    def _computef(self):
        """Compute matrix function of current matrix with defined method"""
        # Should do something different if it is normal

        # Should maybe do something different if self adjoint

        # Should maybe do something different if positive semi-definite
        return scipy.linalg.funm(self.A, self.f)  # OTHERS?

    def rank_one_update(self, b, c=None):
        """Do Krylov subspace update for bb^T or bc^T"""
        pass

    def low_rank_update(self, B, C=None):
        """Block matrices?"""
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __add__(self):
        pass

    def __sub__(self):
        pass

    def __mult__(self):
        pass


class MatrixExponential(MatrixFunction):
    pass

class MatrixSign(MatrixFunction):
    pass

class MatrixSqrt(MatrixFunction):
    pass

class MatrixSin(MatrixFunction):
    pass

class MatrixCos(MatrixFunction):
    pass

class MatrixRationalFunction(MatrixFunction):
    pass

class MatrixPolynomial(MatrixRationalFunction):
    pass

class MatrixInverse(MatrixRationalFunction):
    pass
