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
            display_error_estimate=False,
            implementation="schur-parlett"
    ):
        """DOCSTRING?"""
        self.f = numpy.vectorize(f)
        self.f_description = f_description
        self.display_error_estimate = display_error_estimate
        self.implementation = implementation

    def evaluate_general(self, A):
        if self.implementation == "schur-parlett":
            return scipy.linalg.funm(A,
                                     self.f,
                                     not self.display_error_estimate)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def evaluate_hermitian(self, A, is_positive_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        w, v = scipy.linalg.eigh(A, check_finite=False)  # Assume finite
        if is_positive_semidefinite:
            w = numpy.maximum(w, 0)
        w = self.f(w)
        return (v * w) @ v.conj().T

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
        if isinstance(g, numbers.Number):
            def temp_f(x: any) -> any: self.f(x) + g
            if self.f_description:
                temp_f_description = self.f_description + " + " + str(g)
            else:
                temp_f_description = None
            return MatrixFunction(temp_f,
                                  temp_f_description,
                                  self.display_error_estimate)
        elif isinstance(g, type(self)):
            def temp_f(x: any) -> any: self.f(x) + g.f(x)
            if g.f_description and self.f_description:
                temp_f_description = (self.f_description + " + "
                                      + g.f_description)
            else:
                temp_f_description = None
            d = max(self.display_error_estimate, g.display_error_estimate)
            return MatrixFunction(temp_f, temp_f_description, d)
        else:
            def temp_f(x: any) -> any: return self.f(x) + g(x)
            d = self.display_error_estimate
            return MatrixFunction(temp_f,
                                  display_error_estimate=d)

    def __radd__(self, g):
        return self.__add__(g)

    def __sub__(self):
        pass

    def __mult__(self):
        pass

    def __div__(self):
        pass

class MatrixExponential(MatrixFunction):
    f_description = "numpy.exp(x)"
    f = numpy.exp

    def __init__(self,
                 display_error_estimate=False,
                 implementation="scale-and-square"):
        self.display_error_estimate = display_error_estimate
        self.implementation = implementation

    def evaluate_general(self, A):
        if self.implementation == "scale-and-square":
            return scipy.linalg.expm(A)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))

    def evaluate_hermitian(self, A, is_positive_semidefinite=False):
        """From scipy notes*** NEED TO CITE"""
        if self.implementation == "scale-and-square":
            return scipy.linalg.expm(A)
        elif self.implementation == "diagonal":
            super().evaluate_hermitian(A, is_positive_semidefinite)
        else:
            raise NotImplementedError("Unknown evaluation algorithm:"
                                      + " {}".format(self.implementation))



class MatrixSin(MatrixFunction):
    f_description = "numpy.sin(x)"
    pass


class MatrixCos(MatrixFunction):
    f_description = "numpy.cos(x)"
    pass


class MatrixInverse(MatrixFunction):
    f_description = "1/x"
    pass
