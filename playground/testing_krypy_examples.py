import krypy
import numpy
import matplotlib.pyplot as plt

A = numpy.diag([1.0e-3] + list(range(2, 101)))
b = numpy.random.rand(100)

# sol, out = krypy.cg(A, b)
# sol, out = krypy.minres(A, b)
sol, out = krypy.gmres(A, b, tol=1e-7)

# sol is None if no solution has been found
# out.resnorms the relative residual norms and some more data

# plot residuals
plt.figure(1)
plt.semilogy(out.resnorms)
plt.show()



