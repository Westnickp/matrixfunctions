import numpy as np
import scipy.linalg
from krypy.utils import arnoldi

A = np.diag(np.random.rand(20)) 
# print(A)
# print(scipy.linalg.expm(A))

b = np.ones((20,1))
# print(b)

# Kryl = arnoldi(A, b, 3)
# print(Kryl[0])
# print(Kryl[1])

f = scipy.linalg.expm
fA = f(A)
A2 = A+(b@b.transpose())
fA2 = scipy.linalg.expm(A2)
error = np.zeros(10)
for k in range(1, 11):
    Q, H = arnoldi(A, b, k)
    size_b = np.linalg.norm(b, 2)
    e1 = np.zeros((k, 1))
    e1[0, 0] = 1
    Xmf = f(H[:k, :k] + (size_b**2) * (e1 @ e1.transpose())) - f(H[:k, :k])
    fAUpdate = fA + (Q[:, :k] @ Xmf @ Q[:, :k].transpose())
    error[k-1] = np.linalg.norm(fAUpdate - fA2, 2)

rel_error = error/np.linalg.norm(fA2,2)

print("Errors: ", error)
print("Relative Errors: ", rel_error)
