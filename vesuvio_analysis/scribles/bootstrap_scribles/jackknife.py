import numpy as np

# To do the Jackknife, need to delete column of 2d array at ith position

A = np.arange(25).reshape((5, 5))
print(A)

for j in range(A[0].size):
    B = np.delete(A, j, axis=1)
    print(B)

C = np.append(B, np.zeros((len(B), 1)), axis=1)
print(C)