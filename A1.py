import numpy as np
# https://github.com/scipy/scipy/blob/v1.7.1/scipy/linalg/special_matrices.py#L972-L1039
from scipy.linalg import dft as sc_dft
from implementations import dft
#np.set_printoptions(precision=2, suppress=True)  # for compact output

#"""
m = sc_dft(5)
x = np.array([1, 2, 3, 0, 3])

sc_ft = m @ x

print(sc_ft) # Compute the DFT of x

print(np.allclose(dft(x), sc_ft))
#"""

x = np.random.random(1024)
print(np.allclose(dft(x), np.fft.fft(x)))