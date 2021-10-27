import numpy as np
# https://github.com/scipy/scipy/blob/v1.7.1/scipy/fft/_basic.py#L23-L164
from scipy.fft import fft as sc_fft
from implementations import dft, fft

#np.set_printoptions(precision=2, suppress=True)  # for compact output


#x = np.array([1, 2, 3, 0, 3])
#print(fft(x)) # Compute the DFT of x

x = np.random.random(1024)
print(np.allclose(fft(x), np.fft.fft(x)))
