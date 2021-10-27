import numpy as np
# https://github.com/scipy/scipy/blob/v1.7.1/scipy/fft/_basic.py#L23-L164
from scipy.fft import fft as sc_fft
from implementations import dft, fft

np.set_printoptions(precision=2, suppress=True)  # for compact output

A = np.random.random(4)
B = np.random.random(4)

print('A', A)
print('B', B)

A_dft = dft(A)
B_dft = dft(B)

print('A_dft', A_dft)
print('B_dft', B_dft)

if (np.allclose(A_dft, np.fft.fft(A)) and np.allclose(B_dft, np.fft.fft(B))):
    print("\033[92mPASSED\033[0m DFT")
else:
    print("\033[91mFAILED\033[0m DFT")

A_fft = fft(A)
B_fft = fft(B)

print('A_fft', A_fft)
print('B_fft', B_fft)


if (np.allclose(A_fft, np.fft.fft(A)) and np.allclose(B_fft, np.fft.fft(B))):
    print("\033[92mPASSED\033[0m FFT")
else:
    print("\033[91mFAILED\033[0m FFT")

C = np.multiply(A_fft, B_fft)

print("C = pointwise_multiply( A_fft, B_fft )")
print("C", C)

import rsa
 
# generate public and private keys with
# rsa.newkeys method,this method accepts
# key length as its parameter
# key length should be atleast 16
publicKey, privateKey = rsa.newkeys(2**10)

# this is the string that we will be encrypting
message = C

# rsa.encrypt method is used to encrypt
# string with public key string should be
# encode to byte string before encryption
# with encode method
encMessage = rsa.encrypt(message.tobytes(),
                        publicKey)

print("original C: ", message, message.dtype)
print("encrypted C: ", encMessage)

# the encrypted message can be decrypted
# with ras.decrypt method and private key
# decrypt method returns encoded byte string,
# use decode method to convert it to string
# public key cannot be used for decryption
decMessage = np.frombuffer(rsa.decrypt(encMessage, privateKey), dtype=np.complex128)

print("decrypted C: ", decMessage)

if np.allclose(C, decMessage):
    print("\033[92mPASSED\033[0m RSA Encyption + Decryption")
else:
    print("\033[91mFAILED\033[0m RSA Encyption + Decryption")