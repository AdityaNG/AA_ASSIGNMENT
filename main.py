import numpy as np
import binascii
from scipy.fft import fft as sc_fft
from implementations import dft, fft

np.set_printoptions(precision=2, suppress=True)  # for compact output

print("\033[96mINFO\033[0m IFFT")

A = np.random.random(4)
B = np.random.random(4)

print("\033[96mINFO\033[0m A =", A)
print("\033[96mINFO\033[0m B =", B)

A_dft = dft(A)
B_dft = dft(B)

print("\033[96mINFO\033[0m A_dft =", A_dft)
print("\033[96mINFO\033[0m B_dft =", B_dft)

if (np.allclose(A_dft, np.fft.fft(A)) and np.allclose(B_dft, np.fft.fft(B))):
    print("\033[92mPASSED\033[0m DFT")
else:
    print("\033[91mFAILED\033[0m DFT")

A_fft = fft(A)
B_fft = fft(B)

print("\033[96mINFO\033[0m A_fft =", A_fft)
print("\033[96mINFO\033[0m B_fft =", B_fft)


if (np.allclose(A_fft, np.fft.fft(A)) and np.allclose(B_fft, np.fft.fft(B))):
    print("\033[92mPASSED\033[0m FFT")
else:
    print("\033[91mFAILED\033[0m FFT")

C = np.multiply(A_fft, B_fft)

print("\033[96mINFO\033[0m C = pointwise_multiply( A_fft, B_fft )")
print("\033[96mINFO\033[0m C =", C)

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
C_encrypted = rsa.encrypt(message.tobytes(),
                        publicKey)

print("\033[96mINFO\033[0m C_encrypted =", binascii.hexlify(C_encrypted))

# the encrypted message can be decrypted
# with ras.decrypt method and private key
# decrypt method returns encoded byte string,
# use decode method to convert it to string
# public key cannot be used for decryption
C_decrypted = np.frombuffer(rsa.decrypt(C_encrypted, privateKey), dtype=np.complex128)

print("\033[96mINFO\033[0m C_decrypted =", C_decrypted)

if np.allclose(C, C_decrypted):
    print("\033[92mPASSED\033[0m RSA Encyption + Decryption")
else:
    print("\033[91mFAILED\033[0m RSA Encyption + Decryption")


# TODO : Inverse FFT
A_ifft = np.zeros_like(A)
B_ifft = np.zeros_like(B)

if np.allclose(A, A_ifft) and np.allclose(A, A_ifft):
    print("\033[92mPASSED\033[0m IFFT")
else:
    print("\033[91mFAILED\033[0m IFFT")