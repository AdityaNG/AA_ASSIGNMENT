import time
import numpy as np
import binascii
from scipy.fft import fft as sc_fft
from implementations import dft, fft, ifft, ifft2, chooseKeys, encrypt, decrypt

np.set_printoptions(precision=2, suppress=True)  # for compact output

def main(n):

    print("\033[96mINFO\033[0m IFFT")

    A = np.random.random(n)
    B = np.random.random(n)

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

    message = "Secret Message" # C

    print("Message to be encrypted = ", message)
    print('Encrypting...')
    C_encrypted = encrypt(message)
    print(C_encrypted)
    
    #print("\033[96mINFO\033[0m C_encrypted =", binascii.hexlify(C_encrypted))
    print("\033[96mINFO\033[0m C_encrypted =", C_encrypted)
    
    print('Decryption...')
    C_decrypted = decrypt(C_encrypted)
    print(C_decrypted)

    if np.allclose(C, C_decrypted):
        print("\033[92mPASSED\033[0m RSA Encyption + Decryption")
    else:
        print("\033[91mFAILED\033[0m RSA Encyption + Decryption")


    # TODO : Inverse FFT
    A_ifft = np.zeros_like(A)
    B_ifft = np.zeros_like(B)

    A_ifft = ifft(A_fft)
    B_ifft = ifft(B_fft)

    print(A)
    print(A_ifft)

    if np.allclose(A, A_ifft) and np.allclose(B, B_ifft):
        print("\033[92mPASSED\033[0m IFFT")
    else:
        print("\033[91mFAILED\033[0m IFFT")

if __name__=='__main__':
    timings = {'x':[], 'y':[]}
    for n in [2**i for i in range(2, 12)]:
        start = time.time()
        print("n=", n)
        main(n)
        print()
        print("*"*10)
        end = time.time()
        timings['x'].append(n)
        timings['y'].append(end-start)    
    print(timings)