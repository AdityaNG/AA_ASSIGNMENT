import numpy as np
import random

# 
def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    #print(N, x)
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])
    
def idft(y):
    if bin(len(y))[2:].count('1')>1:
        raise ValueError("must be power of 2")
    else:
        N = len(y)
        y = np.array(y)
        y.reshape((N,1))
        Winv = np.array([[np.exp(2j*np.pi*i*j/N)/N for j in range(N)] for i in range(N)])
        A = np.matmul(Winv,y)
        A = A.flatten()
        return A

def ifft(y):
    if len(y) <= 2:
        return idft(y)
    elif len(y)%2 != 0:
        raise ValueError("must be a power of 2")
    else:
        N = len(y)
        w = np.exp(2j*np.pi/N)
        ae = ifft(y[::2])
        ao = ifft(y[1::2])
        a = [0 for i in range(N)]
        for i in range(N//2):
            a[i] = (ae[i] + (w**i)*ao[i])/2
            a[i+N//2] = (ae[i] - (w**i)*ao[i])/2
        
        return a
        
  
def fast_multi(a,b):
    ''' Input 
        A: a0 a1 a2 ... a(n-2) a(n-1)
        B: b0 b1 b2 ... b(m-2) b(m-1)
        Returns:
        C: c0 c1 c2 ... c(m+n-2) c(m+n-1) which is A*B (convolution)'''
    # The below code padds a,b with required number of 0's
    m = len(a)
    n = len(b)
    l = m+n
    n_l = 2**int(np.ceil(np.log(l)/np.log(2)))
    a_p2 = np.concatenate((a, np.zeros(n_l-m)))
    b_p2 = np.concatenate((b ,np.zeros(n_l-n)))
    
    y_a = fft(a_p2)
    y_b = fft(b_p2)
    y_c = [y_a[i] * y_b[i] for i in range(n_l)]
    
    C = [i.real for i in ifft(y_c)]
    return C
    
    
    
def elementry_multi(A,B):
    ''' Input 
        A: a0 a1 a2 ... a(n-2) a(n-1)
        B: b0 b1 b2 ... b(m-2) b(m-1)
        Returns:
        C: c0 c1 c2 ... c(m+n-2) c(m+n-1) which is A*B (convolution)'''
    n = len(A)
    m = len(B)
    l = m + n    #The number of values in C 
    C = [0 for i in range(l)]
    for i in range(n):
        for j in range(m):
            C[i+j] += A[i]*B[j]
        
    return C
    
    
def fft2(A):
    '''
        A: 2D matrix of dimensions m x n
        Returns: fft(A)
        Raises: Value Error if:
                    i) A is not 2 dimensional
                    ii) The num of attributes of A is not a power of 2
        
        Note: Uses Numpy's 1d fft implementation
    '''
    if len(A.shape) != 2:
        raise ValueError("Input must be of 2 dimensions")
        
    elif [np.log2(i) for i in A.shape] != [int(np.log2(i)) for i in A.shape]:
        raise ValueError("Dimensions must be a power of 2")

    y_r = np.array([np.fft.fft(row) for row in A])
    y_rc = np.array([np.fft.fft(row) for row in y_r.T]).T
    
    return y_rc
    
    
    
def ifft2(y):
    '''
        y: 2D matrix of dimensions m x n
        Returns: ifft2(y)
        Raises: Value Error if:
                    i) y is not 2 dimensional
                    ii) The num of attributes of y is not a power of 2
        Note: Uses ifft implemented here
    '''
    if len(y.shape) != 2:
        raise ValueError("Input must be of 2 dimensions")
    elif [np.log2(i) for i in y.shape] != [int(np.log2(i)) for i in y.shape]:
        raise ValueError("Dimensions must be a power of 2")

    A_r = np.array([ifft(row) for row in y])
    A_rc = np.array([ifft(row) for row in A_r.T]).T
    
    return A_rc
    

"""
Implementation of the RSA algorithm.
It randomly selects two prime numbers from a txt file of prime numbers and 
uses them to produce the public and private keys. Using the keys, it can 
either encrypt or decrypt messages.
"""

def gcd(a, b):
    """
    Performs the Euclidean algorithm and returns the gcd of a and b
    """
    if (b == 0):
        return a
    else:
        return gcd(b, a % b)

def xgcd(a, b):
    """
    Performs the extended Euclidean algorithm
    Returns the gcd, coefficient of a, and coefficient of b
    """
    x, old_x = 0, 1
    y, old_y = 1, 0

    while (b != 0):
        quotient = a // b
        a, b = b, a - quotient * b
        old_x, x = x, old_x - quotient * x
        old_y, y = y, old_y - quotient * y

    return a, old_x, old_y

def horner(a, x):
    result = 0
    for i in range(len(a)-1, -1, -1):
        result = a[i] + (x * result)
    return result

def getPvForm(A):
    pv = []
    for x in range(1, len(A) + 1):
        y = horner(A, x)
        pv.append((x, y))
    return pv

def polynomial_multiplication(P, Q):
    """
        Brute force multiplication
    """
    m = len(P)
    n = len(Q)
    result = [0]*(m+n-1)
    for i in range(m):
        for j in range(n):
            result[i+j] += P[i]*Q[j]
    return result

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
