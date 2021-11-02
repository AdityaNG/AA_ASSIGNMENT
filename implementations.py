import numpy as np

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
    
def inv_dft(y):
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

def inv_fft(y):
    if len(y) <= 2:
        return my_inv_dft(y)
    elif len(y)%2 != 0:
        raise ValueError("must be a power of 2")
    else:
        N = len(y)
        w = np.exp(2j*np.pi/N)
        ae = inv_dft(y[::2])
        ao = inv_dft(y[1::2])
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
    
    C = [i.real for i in inv_fft(y_c)]
    return C
    
    
    
def elementry_multi(A,B):
    ''' Input 
        A: a0 a1 a2 ... a(n-2) a(n-1)
        B: b0 b1 b2 ... b(m-2) b(m-1)
        Returns:
        C: c0 c1 c2 ... c(m+n-2) c(m+n-1) which is A*B (convolution)'''
    n = len(a)
    m = len(b)
    l = m + n    #The number of values in C 
    C = [0 for i in range(l)]
    for i in range(n):
        for j in range(m):
            C[m+n] += A[i]*B[j]
        
    return C
    
    
    
    
    
    
    
    
    