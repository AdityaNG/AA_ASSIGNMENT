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
    n = len(a)
    m = len(b)
    l = m + n    #The number of values in C 
    C = [0 for i in range(l)]
    for i in range(n):
        for j in range(m):
            C[m+n] += A[i]*B[j]
        
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

def chooseE(totient):
    """
    Chooses a random number, 1 < e < totient, and checks whether or not it is 
    coprime with the totient, that is, gcd(e, totient) = 1
    """
    while (True):
        e = random.randrange(2, totient)

        if (gcd(e, totient) == 1):
            return e

def chooseKeys():
    """
    Selects two random prime numbers from a list of prime numbers which has 
    values that go up to 100k. It creates a text file and stores the two 
    numbers there where they can be used later. Using the prime numbers, 
    it also computes and stores the public and private keys in two separate 
    files.
    """

    # choose two random numbers within the range of lines where 
    # the prime numbers are not too small and not too big
    rand1 = random.randint(100, 300)
    rand2 = random.randint(100, 300)

    # store the txt file of prime numbers in a python list
    fo = open('primes-to-100k.txt', 'r')
    lines = fo.read().splitlines()
    fo.close()

    # store our prime numbers in these variables
    prime1 = int(lines[rand1])
    prime2 = int(lines[rand2])

    # compute n, totient, e
    n = prime1 * prime2
    totient = (prime1 - 1) * (prime2 - 1)
    e = chooseE(totient)

    # compute d, 1 < d < totient such that ed = 1 (mod totient)
    # e and d are inverses (mod totient)
    gcd, x, y = xgcd(e, totient)

    # make sure d is positive
    if (x < 0):
        d = x + totient
    else:
        d = x

    # write the public keys n and e to a file
    f_public = open('public_keys.txt', 'w')
    f_public.write(str(n) + '\n')
    f_public.write(str(e) + '\n')
    f_public.close()

    f_private = open('private_keys.txt', 'w')
    f_private.write(str(n) + '\n')
    f_private.write(str(d) + '\n')
    f_private.close()
    
    return n, e, d

def encrypt(message, file_name = 'public_keys.txt', block_size = 2):
    """
    Encrypts a message (string) by raising each character's ASCII value to the 
    power of e and taking the modulus of n. Returns a string of numbers.
    file_name refers to file where the public key is located. If a file is not 
    provided, it assumes that we are encrypting the message using our own 
    public keys. Otherwise, it can use someone else's public key, which is 
    stored in a different file.
    block_size refers to how many characters make up one group of numbers in 
    each index of encrypted_blocks.
    """

    try:
        fo = open(file_name, 'r')

    # check for the possibility that the user tries to encrypt something
    # using a public key that is not found
    except FileNotFoundError:
        print('That file is not found.')
    else:
        n = int(fo.readline())
        e = int(fo.readline())
        fo.close()

        encrypted_blocks = []
        ciphertext = -1

        if (len(message) > 0):
            # initialize ciphertext to the ASCII of the first character of message
            ciphertext = ord(message[0])

        for i in range(1, len(message)):
            # add ciphertext to the list if the max block size is reached
            # reset ciphertext so we can continue adding ASCII codes
            if (i % block_size == 0):
                encrypted_blocks.append(ciphertext)
                ciphertext = 0

            # multiply by 1000 to shift the digits over to the left by 3 places
            # because ASCII codes are a max of 3 digits in decimal
            ciphertext = ciphertext * 1000 + ord(message[i])

        # add the last block to the list
        encrypted_blocks.append(ciphertext)

        # encrypt all of the numbers by taking it to the power of e
        # and modding it by n
        for i in range(len(encrypted_blocks)):
            encrypted_blocks[i] = str((encrypted_blocks[i]**e) % n)

        # create a string from the numbers
        encrypted_message = " ".join(encrypted_blocks)

        return encrypted_message

def decrypt(blocks, block_size = 2):
    """
    Decrypts a string of numbers by raising each number to the power of d and 
    taking the modulus of n. Returns the message as a string.
    block_size refers to how many characters make up one group of numbers in
    each index of blocks.
    """

    fo = open('private_keys.txt', 'r')
    n = int(fo.readline())
    d = int(fo.readline())
    fo.close()

    # turns the string into a list of ints
    list_blocks = blocks.split(' ')
    int_blocks = []

    for s in list_blocks:
        int_blocks.append(int(s))

    message = ""

    # converts each int in the list to block_size number of characters
    # by default, each int represents two characters
    for i in range(len(int_blocks)):
        # decrypt all of the numbers by taking it to the power of d
        # and modding it by n
        int_blocks[i] = (int_blocks[i]**d) % n
        
        tmp = ""
        # take apart each block into its ASCII codes for each character
        # and store it in the message string
        for c in range(block_size):
            tmp = chr(int_blocks[i] % 1000) + tmp
            int_blocks[i] //= 1000
        message += tmp

    return message