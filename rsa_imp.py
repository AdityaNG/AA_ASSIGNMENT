"""
Implementation of the RSA algorithm.
It randomly selects two prime numbers from a txt file of prime numbers and 
uses them to produce the public and private keys. Using the keys, it can 
either encrypt or decrypt messages.
"""
import numpy as np
import binascii
import random

import math
from Crypto.Util import number

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
    for i in range(3, totient, 2):
        if (math.gcd(i, totient) == 1):
            return i

def chooseKeys(bits=64):
    #bits = int(input("Enter number of bits in RSA key: "))
    bits >>= 1
    # prime1 = number.getPrime(bits)
    # prime2 = number.getPrime(bits)
    prime1 = getPrime(bits)
    prime2 = getPrime(bits)
    while (prime2 == prime1):
        # prime2 = number.getPrime(bits)
        prime2 = getPrime(bits)
    n = prime1 * prime2
    totient = (prime1 - 1) * (prime2 - 1)
    e = chooseE(totient)
    _, x, y = xgcd(e, totient)
    d = ((x + totient) % totient)

    return {'public_key': (e, n), 'private_key': (d, n)}

def encrypt(message, key):
    message = str(message)

    e, n = key
    encrypted_blocks = []

    for i in message:
        encrypted_blocks.append(str(binary_exponentiation(ord(i), e, n)))

    encrypted_message = " ".join(encrypted_blocks)

    return encrypted_message

def decrypt(blocks, key):
    d, n = key

    list_blocks = blocks.split(' ')
    message = ""
    for i in range(len(list_blocks)):
        message += chr(binary_exponentiation(int(list_blocks[i]), d, n))

    return eval(message)

    
def nBitRandom(n):
    start = binary_exponentiation(2, n - 1, -1)
    return random.randrange(start+1, 2*start - 1)

def getLowLevelPrime(n):
    while True:
        pc = nBitRandom(n)
        for divisor in first_primes_list:
            if pc % divisor == 0 and divisor**2 <= pc:
                break
        else: return pc

def isMillerRabinPassed(mrc):
    maxDivisionsByTwo = 0
    ec = mrc-1
    while ec % 2 == 0:
        ec >>= 1
        maxDivisionsByTwo += 1
    assert(2**maxDivisionsByTwo * ec == mrc-1)

    def trialComposite(round_tester):
        if pow(round_tester, ec, mrc) == 1:
            return False
        for i in range(maxDivisionsByTwo):
            if pow(round_tester, 2**i * ec, mrc) == mrc-1:
                return False
        return True
    numberOfRabinTrials = 20
    for i in range(numberOfRabinTrials):
        round_tester = random.randrange(2, mrc)
        if trialComposite(round_tester):
            return False
    return True

def getPrime(n):
    while(True):
        prime_candidate = getLowLevelPrime(n)
        if not isMillerRabinPassed(prime_candidate):
            continue
        else:
            return prime_candidate

first_primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                     31, 37, 41, 43, 47, 53, 59, 61, 67,
                     71, 73, 79, 83, 89, 97, 101, 103,
                     107, 109, 113, 127, 131, 137, 139,
                     149, 151, 157, 163, 167, 173, 179,
                     181, 191, 193, 197, 199, 211, 223,
                     227, 229, 233, 239, 241, 251, 257,
                     263, 269, 271, 277, 281, 283, 293,
                     307, 311, 313, 317, 331, 337, 347, 349]

def binary_exponentiation(base, power, mod):
    result = 1
    while (power):
        if (power & 1):
            if (mod != -1):
                result = (result * base) % mod
            else:
                result *= base
        if (mod != -1):
            base = (base * base) % mod
        else:
            base *= base
        power >>= 1
    return result
