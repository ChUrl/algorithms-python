#!/usr/bin/env python

import random

from rich.traceback import install
install()

RAND_SIZE = 16
MOD_SIZE = 10
GEN_SIZE = 8

def isPrime(x):
    count = 0
    for i in range(int(x/2)):
        if x % (i+1) == 0:
            count = count+1
    return count == 1


# Bad idea except for toy examples
PRIMES = [i for i in range(0, 2**MOD_SIZE) if isPrime(i)]

class Party():
    def __init__(self, name, g, p):
        self.name = name
        self.random_num = random.randint(0, 2**RAND_SIZE)
        self.g = g
        self.p = p

    def secret_ingredient(self):
        return self.g**self.random_num % self.p

    def secret(self, ingredient):
        self.secret = ingredient**self.random_num % self.p


class Alice(Party):
    def __init__(self, g, p):
        Party.__init__(self, "Alice", g, p)

class Bob(Party):
    def __init__(self, g, p):
        Party.__init__(self, "Bob", g, p)

#
# Diffie-Hellman Key Exchange
#

# 1: Choose generator and prime modulus publicly
g = random.randint(0, 2**GEN_SIZE)
p = random.choice(PRIMES)

# 2: Choose secret random numbers
alice = Alice(g, p)
bob = Bob(g, p)

# 3: Calculate secret ingredient (public)
a = alice.secret_ingredient()
b = bob.secret_ingredient()

# 4: Calculate shared secret
alice.secret(b)
bob.secret(a)

print(f"Chosen g = {g} and p = {p}")
print(f"Exchanged a = {a} and b = {b}")
print(f"Calculated secrets s = {alice.secret} and s = {bob.secret}")