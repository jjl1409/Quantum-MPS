# Generates states for testing purposes
import numpy as np

# GHZ state
# |GHZ> = (|00...0> + |11...1>) / sqrt(2)
# State vector has length 2 ** n
# This should have an exact MPS representation with d = 2
def ghz_state(n: int) -> np.ndarray:
    psi = np.zeros(2**n)
    # |00...0>; first entry
    psi[0] = 1.0 / np.sqrt(2)
    # |11...1>; last entry
    psi[-1] = 1.0 / np.sqrt(2)
    return psi

# W state
# |W> = (|100...0> + |010...0> + ... + |000...1>) / sqrt(n)
# State vector has length 2 ** n
def w_state(n: int) -> np.ndarray:
    psi = np.zeros(2**n)
    for i in range(n):
        # Bitshift 1 over to get the correct index
        # Iterate through 1 to 2^(n - 1)
        idx = 1 << (n - 1 - i)
        # Thus we have indices at 1, 2, 4, ...
        psi[idx] = 1.0 / np.sqrt(n)
    return psi

# Random state
# Create a random normalized state of n p-level particles
# State vector has length p ** n
# Each entry in the state vector has a random complex probability
# p defaults to 2 (and will probably stay that way)
def random_state(n: int, p: int = 2) -> np.ndarray:
    # Seed set to 1 for consistency across tests
    rng = np.random.RandomState(1)
    psi = rng.randn(p**n) + 1j * rng.randn(p ** n)
    return psi / np.linalg.norm(psi)

# Product state (not to be confused with MPS)
# |phi>^(o * n)
# While phi can be any single site, we can choose plus |+> so every coefficient is equal
# This should have D = 1 MPS (since there is no entanglement)
def product_state(n: int) -> np.ndarray:
    # Can change this to take input instead
    plus = np.array([1, 1]) / np.sqrt(2)
    psi = plus.copy()
    # Since |phi> can be written as |+> x |+> x ... |+> we can repeatedly calculate phi for each
    # n by taking the Kronecker product of phi, plus and repeating N times
    for _ in range(n - 1):
        psi = np.kron(psi, plus)
    return psi