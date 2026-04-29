# Generates states for testing purposes
import numpy as np

# GHZ state
# |GHZ> = (|00...0> + |11...1>) / sqrt(2)
# This should have an exact MPS representation with d = 2
def ghz_state(n: int) -> np.ndarray:
    psi = np.zeros(2**n)
    # |00...0>; first entry
    psi[0] = 1.0 / np.sqrt(2)
    # |11...1>; last entry
    psi[-1] = 1.0 / np.sqrt(2)

# W state
# |W> = (|100...0> + |010...0> + ... + |000...1>) / sqrt(n)
def w_state(n: int) -> np.ndarray:
    psi = np.zeros(2**n)
    for i in range(n):
        # Bitshift 1 over to get the correct index
        # Iterate through 1 to 2^(n - 1)
        idx = 1 << (n - 1 - i)
        # Thus we have indices at 1, 2, 4, ...
        psi[idx] = 1.0 / np.sqrt(N)
    return psi

# Random state
# Create a random normalized state of n p-level particles
# p defaults to 2 (and will probably stay that way)
def random_state(n: int, p: int = 2)
    # Seed set to 1 for consistency across tests
    rng = np.random.RandomState(1)
    psi = 