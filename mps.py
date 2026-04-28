import numpy as np

# Matrix product state with open boundary conditions
# Represented as C_(i1...iN) = A[0]^(i1) . ... . A[n-1]^(iN)
# Each A[k]^i is a matrix of size D_(k) x D_(k + 1)

 class MPS:
    # Instantiate MPS with a list of rank-3 tensors (left_bond, physical, right_bond)
    def __init__(self, tensors: list):
        self.tensors = tensors
    
    # Return the number of sites from the list of tensors
    def n(self) -> int:
        return len(self.tensors)

    # Return the physical dimension per site from the physical size of a tensor
    def p(self) -> int:
        return self.tensors[0].shape[1]
    
    # Return the list of bond dimensions between sites
    # This is done by collecting right bond dimensions for each site except the last
    def bond_dims(self) -> list:
        return [t.shape[2] for t in self.tensors[:-1]]

    # Return the maximum bond dimension across all bond dimensions
    def max_bond_dim(self) -> int:
        return max(self.bond_dims)

# MPS in Vidal's canonical form (Gamma-Lambda)
# Represented as C_(i1...in) = Gamma(0)^(i1) . Lambda(0) ... Gamma(n)^(in) . Lambda(n)
# Bond indexes label Schmidt vectors, and Lambda(k) contains the Schmidt coefficients for the
# bipartition at bond k

class CanonicalMPS:
    # Instantiate canonical MPS from a list of gammas and lambdas
    # Gammas are a list of n tensors (left_bond, physical, right_bond)
    # Lambdas are a list of n - 1 lambda vectors (Schmidt coefficients)
    def __init__(self, gammas: list, lambdas: list):
        self.gammas = gammas
        self.lambdas = lambdas

    # Return the number of sites for the list of vectors
    def n(self) -> int:
        return len(self.gammas)
    
    # Returns the list of bond dimensions between sites
    # Note that this is equivalent to the bond_dims calculation for MPS (just semantics)
    def bond_dims(self) -> list:
        return [l.shape[0] in self.lambdas]

# Decompose an N-particle quantum state into MPS using successive SVDs, returning an MPS
# Successively performs Schmidt decompositions between site 1 and the rest, then site 2 and so on...
# psi is a state vector of length p^Nn (aka a rank-n tensor with each index of dimension p)
# p is the physical dimension per site
# max_bond_dim is the maximum bond dimension d, where signular values beyond rank d are discarded.
# Defaults to 0, which results in exact decomposition
def mps_decomposition(psi: np.ndarray, p: int, max_bond_dim : int = 0) -> MPS:
    # Total dimensions should be p * n
    total_dim = np.prod(psi.shape)
    # Take log base p of total_dim to get n
    n = int(round(np.log(total_dim) / np.log(p)))
    # Reshape into a rank-n tensor
    c = psi.reshape([p] * N)
    
    tensors = []
    remaining = C.copy()

    
