import numpy as np

# Matrix product state with open boundary conditions
# Represented as C_(i1...iN) = A[0]^(i1) . ... . A[n-1]^(iN)
# Each A[k]^i is a matrix of size D_(k) x D_(k + 1)

 class MPS:
    # Instantiate MPS with a list of rank-3 tensors (left_bond, physical, right_bond)
    def __init__(self, tensors : list):
        self.tensors = tensors
    
    # Return the number of sites from the list of tensors
    def n(self) -> int:
        return len(self.tensors)

    # Return the physical dimension per site from the physical size of a tensor
    def p(self) -> int:
        return self.tensors[0].shape[1]
    
    # Return the list of bond dimensions between sites
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
    def __init__(self, tensors :)