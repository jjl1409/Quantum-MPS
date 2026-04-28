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
    c = psi.reshape([p] * n)
    
    tensors = []
    remaining = c.copy()

    # Handle all sites except the last one
    for site in range(n - 1):
        # Current shape is (left_bond * p, remaining_dims...)
        # We want matrix with shape (left bond * p) x (p^(N - site - 1))
        left_size = remaining.shape[0] * remaining_shape[1]

        if site == 0:
            # First site has shape (p, p^(n-1))
            mat = remaining.reshape(p, -1)
            left_bond = 1
        else:
            # Remaining sites have shape (D_left, p, p^{N - site - 1})
            left_bond = remaining_shape[0]
            mat = remaining.reshape(left_bond * p, -1)
    
        # Perform SVD to get u . s . vh
        # u has shape (m, k), vh has shape (k, n), and s has shape (k) (singular values)
        u, s, vh = np.linalg.svd(mat, full_matrices = False)
        d_bond = int(sum(s))

        # D_bond has a minimum value of 1 and is bounded by max_bond_dim
        if max_bond_dim > 0:
            d_bond = max(min(D_bond, max_bond_dim), 1)

        # Truncate to d_bond
        u = u[:, :d_bond]
        s = s[:d_bond]
        vh = vh[:d_bond, :]

        # Reshape u into a rank-3 tensor (left_bond, physical, right_bond)
        a = u.reshape(left_bond, p, d_bond)
        tensors.append(a)

        # Diagonals of s x vh gives us our remaining singular values
        # This has shape (d_bond, p^(n-site-1)) which needs to become (d_bond, p, p^(n-site-2))
        remaining = np.diag(s) @ vh
        remaining_sites = n - site - 1
        if remaining_sites > 1:
            remaining = remaining.reshape(d_bond, p, -1)
    
    # Handle last site where we are left with a rank-3 tensor (d_left, p, 1)
    tensor = remaining.reshape(remaining.shape[0], p, 1)
    tensors.append(tensor)

    return MPS(tensors)



