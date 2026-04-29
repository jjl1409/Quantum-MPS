# Contains MPS and Canonical MPS class definition
# Also contains functions for matrix product state decomposition
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

# Convert an MPS to canonical form
# Bond index gamma corresponds to Schmidt vectors, and lambda vectors contain Schmidt coefficients
# Takes in MPS and returns a CanonicalMPS
# Used to calculate entanglement entropy and local expectation values

def mps_to_canonical(mps: MPS) -> CanonicalMPS:
    n = mps.n
    tensors = [t.copy() for t in mps.tensors]

    # Iterate from left to right (skipping last one) to get into left-canonical form
    # Reshape and perform QR decomposition at each site to make left-orthogonal
    for i in range(N - 1):
        # a should have shape (d_left, p, d_right)
        a = tensors[i]
        d_left, p_dim, d_right = a.shape

        # Reshape to matrix of form (d_left * p) x d_right
        mat = A.reshape(d_left * p_dim, d_right)

        # Perform QR decomposition and backfill to tensors
        q, r = np.linalg.qr(mat)
        new_d = q.shape(1)
        q = q.reshape(d_left, p_dim, new_d)
        tensors[i] = q

        # Absorb R into the next tensor
        # This means tensors(i + 1) has shape (d_right, p, d_right_next)
        # Sum over j for all i, k, l
        tensors[i + 1] = np.einsum('ij,jkl->ikl', R, tensors[i + 1])

    # Iterate from right to left to get lambda and gamma
    gammas = [None] * n
    lambdas = []

    for i in range(n - 1, 0, -1)
        # a should have shape (d_left, p, d_right)
        a = tensors[i]
        d_left, p_dim, d_right = a.shape

        # Reshape to matrix of form (d_left * p) x d_right
        mat = A.reshape(d_left * p_dim, d_right)

        # Perform SVD to get u . s . vh
        # u has shape (m, k), vh has shape (k, n), and s has shape (k) (singular values)
        u, s, vh = np.linalg.svd(mat, full_matrices = False)
        
        # Normalize Schmidt coefficients
        norm = np.linalg.norm(s)
        s = s / norm

        d_bond = len(s)

        # Gamma[i] = diag(1/lambda) . vh (reshaped)
        # First build gammas as vh reshaped, and then handle lambdas
        gammas[i] = vh.reshape(d_bond, p_dim, d_right)
        lambdas.insert(0, s)

        # Absorb u . diag(s) into the next tensor (left tensor)
        # Sum over k for all i, j, l
        tensors[i - 1] = np.einsum('ijk,lk->ijl', tensors[i - 1], u * s[np.newaxis, :])
    
    # Handle the first site gamma
    a = tensors[0]
    # Sum over i, j, k of the conjugate
    norm = np.sqrt(np.einsum('ijk,ijk->', a, np.conj(a)))
    a = a / norm
    gammas[0] = a

    return CanonicalMPS(gammas, lambdas)

# Given an MPS, contract all tensors to reconstruct the full state vector
# Returns a state vector of length p^N (probability distribution across all possibilities)
# Not strictly necessary, but I need a good way to check that my mps conversion is correct
def mps_to_state (mps: MPS) -> np.ndarray

    n = mps.n
    p = mps.p
    # Start with tensor 0, with shape (1, p, d1)
    result = mps.tensors[0]

    for i in range(1, n):
        # Result has shape (1, p^i, d_i). Tensors[i] has shape (d_i, p, d_(i + 1))
        # Goal is to contract over bond index d_i
        # Sum over j for k, l, etc.
        result = np.einsum('...j,jkl->...kl', result, mps.tensors[i])
        # Merge the physical indices together and reshape
        shape = list(result.shape)
        # If we have shape (1, p, p, ... , p, D_(i + 1)) we need to merge all middle dimensions into one
        new_shape = [shape[0], -1, shape[-1]]
        result = result.reshape(new_shape)

    # Final shape should be (1, p^N, 1) -> flatten to p^n
    return result.flatten()

# Given a canonical MPS, convert back to statevector
# Convert partially back to MPS and then call mps_to_state
def canonical_to_state(cmps: CanonicalMPS) -> np.ndarray:
    n = cmps.n

    # Construct MPS from gamma-lambda with einsum
    tensors = []
    for i in range(n):
        # gammas are constructed as (d_left, p, d_right)
        g = cmps.gammas[i]
        # For n - 1 gammas
        if i < n - 1:
            l = cmps.lambdas[i]
            # Reconstruct a(i) from gamma(i) . diag(l(i))
            # Iterate over all i, j, k
            a = np.einsum('ijk,k->ijk', g, l)
        # In the n - 1 gamma case (the rightmost)
        else:
            a = g.copy()
        tensors.append(a)
    return mps_to_state(MPS(tensors))

# Compute the von Neumann entanglement entropy at a given bond
# The easiest way to do it is to use canonical form and compute from the Schmidt coefficients (lambdas)
# Given a CanonicalMPS and a bond index (from 0 to N - 2), returns float entropy S
def entanglement_entropy(cmps: CanonicalMPS, bond: int) -> float:
    l = cmps.lambdas[bond]
    # Schmidt coefficients squared = eigenvalues of reduced density matrix
    probs = l ** 2
    # The equation can be written as -sum_i((lambda_i^2) * log(lambda_i^2))
    return -sum(probs * np.log2(probs))