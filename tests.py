from mps import (
    MPS, CanonicalMPS, mps_decomposition, mps_to_canonical, mps_to_state, canonical_to_state,
    entanglement_entropy)
from states import (
    ghz_state, w_state, random_state, product_state
)
import numpy as np

np.set_printoptions(precision=6, suppress=True)
# State tests
ghz = ghz_state(10)
w = w_state(5)
random = random_state(5)
product = product_state(5)
# print(ghz_state(5))
# print(w_state(5))
# print(random_state(5))
# print(product_state(5))

# MPS tests
# print(ghz)
# print(mps_to_state(mps_decomposition(ghz)))
# print(w)
# print(mps_to_state(mps_decomposition(w)))
# print(random)
# print(mps_to_state(mps_decomposition(random)))

# Canonical tests
cmps_ghz = mps_to_canonical(mps_decomposition(ghz))
cmps_w = mps_to_canonical(mps_decomposition(w))
cmps_product = mps_to_canonical(mps_decomposition(product))
# print(ghz)
# print(canonical_to_state(cmps_ghz))
# print(w)
# print(canonical_to_state(cmps_w))

# Entanglement entropy tests
# for i in range(cmps_w.n() - 1):
#     print(entanglement_entropy(cmps_w, i))
for i in range(cmps_ghz.n() - 1):
    print(entanglement_entropy(cmps_ghz, i))
# for i in range(cmps_product.n() - 1):
#     print(entanglement_entropy(cmps_product, i))