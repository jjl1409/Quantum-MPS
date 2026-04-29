from mps import (
    MPS, CanonicalMPS, mps_decomposition, mps_to_canonical, mps_to_state, canonical_to_state,
    entanglement_entropy)
from states import (
    ghz_state, w_state, random_state, product_state
)
import numpy as np

# State tests
ghz = ghz_state(5)
# print(ghz_state(5))
# print(w_state(5))
# print(random_state(5))
# print(product_state(5))

# MPS tests
print(mps_decomposition(ghz_state(5)))