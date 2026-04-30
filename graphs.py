from mps import (
    MPS, CanonicalMPS, mps_decomposition, mps_to_canonical, mps_to_state, canonical_to_state,
    entanglement_entropy)
from states import (
    ghz_state, w_state, random_state, product_state
)
import numpy as np
import matplotlib.pyplot as plt

# Define dictionary of states
n = 10
states = {
    "GHZ State": ghz_state(n),
    "W State": w_state(n),
    "Random State": random_state(n),
    "Product State": product_state(n)
}

# Plot 1: All states and their entropy
plt.figure()
for label, psi in states.items():
    mps = mps_decomposition(psi)
    cmps = mps_to_canonical(mps)

    bonds = np.arange(n - 1)
    entropies = [entanglement_entropy(cmps, b) for b in bonds]
    bond_labels = [f"{i}|{i+1}" for i in bonds]
    plt.plot(bonds, entropies, 'o-', label = label, linewidth = 2, markersize = 6)

plt.xticks(bonds, bond_labels)
plt.xlabel("Bond position")
plt.ylabel("Entanglement entropy S (bits)")
plt.title("Entanglement entropy vs. bond position")
plt.legend()
plt.tight_layout()
plt.show()
