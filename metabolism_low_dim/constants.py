"""Shared constants for phase-diagram sampling."""

from __future__ import annotations

ATOMIC_MASS = {
    "C": 12.011,
    "H": 1.00794,
    "N": 14.0067,
    "O": 15.9994,
    "P": 30.973762,
    "S": 32.065,
}

MACROMOLECULES = (
    "protein",
    "sugar",
    "nucleic_acid",
    "membrane_lipid",
    "storage_lipid",
    "metabolite",
)
INDEPENDENT_MACROMOLECULES = (
    "protein",
    "nucleic_acid",
    "membrane_lipid",
    "storage_lipid",
    "metabolite",
)

ELEMENTS_FOR_RANGES = ("C", "N", "O", "P")
RESIDUE_ELEMENTS = ("C", "H", "N", "O", "S")
NA_ELEMENTS = ("C", "H", "N", "O", "P")
NA_POOLS = ("RNA", "DNA")
