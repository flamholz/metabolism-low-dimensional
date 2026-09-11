"""Core simulation and stoichiometry math."""

from __future__ import annotations

import numpy as np

from .constants import ATOMIC_MASS, MACROMOLECULES, NA_ELEMENTS, NA_POOLS, RESIDUE_ELEMENTS


def sample_mass_fractions_from_ranges(
    rng: np.random.Generator,
    n_samples: int,
    mass_fraction_ranges: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Sample non-sugar pools independently; sugar = 1 - sum.

    Draws are rejected when sugar < 0 (infeasible), so we oversample in batches
    until n_samples valid rows are collected.
    """
    batch = max(n_samples * 4, 10_000)
    collected: list[np.ndarray] = []
    n_collected = 0
    while n_collected < n_samples:
        prot = rng.uniform(*mass_fraction_ranges["protein"], size=batch)
        na = rng.uniform(*mass_fraction_ranges["nucleic_acid"], size=batch)
        membrane_lipid = rng.uniform(*mass_fraction_ranges["membrane_lipid"], size=batch)
        storage_lipid = rng.uniform(*mass_fraction_ranges["storage_lipid"], size=batch)
        metabolite = rng.uniform(*mass_fraction_ranges["metabolite"], size=batch)
        sugar = 1.0 - prot - na - membrane_lipid - storage_lipid - metabolite
        valid = sugar >= 0.0
        rows = np.stack(
            [
                prot[valid],
                sugar[valid],
                na[valid],
                membrane_lipid[valid],
                storage_lipid[valid],
                metabolite[valid],
            ],
            axis=1,
        )
        collected.append(rows)
        n_collected += rows.shape[0]
    return np.concatenate(collected, axis=0)[:n_samples]


def sample_protein_cn_from_aa(
    rng: np.random.Generator,
    n_samples: int,
    concentration: float,
    aa_codes: tuple[str, ...],
    residue_element_counts: dict[str, dict[str, int]],
    observed_aa_mean: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array([observed_aa_mean[aa] for aa in aa_codes], dtype=float)
    mean /= mean.sum()
    alpha = mean * concentration
    aa_frequencies = rng.dirichlet(alpha, size=n_samples)

    c_counts = np.array([residue_element_counts[aa]["C"] for aa in aa_codes], dtype=float)
    n_counts = np.array([residue_element_counts[aa]["N"] for aa in aa_codes], dtype=float)
    residue_masses = np.array(
        [
            sum(residue_element_counts[aa][elem] * ATOMIC_MASS[elem] for elem in RESIDUE_ELEMENTS)
            for aa in aa_codes
        ],
        dtype=float,
    )

    mean_residue_mass = aa_frequencies @ residue_masses
    c_mass_fraction = (aa_frequencies @ c_counts) * ATOMIC_MASS["C"] / mean_residue_mass
    n_mass_fraction = (aa_frequencies @ n_counts) * ATOMIC_MASS["N"] / mean_residue_mass
    return c_mass_fraction, n_mass_fraction


def sample_nucleic_acid_from_gc(
    rng: np.random.Generator,
    n_samples: int,
    gc_mean: float,
    gc_concentration: float,
    pool_concentration: float,
    na_residue_element_counts: dict[str, dict[str, dict[str, int]]],
    na_pool_mix_mean: dict[str, float],
) -> dict[str, np.ndarray]:
    gc_alpha = gc_mean * gc_concentration
    gc_beta = (1.0 - gc_mean) * gc_concentration
    gc_content = rng.beta(gc_alpha, gc_beta, size=n_samples)

    gc_nts = {"RNA": ("G", "C"), "DNA": ("G", "C")}

    per_pool_element_mass_fraction: dict[str, dict[str, np.ndarray]] = {}
    for pool in NA_POOLS:
        nt_codes = tuple(na_residue_element_counts[pool].keys())
        nt_freq_list = []
        for nt in nt_codes:
            if nt in gc_nts[pool]:
                nt_freq_list.append(gc_content / 2.0)
            else:
                nt_freq_list.append((1.0 - gc_content) / 2.0)
        nt_frequencies = np.stack(nt_freq_list, axis=1)

        residue_masses = np.array(
            [
                sum(na_residue_element_counts[pool][nt][elem] * ATOMIC_MASS[elem] for elem in NA_ELEMENTS)
                for nt in nt_codes
            ],
            dtype=float,
        )
        mean_residue_mass = nt_frequencies @ residue_masses

        pool_fracs: dict[str, np.ndarray] = {}
        for element in ("C", "N", "O", "P"):
            element_counts = np.array(
                [na_residue_element_counts[pool][nt][element] for nt in nt_codes],
                dtype=float,
            )
            pool_fracs[element] = (
                (nt_frequencies @ element_counts) * ATOMIC_MASS[element] / mean_residue_mass
            )
        per_pool_element_mass_fraction[pool] = pool_fracs

    pool_mean = np.array([na_pool_mix_mean[pool] for pool in NA_POOLS], dtype=float)
    pool_mean /= pool_mean.sum()
    pool_alpha = pool_mean * pool_concentration
    pool_fractions = rng.dirichlet(pool_alpha, size=n_samples)

    sampled = {element: np.zeros(n_samples) for element in ("C", "N", "O", "P")}
    for i, pool in enumerate(NA_POOLS):
        for element in sampled:
            sampled[element] += pool_fractions[:, i] * per_pool_element_mass_fraction[pool][element]
    return sampled


def sample_element_fractions(
    rng: np.random.Generator,
    n_samples: int,
    protein_aa_mode: str,
    protein_aa_concentration: float,
    nucleic_acid_nt_mode: str,
    nucleic_acid_gc_concentration: float,
    nucleic_acid_pool_concentration: float,
    element_ranges: dict[str, dict[str, tuple[float, float]]],
    aa_codes: tuple[str, ...],
    residue_element_counts: dict[str, dict[str, int]],
    observed_aa_mean: dict[str, float],
    na_residue_element_counts: dict[str, dict[str, dict[str, int]]],
    na_gc_mean: float,
    na_pool_mix_mean: dict[str, float],
) -> dict[str, dict[str, np.ndarray]]:
    sampled: dict[str, dict[str, np.ndarray]] = {}

    protein_c: np.ndarray | None = None
    protein_n: np.ndarray | None = None
    if protein_aa_mode == "observed":
        protein_c, protein_n = sample_protein_cn_from_aa(
            rng=rng,
            n_samples=n_samples,
            concentration=protein_aa_concentration,
            aa_codes=aa_codes,
            residue_element_counts=residue_element_counts,
            observed_aa_mean=observed_aa_mean,
        )

    na_sampled: dict[str, np.ndarray] | None = None
    if nucleic_acid_nt_mode == "observed":
        na_sampled = sample_nucleic_acid_from_gc(
            rng=rng,
            n_samples=n_samples,
            gc_mean=na_gc_mean,
            gc_concentration=nucleic_acid_gc_concentration,
            pool_concentration=nucleic_acid_pool_concentration,
            na_residue_element_counts=na_residue_element_counts,
            na_pool_mix_mean=na_pool_mix_mean,
        )

    for macro in MACROMOLECULES:
        sampled[macro] = {}
        for element in ("C", "N", "O", "P"):
            low, high = element_ranges[macro][element]
            if macro == "protein" and protein_aa_mode == "observed" and element == "C":
                sampled[macro][element] = protein_c  # type: ignore[assignment]
            elif macro == "protein" and protein_aa_mode == "observed" and element == "N":
                sampled[macro][element] = protein_n  # type: ignore[assignment]
            elif macro == "nucleic_acid" and nucleic_acid_nt_mode == "observed":
                sampled[macro][element] = na_sampled[element]  # type: ignore[index]
            else:
                sampled[macro][element] = rng.uniform(low, high, size=n_samples)
    return sampled


def compute_elemental_totals(
    mass_fractions: np.ndarray,
    element_fractions: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    totals = {
        "C": np.zeros(mass_fractions.shape[0]),
        "N": np.zeros(mass_fractions.shape[0]),
        "O": np.zeros(mass_fractions.shape[0]),
        "P": np.zeros(mass_fractions.shape[0]),
    }
    for i, macro in enumerate(MACROMOLECULES):
        for element in totals:
            totals[element] += mass_fractions[:, i] * element_fractions[macro][element]
    return totals


def to_molar_ratio(
    numerator_mass_fraction: np.ndarray,
    denominator_mass_fraction: np.ndarray,
    numerator_element: str,
    denominator_element: str,
) -> np.ndarray:
    numerator_moles = numerator_mass_fraction / ATOMIC_MASS[numerator_element]
    denominator_moles = denominator_mass_fraction / ATOMIC_MASS[denominator_element]
    return numerator_moles / denominator_moles
