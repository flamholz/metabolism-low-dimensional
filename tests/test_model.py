from __future__ import annotations

import unittest

import numpy as np

from metabolism_low_dim.constants import ATOMIC_MASS
from metabolism_low_dim.constants import MACROMOLECULES
from metabolism_low_dim.model import (
    _sample_ranged_elements_with_closure,
    compute_elemental_totals,
    sample_mass_fractions_from_ranges,
    sample_nucleic_acid_from_gc,
    sample_protein_cno_from_aa,
    sample_protein_cno_from_empirical_aa,
    to_molar_ratio,
)


class TestModel(unittest.TestCase):
    def test_mass_fraction_sampling_closure(self) -> None:
        rng = np.random.default_rng(7)
        ranges = {
            "protein": (0.3, 0.65),
            "nucleic_acid": (0.03, 0.3),
            "membrane_lipid": (0.03, 0.08),
            "storage_lipid": (0.02, 0.22),
            "metabolite": (0.01, 0.12),
        }
        mass_fractions = sample_mass_fractions_from_ranges(rng, 5000, ranges)

        self.assertEqual(mass_fractions.shape, (5000, 6))
        # protein, sugar, nucleic_acid, membrane_lipid, storage_lipid, metabolite
        sugar = mass_fractions[:, 1]
        self.assertTrue(np.all(sugar >= 0.0))
        self.assertTrue(np.allclose(np.sum(mass_fractions, axis=1), 1.0, atol=1e-12))

    def test_to_molar_ratio(self) -> None:
        n_mass = np.array([0.14], dtype=float)
        c_mass = np.array([0.47], dtype=float)
        ratio = to_molar_ratio(n_mass, c_mass, "N", "C")

        expected = (0.14 / 14.0067) / (0.47 / 12.011)
        self.assertAlmostEqual(float(ratio[0]), expected, places=12)

    def test_to_molar_ratio_other_elements(self) -> None:
        c_mass = np.array([0.50, 0.45], dtype=float)
        o_mass = np.array([0.30, 0.35], dtype=float)
        p_mass = np.array([0.02, 0.015], dtype=float)

        oc_ratio = to_molar_ratio(o_mass, c_mass, "O", "C")
        pc_ratio = to_molar_ratio(p_mass, c_mass, "P", "C")

        expected_oc = (o_mass / 15.9994) / (c_mass / 12.011)
        expected_pc = (p_mass / 30.973762) / (c_mass / 12.011)

        self.assertTrue(np.allclose(oc_ratio, expected_oc, atol=1e-12))
        self.assertTrue(np.allclose(pc_ratio, expected_pc, atol=1e-12))

    def test_compute_elemental_totals_all_elements(self) -> None:
        # Two synthetic cells with fixed pool masses.
        mass_fractions = np.array(
            [
                [0.30, 0.15, 0.10, 0.20, 0.20, 0.05],
                [0.10, 0.25, 0.20, 0.10, 0.30, 0.05],
            ],
            dtype=float,
        )

        element_fractions = {
            "protein": {
                "C": np.array([0.50, 0.52]),
                "N": np.array([0.18, 0.17]),
                "O": np.array([0.22, 0.21]),
                "P": np.array([0.01, 0.01]),
            },
            "sugar": {
                "C": np.array([0.42, 0.43]),
                "N": np.array([0.01, 0.01]),
                "O": np.array([0.52, 0.51]),
                "P": np.array([0.00, 0.00]),
            },
            "nucleic_acid": {
                "C": np.array([0.34, 0.35]),
                "N": np.array([0.16, 0.16]),
                "O": np.array([0.34, 0.33]),
                "P": np.array([0.09, 0.09]),
            },
            "membrane_lipid": {
                "C": np.array([0.70, 0.68]),
                "N": np.array([0.01, 0.01]),
                "O": np.array([0.20, 0.22]),
                "P": np.array([0.03, 0.03]),
            },
            "storage_lipid": {
                "C": np.array([0.78, 0.79]),
                "N": np.array([0.00, 0.00]),
                "O": np.array([0.10, 0.09]),
                "P": np.array([0.00, 0.00]),
            },
            "metabolite": {
                "C": np.array([0.30, 0.32]),
                "N": np.array([0.06, 0.06]),
                "O": np.array([0.36, 0.35]),
                "P": np.array([0.16, 0.15]),
            },
        }

        totals = compute_elemental_totals(mass_fractions, element_fractions)

        for element in ("C", "N", "O", "P"):
            expected = np.zeros(mass_fractions.shape[0])
            for i, macro in enumerate(MACROMOLECULES):
                expected += mass_fractions[:, i] * element_fractions[macro][element]
            self.assertTrue(np.allclose(totals[element], expected, atol=1e-12))

    def test_sample_protein_cno_from_aa_single_residue(self) -> None:
        rng = np.random.default_rng(7)
        aa_codes = ("X",)
        residue_element_counts = {
            "X": {"C": 5, "H": 8, "N": 1, "O": 2, "S": 0},
        }
        observed_aa_mean = {"X": 1.0}

        c_frac, n_frac, o_frac = sample_protein_cno_from_aa(
            rng=rng,
            n_samples=10,
            concentration=250.0,
            aa_codes=aa_codes,
            residue_element_counts=residue_element_counts,
            observed_aa_mean=observed_aa_mean,
        )

        residue_mass = (
            5 * ATOMIC_MASS["C"]
            + 8 * ATOMIC_MASS["H"]
            + 1 * ATOMIC_MASS["N"]
            + 2 * ATOMIC_MASS["O"]
        )
        expected_c = (5 * ATOMIC_MASS["C"]) / residue_mass
        expected_n = (1 * ATOMIC_MASS["N"]) / residue_mass
        expected_o = (2 * ATOMIC_MASS["O"]) / residue_mass

        self.assertTrue(np.allclose(c_frac, expected_c, atol=1e-12))
        self.assertTrue(np.allclose(n_frac, expected_n, atol=1e-12))
        self.assertTrue(np.allclose(o_frac, expected_o, atol=1e-12))

    def test_sample_protein_cno_from_empirical_aa_bootstraps_real_rows(self) -> None:
        rng = np.random.default_rng(7)
        aa_codes = ("X", "Y")
        residue_element_counts = {
            "X": {"C": 5, "H": 8, "N": 1, "O": 2, "S": 0},
            "Y": {"C": 3, "H": 4, "N": 2, "O": 1, "S": 0},
        }

        # Two synthetic "genomes" with distinct, extreme compositions.
        empirical_aa_codes = ("X", "Y")
        empirical_frequencies = np.array([[1.0, 0.0], [0.0, 1.0]])

        def residue_mass(counts: dict[str, int]) -> float:
            return (
                counts["C"] * ATOMIC_MASS["C"]
                + counts["H"] * ATOMIC_MASS["H"]
                + counts["N"] * ATOMIC_MASS["N"]
                + counts["O"] * ATOMIC_MASS["O"]
            )

        mass_x = residue_mass(residue_element_counts["X"])
        mass_y = residue_mass(residue_element_counts["Y"])
        c_x = 5 * ATOMIC_MASS["C"] / mass_x
        c_y = 3 * ATOMIC_MASS["C"] / mass_y

        c_frac, n_frac, o_frac = sample_protein_cno_from_empirical_aa(
            rng=rng,
            n_samples=200,
            aa_codes=aa_codes,
            residue_element_counts=residue_element_counts,
            empirical_aa_codes=empirical_aa_codes,
            empirical_frequencies=empirical_frequencies,
        )

        # Every sample must exactly match one of the two input genomes' C
        # fraction -- proof this is a bootstrap over real rows, not
        # synthetic interpolation between them.
        is_genome_x = np.isclose(c_frac, c_x, atol=1e-12)
        is_genome_y = np.isclose(c_frac, c_y, atol=1e-12)
        self.assertTrue(np.all(is_genome_x | is_genome_y))
        # With 200 draws from 2 genomes, both should appear.
        self.assertTrue(np.any(is_genome_x))
        self.assertTrue(np.any(is_genome_y))

    def test_sample_protein_cno_from_empirical_aa_reorders_columns(self) -> None:
        rng = np.random.default_rng(7)
        aa_codes = ("X", "Y")
        residue_element_counts = {
            "X": {"C": 5, "H": 8, "N": 1, "O": 2, "S": 0},
            "Y": {"C": 3, "H": 4, "N": 2, "O": 1, "S": 0},
        }
        # Columns in the opposite order from aa_codes.
        empirical_aa_codes = ("Y", "X")
        empirical_frequencies = np.array([[0.0, 1.0]])  # Y=0.0, X=1.0 -> all-X genome

        c_frac, _, _ = sample_protein_cno_from_empirical_aa(
            rng=rng,
            n_samples=5,
            aa_codes=aa_codes,
            residue_element_counts=residue_element_counts,
            empirical_aa_codes=empirical_aa_codes,
            empirical_frequencies=empirical_frequencies,
        )

        mass_x = sum(residue_element_counts["X"][e] * ATOMIC_MASS[e] for e in ("C", "H", "N", "O"))
        expected_c = 5 * ATOMIC_MASS["C"] / mass_x
        self.assertTrue(np.allclose(c_frac, expected_c, atol=1e-12))

    def test_sample_ranged_elements_with_closure_enforces_mass_balance(self) -> None:
        rng = np.random.default_rng(7)
        # metabolite's own ranges allow C+N+O+P up to 1.35 if drawn
        # independently at their extremes, so this exercises real rejection.
        ranges = {
            "C": (0.20, 0.45),
            "N": (0.02, 0.10),
            "O": (0.22, 0.50),
            "P": (0.08, 0.30),
        }
        sampled = _sample_ranged_elements_with_closure(rng, 5000, ranges)

        for element in ranges:
            self.assertEqual(sampled[element].shape, (5000,))
            low, high = ranges[element]
            self.assertTrue(np.all(sampled[element] >= low))
            self.assertTrue(np.all(sampled[element] <= high))

        total = sum(sampled[element] for element in ranges)
        self.assertTrue(np.all(total <= 1.0 + 1e-12))

    def test_sample_nucleic_acid_from_gc_fixed_composition(self) -> None:
        rng = np.random.default_rng(7)

        # Use identical residue chemistry for all nucleotides in both pools.
        # This makes element mass fractions analytically fixed regardless of
        # sampled GC content or RNA/DNA mixing.
        residue = {"C": 10, "H": 12, "N": 2, "O": 7, "P": 1}
        na_residue_element_counts = {
            "RNA": {nt: dict(residue) for nt in ("A", "U", "G", "C")},
            "DNA": {nt: dict(residue) for nt in ("A", "T", "G", "C")},
        }
        na_pool_mix_mean = {"RNA": 0.9, "DNA": 0.1}

        sampled = sample_nucleic_acid_from_gc(
            rng=rng,
            n_samples=50,
            gc_mean=0.5,
            gc_concentration=50.0,
            pool_concentration=300.0,
            na_residue_element_counts=na_residue_element_counts,
            na_pool_mix_mean=na_pool_mix_mean,
        )

        residue_mass = (
            residue["C"] * ATOMIC_MASS["C"]
            + residue["H"] * ATOMIC_MASS["H"]
            + residue["N"] * ATOMIC_MASS["N"]
            + residue["O"] * ATOMIC_MASS["O"]
            + residue["P"] * ATOMIC_MASS["P"]
        )

        expected = {
            "C": residue["C"] * ATOMIC_MASS["C"] / residue_mass,
            "N": residue["N"] * ATOMIC_MASS["N"] / residue_mass,
            "O": residue["O"] * ATOMIC_MASS["O"] / residue_mass,
            "P": residue["P"] * ATOMIC_MASS["P"] / residue_mass,
        }

        for element in ("C", "N", "O", "P"):
            self.assertEqual(sampled[element].shape, (50,))
            self.assertTrue(np.allclose(sampled[element], expected[element], atol=1e-12))

    def test_sample_nucleic_acid_from_gc_distinguishes_rna_dna(self) -> None:
        # Build pools where RNA and DNA differ only in oxygen content.
        # All nucleotides within a pool are identical so GC sampling does not
        # affect per-pool elemental mass fractions.
        rna_residue = {"C": 10, "H": 12, "N": 2, "O": 8, "P": 1}
        dna_residue = {"C": 10, "H": 12, "N": 2, "O": 7, "P": 1}
        na_residue_element_counts = {
            "RNA": {nt: dict(rna_residue) for nt in ("A", "U", "G", "C")},
            "DNA": {nt: dict(dna_residue) for nt in ("A", "T", "G", "C")},
        }

        def residue_mass(residue: dict[str, int]) -> float:
            return (
                residue["C"] * ATOMIC_MASS["C"]
                + residue["H"] * ATOMIC_MASS["H"]
                + residue["N"] * ATOMIC_MASS["N"]
                + residue["O"] * ATOMIC_MASS["O"]
                + residue["P"] * ATOMIC_MASS["P"]
            )

        rna_mass = residue_mass(rna_residue)
        dna_mass = residue_mass(dna_residue)
        rna_o_frac = rna_residue["O"] * ATOMIC_MASS["O"] / rna_mass
        dna_o_frac = dna_residue["O"] * ATOMIC_MASS["O"] / dna_mass

        n_samples = 4000
        rna_heavy_mix = {"RNA": 0.95, "DNA": 0.05}
        dna_heavy_mix = {"RNA": 0.05, "DNA": 0.95}

        rna_heavy = sample_nucleic_acid_from_gc(
            rng=np.random.default_rng(7),
            n_samples=n_samples,
            gc_mean=0.5,
            gc_concentration=50.0,
            pool_concentration=1000.0,
            na_residue_element_counts=na_residue_element_counts,
            na_pool_mix_mean=rna_heavy_mix,
        )
        dna_heavy = sample_nucleic_acid_from_gc(
            rng=np.random.default_rng(8),
            n_samples=n_samples,
            gc_mean=0.5,
            gc_concentration=50.0,
            pool_concentration=1000.0,
            na_residue_element_counts=na_residue_element_counts,
            na_pool_mix_mean=dna_heavy_mix,
        )

        # Oxygen fraction should track the RNA/DNA mixing choice.
        self.assertGreater(np.mean(rna_heavy["O"]), np.mean(dna_heavy["O"]))

        expected_rna_heavy_o = 0.95 * rna_o_frac + 0.05 * dna_o_frac
        expected_dna_heavy_o = 0.05 * rna_o_frac + 0.95 * dna_o_frac
        self.assertAlmostEqual(float(np.mean(rna_heavy["O"])), expected_rna_heavy_o, places=3)
        self.assertAlmostEqual(float(np.mean(dna_heavy["O"])), expected_dna_heavy_o, places=3)


if __name__ == "__main__":
    unittest.main()
