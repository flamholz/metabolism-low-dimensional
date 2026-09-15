from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from metabolism_low_dim.constants import MACROMOLECULES
from metabolism_low_dim.data_io import (
    load_aa_frequencies_by_genome,
    load_element_ranges,
    load_empirical_ratios,
    load_mass_fraction_ranges,
    load_na_residue_element_counts,
    load_observed_aa_mean,
    load_residue_element_counts,
)


class TestDataIO(unittest.TestCase):
    def test_load_empirical_ratios_cp_np_schema(self) -> None:
        csv_text = "cp_mean,np_mean\n106,16\n53,8\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empirical.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            nc, pc = load_empirical_ratios(csv_path)

        expected_nc = np.array([16.0 / 106.0, 8.0 / 53.0], dtype=float)
        expected_pc = np.array([1.0 / 106.0, 1.0 / 53.0], dtype=float)
        self.assertTrue(np.allclose(nc, expected_nc))
        self.assertTrue(np.allclose(pc, expected_pc))

    def test_load_vrede_empirical_csv(self) -> None:
        csv_path = Path("output/vrede2002_empirical.csv")
        nc, pc = load_empirical_ratios(csv_path)

        # 13 non-empty records in the transformed Vrede table.
        self.assertEqual(len(nc), 13)
        self.assertEqual(len(pc), 13)
        self.assertTrue(np.all(nc > 0.0))
        self.assertTrue(np.all(pc > 0.0))

    def test_load_mass_fraction_ranges(self) -> None:
        data = {
            "macromolecules": {
                "protein": [0.30, 0.60],
                "nucleic_acid": [0.05, 0.20],
                "membrane_lipid": [0.03, 0.08],
                "storage_lipid": [0.02, 0.20],
                "metabolite": [0.01, 0.10],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "mass_fraction_ranges.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            ranges = load_mass_fraction_ranges(json_path)

        self.assertEqual(ranges["protein"], (0.30, 0.60))
        self.assertEqual(
            set(ranges.keys()),
            {"protein", "nucleic_acid", "membrane_lipid", "storage_lipid", "metabolite"},
        )

    def test_load_mass_fraction_ranges_rejects_invalid_range(self) -> None:
        data = {
            "macromolecules": {
                "protein": [0.60, 0.30],  # low >= high
                "nucleic_acid": [0.05, 0.20],
                "membrane_lipid": [0.03, 0.08],
                "storage_lipid": [0.02, 0.20],
                "metabolite": [0.01, 0.10],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "mass_fraction_ranges.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mass_fraction_ranges(json_path)

    def test_load_mass_fraction_ranges_from_repo_data(self) -> None:
        ranges = load_mass_fraction_ranges(Path("data/mass_fraction_ranges.json"))
        for macro in ("protein", "nucleic_acid", "membrane_lipid", "storage_lipid", "metabolite"):
            low, high = ranges[macro]
            self.assertLess(low, high)

    def test_load_element_ranges(self) -> None:
        data = {
            "macromolecules": {
                macro: {"C": [0.30, 0.50], "N": [0.05, 0.10], "O": [0.20, 0.40], "P": [0.00, 0.02]}
                for macro in MACROMOLECULES
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "element_ranges.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            element_ranges = load_element_ranges(json_path)

        self.assertEqual(element_ranges["protein"]["C"], (0.30, 0.50))
        self.assertEqual(set(element_ranges.keys()), set(MACROMOLECULES))

    def test_load_element_ranges_rejects_missing_element(self) -> None:
        data = {
            "macromolecules": {
                macro: {"C": [0.30, 0.50], "N": [0.05, 0.10], "O": [0.20, 0.40]}  # missing P
                for macro in MACROMOLECULES
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "element_ranges.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_element_ranges(json_path)

    def test_load_element_ranges_from_repo_data(self) -> None:
        element_ranges = load_element_ranges(Path("data/element_ranges.json"))
        for macro in MACROMOLECULES:
            for element in ("C", "N", "O", "P"):
                low, high = element_ranges[macro][element]
                self.assertLessEqual(low, high)

    def test_load_residue_element_counts_uses_polymer_form(self) -> None:
        data = {
            "residues": {
                "X": {
                    "free": {"C": 5, "H": 8, "N": 1, "O": 2, "S": 0},
                    "polymer": {"C": 5, "H": 6, "N": 1, "O": 1, "S": 0},
                    "note": "synthetic test residue",
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "aa_residue_element_counts.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            aa_codes, counts = load_residue_element_counts(json_path)

        self.assertEqual(aa_codes, ("X",))
        self.assertEqual(counts["X"], {"C": 5, "H": 6, "N": 1, "O": 1, "S": 0})

    def test_load_residue_element_counts_from_repo_data(self) -> None:
        aa_codes, counts = load_residue_element_counts(Path("data/aa_residue_element_counts.json"))

        self.assertEqual(len(aa_codes), 20)
        # Glycine residue (dehydrated): C2H3NO.
        self.assertEqual(counts["G"], {"C": 2, "H": 3, "N": 1, "O": 1, "S": 0})

    def test_load_na_residue_element_counts_uses_polymer_form(self) -> None:
        data = {
            "pools": {
                "RNA": {
                    "A": {
                        "free": {"C": 10, "H": 14, "N": 5, "O": 7, "P": 1},
                        "polymer": {"C": 10, "H": 12, "N": 5, "O": 6, "P": 1},
                        "note": "synthetic test nucleotide",
                    }
                },
                "DNA": {
                    "A": {
                        "free": {"C": 10, "H": 14, "N": 5, "O": 6, "P": 1},
                        "polymer": {"C": 10, "H": 12, "N": 5, "O": 5, "P": 1},
                        "note": "synthetic test nucleotide",
                    }
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "na_residue_element_counts.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            counts = load_na_residue_element_counts(json_path)

        self.assertEqual(counts["RNA"]["A"], {"C": 10, "H": 12, "N": 5, "O": 6, "P": 1})
        self.assertEqual(counts["DNA"]["A"], {"C": 10, "H": 12, "N": 5, "O": 5, "P": 1})

    def test_load_na_residue_element_counts_from_repo_data(self) -> None:
        counts = load_na_residue_element_counts(Path("data/na_residue_element_counts.json"))

        # RNA "A" residue (dehydrated AMP): C10H12N5O6P.
        self.assertEqual(counts["RNA"]["A"], {"C": 10, "H": 12, "N": 5, "O": 6, "P": 1})
        self.assertEqual(set(counts.keys()), {"RNA", "DNA"})

    def test_load_observed_aa_mean(self) -> None:
        data = {
            "amino_acids": {
                "A": {"mean": 0.09, "std": 0.03, "min": 0.01, "max": 0.16},
                "G": {"mean": 0.07, "std": 0.01, "min": 0.03, "max": 0.11},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "aa_frequencies.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            observed_aa_mean = load_observed_aa_mean(json_path, aa_codes=("A", "G"))

        self.assertEqual(observed_aa_mean, {"A": 0.09, "G": 0.07})

    def test_load_observed_aa_mean_rejects_missing_aa(self) -> None:
        data = {"amino_acids": {"A": {"mean": 0.09, "std": 0.03, "min": 0.01, "max": 0.16}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "aa_frequencies.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_observed_aa_mean(json_path, aa_codes=("A", "G"))

    def test_load_observed_aa_mean_from_repo_data_sums_to_one(self) -> None:
        aa_codes, _ = load_residue_element_counts(Path("data/aa_residue_element_counts.json"))
        observed_aa_mean = load_observed_aa_mean(
            Path("output/moura2013_aa_frequencies.json"), aa_codes
        )

        self.assertEqual(set(observed_aa_mean.keys()), set(aa_codes))
        self.assertAlmostEqual(sum(observed_aa_mean.values()), 1.0, places=9)

    def test_load_aa_frequencies_by_genome(self) -> None:
        csv_text = (
            "organism,domain,A,G\n"
            "org1,BACTERIA,0.6,0.4\n"
            "org2,ARCHAEA,0.3,0.7\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "aa_frequencies_by_genome.csv"
            csv_path.write_text(csv_text, encoding="utf-8")
            aa_codes, frequencies = load_aa_frequencies_by_genome(csv_path)

        self.assertEqual(aa_codes, ("A", "G"))
        self.assertEqual(frequencies.shape, (2, 2))
        self.assertTrue(np.allclose(frequencies, [[0.6, 0.4], [0.3, 0.7]]))

    def test_load_aa_frequencies_by_genome_from_repo_data(self) -> None:
        aa_codes, frequencies = load_aa_frequencies_by_genome(
            Path("output/moura2013_aa_frequencies_by_genome.csv")
        )

        self.assertEqual(len(aa_codes), 20)
        self.assertEqual(frequencies.shape, (1086, 20))
        row_sums = frequencies.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
