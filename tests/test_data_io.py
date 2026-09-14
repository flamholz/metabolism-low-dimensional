from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from metabolism_low_dim.constants import MACROMOLECULES
from metabolism_low_dim.data_io import (
    load_element_ranges,
    load_empirical_ratios,
    load_mass_fraction_ranges,
    load_na_residue_element_counts,
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
        csv_path = Path("data/vrede2002_empirical.csv")
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


if __name__ == "__main__":
    unittest.main()
