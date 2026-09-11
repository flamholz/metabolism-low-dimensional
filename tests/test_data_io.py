from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from metabolism_low_dim.data_io import load_empirical_ratios


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


if __name__ == "__main__":
    unittest.main()
