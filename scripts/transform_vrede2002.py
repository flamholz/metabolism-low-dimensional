#!/usr/bin/env python3
"""Transform Vrede 2002 Table 3 source data into empirical ratio CSV.

Input:
- data/vrede2002_table3.csv (contains mean +- uncertainty strings)

Output:
- output/vrede2002_empirical.csv with numeric columns:
  C_to_P,N_to_P
"""

from __future__ import annotations

import csv
from pathlib import Path


ATOMIC_MASS = {
    "C": 12.011,
    "N": 14.0067,
    "P": 30.973762,
}


def parse_mean(value: str) -> float:
    """Extract the leading mean from values like '153 +- 17' or plain numerics."""
    text = value.strip()
    if not text:
        raise ValueError("Empty numeric field")
    # Keep only the mean component and normalize common unicode/minus variants.
    mean_text = text.split("+-", 1)[0].split("±", 1)[0].strip()
    mean_text = mean_text.replace("−", "-")
    return float(mean_text)


def transform(input_csv: Path, output_csv: Path) -> int:
    rows_out: list[tuple[float, float]] = []

    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip blank trailing rows.
            if not row or all((v is None or str(v).strip() == "") for v in row.values()):
                continue

            carbon = parse_mean(row["Carbon (fg cell−1)"])
            nitrogen = parse_mean(row["Nitrogen (fg cell−1)"])
            phosphorus = parse_mean(row["Phosphorus (fg cell−1)"])
            if phosphorus <= 0.0:
                raise ValueError("Phosphorus mean must be positive for ratio conversion")

            c_moles = carbon / ATOMIC_MASS["C"]
            n_moles = nitrogen / ATOMIC_MASS["N"]
            p_moles = phosphorus / ATOMIC_MASS["P"]

            c_to_p = c_moles / p_moles
            n_to_p = n_moles / p_moles
            rows_out.append((c_to_p, n_to_p))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["C_to_P", "N_to_P"])
        for c_to_p, n_to_p in rows_out:
            writer.writerow([f"{c_to_p:.10g}", f"{n_to_p:.10g}"])

    return len(rows_out)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = repo_root / "data" / "vrede2002_table3.csv"
    output_csv = repo_root / "output" / "vrede2002_empirical.csv"

    n_rows = transform(input_csv=input_csv, output_csv=output_csv)
    print(f"Wrote {n_rows} rows to {output_csv}")


if __name__ == "__main__":
    main()
