#!/usr/bin/env python3
"""Transform Makino 2003 E. coli extracted ratios into empirical overlay CSV.

Input format (long):
- data/makino2003_coli.csv with columns:
  value_type,RNA_content_pct,value
  where value_type is one of {C_to_P_ratio, N_to_P_ratio}

Output format (wide, overlay-ready):
- output/makino2003_empirical.csv with columns:
  C_to_P,N_to_P,RNA_content_pct

The phase-diagram loader only requires C_to_P and N_to_P.
RNA_content_pct is retained for traceability.
"""

from __future__ import annotations

import csv
from pathlib import Path


def load_series(input_csv: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    c_to_p: list[tuple[float, float]] = []
    n_to_p: list[tuple[float, float]] = []

    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"value_type", "RNA_content_pct", "value"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{input_csv} must include columns {sorted(required)}; got {reader.fieldnames}"
            )

        for row in reader:
            value_type = row["value_type"].strip()
            rna_pct = float(row["RNA_content_pct"])
            value = float(row["value"])

            if value_type == "C_to_P_ratio":
                c_to_p.append((rna_pct, value))
            elif value_type == "N_to_P_ratio":
                n_to_p.append((rna_pct, value))
            else:
                raise ValueError(f"Unsupported value_type '{value_type}' in {input_csv}")

    if not c_to_p:
        raise ValueError(f"No C_to_P_ratio rows found in {input_csv}")
    if not n_to_p:
        raise ValueError(f"No N_to_P_ratio rows found in {input_csv}")
    return c_to_p, n_to_p


def pair_by_sorted_rna(
    c_to_p: list[tuple[float, float]],
    n_to_p: list[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    """Pair C:P and N:P observations by sorted RNA% rank.

    Extracted values come from separate traces and RNA values are slightly noisy,
    so exact joins are brittle. Rank-based pairing is robust for this monotonic
    dataset and preserves one-to-one correspondence.
    """
    c_sorted = sorted(c_to_p, key=lambda x: x[0])
    n_sorted = sorted(n_to_p, key=lambda x: x[0])
    if len(c_sorted) != len(n_sorted):
        raise ValueError(
            "Unequal C_to_P_ratio and N_to_P_ratio counts; cannot pair one-to-one "
            f"({len(c_sorted)} vs {len(n_sorted)})"
        )

    paired: list[tuple[float, float, float]] = []
    for (rna_c, c_val), (rna_n, n_val) in zip(c_sorted, n_sorted):
        rna_pct = 0.5 * (rna_c + rna_n)
        paired.append((c_val, n_val, rna_pct))

    return paired


def transform(input_csv: Path, output_csv: Path) -> int:
    c_to_p, n_to_p = load_series(input_csv)
    paired = pair_by_sorted_rna(c_to_p, n_to_p)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["C_to_P", "N_to_P", "RNA_content_pct"])
        for c_val, n_val, rna_pct in paired:
            writer.writerow([f"{c_val:.10g}", f"{n_val:.10g}", f"{rna_pct:.10g}"])
    return len(paired)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = repo_root / "data" / "makino2003_coli.csv"
    output_csv = repo_root / "output" / "makino2003_empirical.csv"

    n_rows = transform(input_csv=input_csv, output_csv=output_csv)
    print(f"Wrote {n_rows} rows to {output_csv}")


if __name__ == "__main__":
    main()