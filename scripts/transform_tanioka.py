#!/usr/bin/env python3
"""Transform Tanioka & Matsumoto (2020) algal C:N:P meta-analysis into empirical ratio CSV.

Source
------
Tanioka T, Matsumoto K (2020). "A meta-analysis on environmental drivers of
marine phytoplankton C:N:P." Biogeosciences 17, 2939-2954.
https://doi.org/10.5194/bg-17-2939-2020

Table
-----
data/tanioka2020_algalCNP_SItable.xlsx has one row per measurement, in five
tabs (P, N, Fe, I, T) covering nutrient/light/temperature-limitation
experiments. Each row is either a 'CP' (C:P molar ratio) or 'CN' (C:N molar
ratio) measurement, with Yc/Yt giving the mean ratio at a control/treatment
condition (e.g. nutrient-limited vs -replete). A CP row and a CN row from
the same underlying experiment must be paired to get one plottable (C:N,
C:P) point; N:P is then derived from the two paired ratios via
N:P = (C:P) / (C:N), since both express moles of C per mole of the other
element.

Pairing strategy
-----------------
Group rows within each tab by every column except the identifying/measured
ones (Es_id, Variable, Yc/Sc/Nc, Yt/St/Nt, Graphclick, Notes); this
resolves the vast majority of experiments to a clean one-CP/one-CN pair
per group. A handful of groups (in the I and T tabs) still contain two or
three such pairs because the distinguishing condition (e.g. "N:P = 45" vs
"N:P = 90") is only recorded in the free-text Notes column, not a
structured one; within those groups, rows still strictly alternate
CP/CN in Es_id order, so sorting by Es_id and pairing consecutive rows
recovers the correct pairs. Rows with no counterpart at all (e.g. a study
that reports C:N but never measured C:P) are dropped.

Output
------
Two files, since the full raw data is too dense to plot legibly (468
points from many correlated repeated measures per study):

- output/tanioka2020_empirical.csv: every paired (C:P, N:P) point, with
  columns C_to_P,N_to_P,driver,condition,Study,Species,PFT. 'driver' is
  the source tab (P/N/Fe/I/T); 'condition' is 'control' (the
  limited/starting state, Yc) or 'treatment' (the replete/manipulated
  state, Yt); the rest are retained for traceability.
- output/tanioka2020_empirical_by_pft.csv: one point per plankton
  functional type (PFT), averaged over that PFT's 'control' rows only
  (across all five drivers). 'control' is used rather than both states to
  avoid double-counting -- a study's control and treatment measurements
  are correlated repeated measures of the same experiment, not
  independent points. Averaging is geometric (mean of logs), since C:P
  and N:P are multiplicative ratios spanning a couple of orders of
  magnitude here; this matches how Martiny et al. (2013) -- also plotted
  as an overlay in this repo -- computed lognormal means for the same
  kind of ratio data. Columns: PFT,C_to_P,N_to_P,n.

Both are accepted as-is by metabolism_low_dim.data_io.load_empirical_ratios
(only C_to_P and N_to_P are required).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import openpyxl

SHEETS = ("P", "N", "Fe", "I", "T")
EXCLUDE_COLUMNS = {"Es_id", "Variable", "Yc", "Sc", "Nc", "Yt", "St", "Nt", "Graphclick", "Notes"}


def _read_sheet_rows(xlsx_path: Path, sheet: str) -> tuple[list[str], list[dict]]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    records = [
        dict(zip(header, row)) for row in rows[1:] if row[header.index("Es_id")] is not None
    ]
    return header, records


def _pair_cp_cn(header: list[str], records: list[dict]) -> list[tuple[dict, dict]]:
    """Group records into (CP record, CN record) pairs for the same experiment."""
    key_columns = [c for c in header if c not in EXCLUDE_COLUMNS]

    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = tuple(record[c] for c in key_columns)
        groups.setdefault(key, []).append(record)

    pairs: list[tuple[dict, dict]] = []
    for group in groups.values():
        # Some groups contain more than one CP/CN pair because the only
        # thing distinguishing their sub-experiments is free-text Notes,
        # not a structured column (e.g. two light-response experiments at
        # different N:P ratios). Zipping the CP list against the CN list,
        # both in Es_id order, recovers the correct pairing in every case
        # checked against this table. Any surplus on either side (e.g. a
        # study that reported C:N for two sub-experiments but C:P for
        # neither) has no counterpart and is dropped.
        cps = sorted((r for r in group if r["Variable"] == "CP"), key=lambda r: r["Es_id"])
        cns = sorted((r for r in group if r["Variable"] == "CN"), key=lambda r: r["Es_id"])
        pairs.extend(zip(cps, cns))
    return pairs


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def transform(xlsx_path: Path, output_csv: Path, output_by_pft_csv: Path) -> tuple[int, int]:
    rows_out: list[dict] = []

    for sheet in SHEETS:
        header, records = _read_sheet_rows(xlsx_path, sheet)
        for cp, cn in _pair_cp_cn(header, records):
            for condition, cp_value, cn_value in (
                ("control", cp["Yc"], cn["Yc"]),
                ("treatment", cp["Yt"], cn["Yt"]),
            ):
                if cp_value is None or cn_value is None or cn_value == 0:
                    continue
                rows_out.append(
                    {
                        "C_to_P": cp_value,
                        "N_to_P": cp_value / cn_value,
                        "driver": sheet,
                        "condition": condition,
                        "Study": cp["Study"],
                        "Species": cp["Species"],
                        "PFT": cp["PFT"],
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["C_to_P", "N_to_P", "driver", "condition", "Study", "Species", "PFT"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            row = dict(row)
            row["C_to_P"] = f"{row['C_to_P']:.10g}"
            row["N_to_P"] = f"{row['N_to_P']:.10g}"
            writer.writerow(row)

    by_pft: dict[str, list[dict]] = {}
    for row in rows_out:
        if row["condition"] == "control":
            by_pft.setdefault(row["PFT"], []).append(row)

    output_by_pft_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_by_pft_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["PFT", "C_to_P", "N_to_P", "n"])
        writer.writeheader()
        for pft in sorted(by_pft):
            group = by_pft[pft]
            writer.writerow(
                {
                    "PFT": pft,
                    "C_to_P": f"{_geometric_mean([r['C_to_P'] for r in group]):.10g}",
                    "N_to_P": f"{_geometric_mean([r['N_to_P'] for r in group]):.10g}",
                    "n": len(group),
                }
            )

    return len(rows_out), len(by_pft)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    xlsx_path = repo_root / "data" / "tanioka2020_algalCNP_SItable.xlsx"
    output_csv = repo_root / "output" / "tanioka2020_empirical.csv"
    output_by_pft_csv = repo_root / "output" / "tanioka2020_empirical_by_pft.csv"

    n_rows, n_pft = transform(
        xlsx_path=xlsx_path, output_csv=output_csv, output_by_pft_csv=output_by_pft_csv
    )
    print(f"Wrote {n_rows} rows to {output_csv}")
    print(f"Wrote {n_pft} rows to {output_by_pft_csv}")


if __name__ == "__main__":
    main()
