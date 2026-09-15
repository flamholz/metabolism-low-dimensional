#!/usr/bin/env python3
"""Transform Moura, Savageau & Alves (2013) Table S2 into amino-acid frequency data.

Source
------
Moura A, Savageau MA, Alves R (2013). "Relative Amino Acid Composition
Signatures of Organisms and Environments." PLoS ONE 8(10): e77319.
https://doi.org/10.1371/journal.pone.0077319

Table S2 (data/moura2013_tableS2.xlsx) reports per-organism relative
amino-acid composition for 1086 fully-sequenced, unicellular KEGG genomes
(961 Bacteria, 72 Archaea, 53 Eukarya), in three flavors per amino acid:
plain relative frequency ('f...'), and two expression-weighted variants
('d...', 'CAI...'). This script uses only the plain 'f...' columns, since
the model represents bulk average protein composition rather than
differential expression.

Output
------
- output/moura2013_aa_frequencies.json: per amino acid, the mean/std/min/max
  of its 'f...' frequency across all 1086 genomes (unweighted by domain or
  habitat), plus per-domain genome counts for context.
- output/moura2013_aa_frequencies_by_genome.csv: the full per-genome table
  (organism, domain, then the 20 rescaled amino-acid frequencies), used to
  sample directly from real genome compositions rather than a synthetic
  distribution around the mean.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import openpyxl

AA_COLUMN_TO_CODE = {
    "fAla": "A", "fArg": "R", "fAsn": "N", "fAsp": "D", "fCys": "C",
    "fGln": "Q", "fGlu": "E", "fGly": "G", "fHis": "H", "fIle": "I",
    "fLeu": "L", "fLys": "K", "fMet": "M", "fPhe": "F", "fPro": "P",
    "fSer": "S", "fThr": "T", "fTrp": "W", "fTyr": "Y", "fVal": "V",
}
AA_CODES = tuple(AA_COLUMN_TO_CODE.values())


def load_genome_records(xlsx_path: Path) -> list[dict]:
    """Return one record per genome: organism, domain, and rescaled amino-acid frequencies.

    Per-genome frequencies don't always sum to exactly 1.0 (observed range
    in this table: 0.993-1.0), likely from rounding or amino acids outside
    the 20 standard residues being excluded; each record's frequencies are
    rescaled here so they do.
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Table S2"]
    rows = list(ws.iter_rows(values_only=True))

    header = rows[1]
    column_index = {name: i for i, name in enumerate(header) if name in AA_COLUMN_TO_CODE}
    if len(column_index) != len(AA_COLUMN_TO_CODE):
        missing = set(AA_COLUMN_TO_CODE) - set(column_index)
        raise ValueError(f"Missing expected columns in {xlsx_path}: {missing}")
    domain_index = header.index("Domain")

    records: list[dict] = []
    for row in rows[2:]:
        if row[0] is None:
            continue
        raw = {code: float(row[column_index[column]]) for column, code in AA_COLUMN_TO_CODE.items()}
        row_sum = sum(raw.values())
        records.append({
            "organism": row[0],
            "domain": row[domain_index],
            "frequencies": {code: value / row_sum for code, value in raw.items()},
        })

    return records


def transform(xlsx_path: Path, output_json: Path, output_csv: Path) -> int:
    records = load_genome_records(xlsx_path)
    n_genomes = len(records)

    domain_counts: dict[str, int] = {}
    for record in records:
        domain_counts[record["domain"]] = domain_counts.get(record["domain"], 0) + 1

    amino_acids = {}
    for code in AA_CODES:
        arr = np.array([record["frequencies"][code] for record in records], dtype=float)
        amino_acids[code] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary = {
        "description": (
            "Amino-acid composition frequency statistics computed from the unweighted "
            "per-genome 'f' frequencies (Table S2, faa columns) in Moura, Savageau & Alves "
            "(2013), 'Relative Amino Acid Composition Signatures of Organisms and "
            "Environments', PLoS ONE 8(10):e77319, "
            "https://doi.org/10.1371/journal.pone.0077319. Computed by "
            "scripts/transform_moura2013.py from data/moura2013_tableS2.xlsx. 'mean'/'std'/"
            "'min'/'max' describe each amino acid's relative frequency across all genomes, "
            "each genome weighted equally regardless of domain or habitat. The full per-genome "
            "table (used for direct empirical resampling rather than a synthetic distribution "
            "around the mean) is in output/moura2013_aa_frequencies_by_genome.csv."
        ),
        "n_genomes": n_genomes,
        "n_genomes_by_domain": domain_counts,
        "amino_acids": amino_acids,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["organism", "domain", *AA_CODES])
        for record in records:
            writer.writerow([
                record["organism"],
                record["domain"],
                *(f"{record['frequencies'][code]:.10g}" for code in AA_CODES),
            ])

    return n_genomes


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    xlsx_path = repo_root / "data" / "moura2013_tableS2.xlsx"
    output_json = repo_root / "output" / "moura2013_aa_frequencies.json"
    output_csv = repo_root / "output" / "moura2013_aa_frequencies_by_genome.csv"

    n_genomes = transform(xlsx_path=xlsx_path, output_json=output_json, output_csv=output_csv)
    print(f"Wrote amino-acid frequency data from {n_genomes} genomes to:")
    print(f"- {output_json}")
    print(f"- {output_csv}")


if __name__ == "__main__":
    main()
