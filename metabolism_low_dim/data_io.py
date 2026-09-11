"""CSV input/output helpers for phase-diagram workflows."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .constants import ELEMENTS_FOR_RANGES, INDEPENDENT_MACROMOLECULES, MACROMOLECULES, NA_ELEMENTS, NA_POOLS, RESIDUE_ELEMENTS


def load_mass_fraction_ranges(csv_path: Path) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            macro = row["macromolecule"].strip()
            low = float(row["low"])
            high = float(row["high"])
            if low >= high:
                raise ValueError(f"Invalid range for {macro} in {csv_path}: {low} >= {high}")
            ranges[macro] = (low, high)

    missing = [m for m in INDEPENDENT_MACROMOLECULES if m not in ranges]
    if missing:
        raise ValueError(f"Missing macromolecules in {csv_path}: {missing}")
    return ranges


def load_element_ranges(csv_path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    element_ranges: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in MACROMOLECULES}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            macro = row["macromolecule"].strip()
            element = row["element"].strip()
            if macro not in MACROMOLECULES:
                raise ValueError(f"Unknown macromolecule in {csv_path}: {macro}")
            if element not in ELEMENTS_FOR_RANGES:
                raise ValueError(f"Unsupported element in {csv_path}: {element}")

            low = float(row["low"])
            high = float(row["high"])
            if low > high:
                raise ValueError(f"Invalid range in {csv_path} for {macro}/{element}: {low}>{high}")
            element_ranges[macro][element] = (low, high)

    for macro in MACROMOLECULES:
        for element in ELEMENTS_FOR_RANGES:
            if element not in element_ranges[macro]:
                raise ValueError(f"Missing range in {csv_path} for {macro}/{element}")
    return element_ranges


def load_residue_element_counts(csv_path: Path) -> tuple[tuple[str, ...], dict[str, dict[str, int]]]:
    aa_codes: list[str] = []
    residue_element_counts: dict[str, dict[str, int]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aa = row["aa"].strip()
            aa_codes.append(aa)
            residue_element_counts[aa] = {element: int(row[element]) for element in RESIDUE_ELEMENTS}

    if not aa_codes:
        raise ValueError(f"No residues found in {csv_path}")
    return tuple(aa_codes), residue_element_counts


def load_observed_aa_mean(csv_path: Path, aa_codes: tuple[str, ...]) -> dict[str, float]:
    observed_aa_mean: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aa = row["aa"].strip()
            observed_aa_mean[aa] = float(row["mean"])

    missing = [aa for aa in aa_codes if aa not in observed_aa_mean]
    if missing:
        raise ValueError(f"Missing amino-acid means in {csv_path}: {missing}")
    return observed_aa_mean


def load_empirical_ratios(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    nc_values: list[float] = []
    pc_values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in empirical CSV: {csv_path}")

    columns = set(rows[0].keys())

    if {"N_to_C", "P_to_C"}.issubset(columns):
        for row in rows:
            nc_values.append(float(row["N_to_C"]))
            pc_values.append(float(row["P_to_C"]))
    elif {"C_to_P", "N_to_P"}.issubset(columns):
        for row in rows:
            c_to_p = float(row["C_to_P"])
            n_to_p = float(row["N_to_P"])
            nc_values.append(n_to_p / c_to_p)
            pc_values.append(1.0 / c_to_p)
    elif {"cp_mean", "np_mean"}.issubset(columns):
        for row in rows:
            c_to_p = float(row["cp_mean"])
            n_to_p = float(row["np_mean"])
            nc_values.append(n_to_p / c_to_p)
            pc_values.append(1.0 / c_to_p)
    else:
        raise ValueError(
            "Empirical CSV must contain (N_to_C,P_to_C), (C_to_P,N_to_P), or (cp_mean,np_mean)"
        )

    return np.asarray(nc_values, dtype=float), np.asarray(pc_values, dtype=float)


def load_na_residue_element_counts(
    csv_path: Path,
) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {pool: {} for pool in NA_POOLS}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pool = row["pool"].strip().upper()
            nt = row["nucleotide"].strip().upper()
            if pool not in NA_POOLS:
                raise ValueError(f"Unknown nucleic-acid pool in {csv_path}: {pool}")
            counts[pool][nt] = {element: int(row[element]) for element in NA_ELEMENTS}

    for pool in NA_POOLS:
        if not counts[pool]:
            raise ValueError(f"No nucleotides found in {csv_path} for pool {pool}")
    return counts


def load_na_gc_content(csv_path: Path) -> tuple[float, float]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {csv_path}, got {len(rows)}")
    mean_gc = float(rows[0]["mean_gc"])
    concentration = float(rows[0]["concentration"])
    if not (0.0 < mean_gc < 1.0):
        raise ValueError(f"mean_gc must be in (0,1), got {mean_gc}")
    if concentration <= 0.0:
        raise ValueError(f"concentration must be positive, got {concentration}")
    return mean_gc, concentration


def load_na_pool_mix_mean(csv_path: Path) -> dict[str, float]:
    mix_mean: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pool = row["pool"].strip().upper()
            if pool not in NA_POOLS:
                raise ValueError(f"Unknown nucleic-acid pool in {csv_path}: {pool}")
            mix_mean[pool] = float(row["mean_mass_fraction"])

    missing = [pool for pool in NA_POOLS if pool not in mix_mean]
    if missing:
        raise ValueError(f"Missing pools in {csv_path}: {missing}")
    return mix_mean


def save_samples_csv(
    outpath: Path,
    mass_fractions: np.ndarray,
    nc_ratio: np.ndarray,
    oc_ratio: np.ndarray,
    pc_ratio: np.ndarray,
) -> None:
    macro_header = ",".join(MACROMOLECULES)
    header = f"{macro_header},N_to_C,O_to_C,P_to_C"
    matrix = np.column_stack((mass_fractions, nc_ratio, oc_ratio, pc_ratio))
    np.savetxt(outpath, matrix, delimiter=",", header=header, comments="")
