#!/usr/bin/env python3
"""Construct a sampled N:C vs P:C phase diagram from macromolecular composition."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from metabolism_low_dim.data_io import (
    load_element_ranges,
    load_empirical_ratios,
    load_mass_fraction_ranges,
    load_na_gc_content,
    load_na_pool_mix_mean,
    load_na_residue_element_counts,
    load_observed_aa_mean,
    load_residue_element_counts,
    save_samples_csv,
)
from metabolism_low_dim.model import (
    compute_elemental_totals,
    sample_element_fractions,
    sample_mass_fractions_from_ranges,
    to_molar_ratio,
)
from metabolism_low_dim.plot_utils import set_plotting_style
from metabolism_low_dim.plotting import plot_phase_diagram


def _default_label_for_path(csv_path: Path) -> str:
    stem = csv_path.stem.lower()
    if "martiny" in stem:
        return "Martiny 2013"
    if "vrede" in stem:
        return "Vrede 2002"
    return csv_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a sampled N:C vs P:C phase diagram from macromolecular composition."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing numeric input CSV files (default: data).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200_000,
        help="Number of Monte Carlo samples to draw (default: 200000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("figures/constrained_elemental_stoich.png"),
        help="Output path for the plot (default: figures/constrained_elemental_stoich.png).",
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=Path("output/elemental_stoich_samples.csv"),
        help="Output path for the sampled CSV (default: output/elemental_stoich_samples.csv).",
    )
    parser.add_argument(
        "--protein-aa-mode",
        choices=("observed", "range"),
        default="observed",
        help=(
            "How to sample protein composition: 'observed' samples amino-acid frequencies "
            "from an empirical prior then computes protein C/N; 'range' uses the original "
            "independent protein C/N ranges (default: observed)."
        ),
    )
    parser.add_argument(
        "--protein-aa-concentration",
        type=float,
        default=250.0,
        help=(
            "Dirichlet concentration for amino-acid frequency sampling when "
            "--protein-aa-mode=observed (default: 250)."
        ),
    )
    parser.add_argument(
        "--nucleic-acid-nt-mode",
        choices=("observed", "range"),
        default="observed",
        help=(
            "How to sample nucleic-acid composition: 'observed' samples GC content from a "
            "Beta prior and derives nucleotide frequencies assuming equal strand usage; "
            "'range' uses the broad independent nucleic-acid ranges (default: observed)."
        ),
    )
    parser.add_argument(
        "--nucleic-acid-gc-concentration",
        type=float,
        default=None,
        help=(
            "Optional Beta-distribution concentration for GC-content sampling when "
            "--nucleic-acid-nt-mode=observed. If omitted, uses concentration from "
            "data/na_gc_content.csv."
        ),
    )
    parser.add_argument(
        "--nucleic-acid-pool-concentration",
        type=float,
        default=300.0,
        help=(
            "Dirichlet concentration for RNA/DNA pool mixing when "
            "--nucleic-acid-nt-mode=observed (default: 300)."
        ),
    )
    parser.add_argument(
        "--empirical-csv",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional empirical CSV overlay. Can be provided multiple times. "
            "Accepted column sets include: N_to_C and P_to_C; or C_to_P and N_to_P."
        ),
    )
    parser.add_argument(
        "--empirical-label",
        action="append",
        default=[],
        help=(
            "Label for an --empirical-csv entry. If provided, the count must match "
            "the number of --empirical-csv flags."
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    set_plotting_style()
    rng = np.random.default_rng(args.seed)

    mass_fraction_ranges = load_mass_fraction_ranges(args.data_dir / "mass_fraction_ranges.csv")
    element_ranges = load_element_ranges(args.data_dir / "element_ranges.csv")
    aa_codes, residue_element_counts = load_residue_element_counts(
        args.data_dir / "aa_residue_element_counts.csv"
    )
    observed_aa_mean = load_observed_aa_mean(args.data_dir / "aa_observed_mean.csv", aa_codes)
    na_residue_element_counts = load_na_residue_element_counts(
        args.data_dir / "na_residue_element_counts.csv"
    )
    na_gc_mean, na_gc_concentration_from_file = load_na_gc_content(args.data_dir / "na_gc_content.csv")
    na_pool_mix_mean = load_na_pool_mix_mean(args.data_dir / "na_pool_mix_mean.csv")

    if args.empirical_label and len(args.empirical_label) != len(args.empirical_csv):
        raise ValueError("--empirical-label count must match --empirical-csv count")

    empirical_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen_paths: set[Path] = set()

    default_candidates = (
        args.data_dir / "martiny_table_s2_latitude_metadata.csv",
        args.data_dir / "martiny_empirical_template.csv",
    )
    for candidate in default_candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            seen_paths.add(resolved)
            nc_ratio_emp, pc_ratio_emp = load_empirical_ratios(candidate)
            empirical_series.append((_default_label_for_path(candidate), nc_ratio_emp, pc_ratio_emp))
            break

    for i, empirical_csv_path in enumerate(args.empirical_csv):
        resolved = empirical_csv_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        nc_ratio_emp, pc_ratio_emp = load_empirical_ratios(empirical_csv_path)
        label = (
            args.empirical_label[i]
            if i < len(args.empirical_label)
            else _default_label_for_path(empirical_csv_path)
        )
        empirical_series.append((label, nc_ratio_emp, pc_ratio_emp))

    gc_concentration = (
        na_gc_concentration_from_file
        if args.nucleic_acid_gc_concentration is None
        else args.nucleic_acid_gc_concentration
    )

    mass_fractions = sample_mass_fractions_from_ranges(
        rng=rng,
        n_samples=args.n_samples,
        mass_fraction_ranges=mass_fraction_ranges,
    )
    element_fractions = sample_element_fractions(
        rng=rng,
        n_samples=args.n_samples,
        protein_aa_mode=args.protein_aa_mode,
        protein_aa_concentration=args.protein_aa_concentration,
        nucleic_acid_nt_mode=args.nucleic_acid_nt_mode,
        nucleic_acid_gc_concentration=gc_concentration,
        nucleic_acid_pool_concentration=args.nucleic_acid_pool_concentration,
        element_ranges=element_ranges,
        aa_codes=aa_codes,
        residue_element_counts=residue_element_counts,
        observed_aa_mean=observed_aa_mean,
        na_residue_element_counts=na_residue_element_counts,
        na_gc_mean=na_gc_mean,
        na_pool_mix_mean=na_pool_mix_mean,
    )
    totals = compute_elemental_totals(mass_fractions=mass_fractions, element_fractions=element_fractions)

    nc_ratio = to_molar_ratio(totals["N"], totals["C"], "N", "C")
    oc_ratio = to_molar_ratio(totals["O"], totals["C"], "O", "C")
    pc_ratio = to_molar_ratio(totals["P"], totals["C"], "P", "C")

    plot_png = args.plot_path
    samples_csv = args.samples_csv
    plot_png.parent.mkdir(parents=True, exist_ok=True)
    samples_csv.parent.mkdir(parents=True, exist_ok=True)

    save_samples_csv(samples_csv, mass_fractions, nc_ratio, oc_ratio, pc_ratio)
    plot_phase_diagram(
        plot_png,
        nc_ratio,
        pc_ratio,
        mass_fractions,
        empirical_series=empirical_series,
    )
    return plot_png, samples_csv


def main() -> None:
    args = parse_args()
    plot_png, samples_csv = run(args)

    print("Wrote:")
    print(f"- {plot_png}")
    print(f"- {samples_csv}")


if __name__ == "__main__":
    main()