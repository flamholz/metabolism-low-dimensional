#!/usr/bin/env python3
"""Monte Carlo sample macromolecular composition and elemental ratios.

This is the slow step: it draws mass fractions and per-macromolecule
element fractions, then computes N:C, O:C, and P:C molar ratios and writes
them to a CSV. Plotting is handled separately by
``plot_elemental_stoich.py``, which reads that CSV back in, so aesthetic
changes to the figure don't require re-sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from metabolism_low_dim.data_io import (
    load_element_ranges,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo sample macromolecular composition and elemental ratios."
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
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(args.seed)

    mass_fraction_ranges = load_mass_fraction_ranges(args.data_dir / "mass_fraction_ranges.json")
    element_ranges = load_element_ranges(args.data_dir / "element_ranges.json")
    aa_codes, residue_element_counts = load_residue_element_counts(
        args.data_dir / "aa_residue_element_counts.json"
    )
    observed_aa_mean = load_observed_aa_mean(args.data_dir / "aa_observed_mean.csv", aa_codes)
    na_residue_element_counts = load_na_residue_element_counts(
        args.data_dir / "na_residue_element_counts.json"
    )
    na_gc_mean, na_gc_concentration_from_file = load_na_gc_content(args.data_dir / "na_gc_content.csv")
    na_pool_mix_mean = load_na_pool_mix_mean(args.data_dir / "na_pool_mix_mean.csv")

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

    samples_csv = args.samples_csv
    samples_csv.parent.mkdir(parents=True, exist_ok=True)
    save_samples_csv(samples_csv, mass_fractions, nc_ratio, oc_ratio, pc_ratio)
    return samples_csv


def main() -> None:
    args = parse_args()
    samples_csv = run(args)
    print(f"Wrote {samples_csv}")


if __name__ == "__main__":
    main()
