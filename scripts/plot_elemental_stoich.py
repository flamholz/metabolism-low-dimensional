#!/usr/bin/env python3
"""Plot the N:C vs P:C phase diagram from a samples CSV.

Reads the samples CSV written by ``sample_elemental_stoich.py`` (the slow
step) and produces the plot. Kept separate so plot-aesthetic changes don't
require re-sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from metabolism_low_dim.data_io import load_empirical_ratios, load_samples_csv
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
        description="Plot the N:C vs P:C phase diagram from a samples CSV."
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=Path("output/elemental_stoich_samples.csv"),
        help="Input samples CSV, written by sample_elemental_stoich.py "
        "(default: output/elemental_stoich_samples.csv).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to look in for the default empirical overlay (default: data).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("figures/constrained_elemental_stoich.png"),
        help="Output path for the plot (default: figures/constrained_elemental_stoich.png).",
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


def run(args: argparse.Namespace) -> Path:
    set_plotting_style()

    mass_fractions, ratios = load_samples_csv(args.samples_csv)
    nc_ratio = ratios["N_to_C"]
    pc_ratio = ratios["P_to_C"]

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

    plot_png = args.plot_path
    plot_png.parent.mkdir(parents=True, exist_ok=True)
    plot_phase_diagram(
        plot_png,
        nc_ratio,
        pc_ratio,
        mass_fractions,
        empirical_series=empirical_series,
    )
    return plot_png


def main() -> None:
    args = parse_args()
    plot_png = run(args)
    print(f"Wrote {plot_png}")


if __name__ == "__main__":
    main()
