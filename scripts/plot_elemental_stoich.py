#!/usr/bin/env python3
"""Plot N:C vs P:C elemental stoichiometry from a samples CSV.

Reads the samples CSV written by ``sample_elemental_stoich.py`` (the slow
step) and produces the plot. Kept separate so plot-aesthetic changes don't
require re-sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metabolism_low_dim.data_io import load_empirical_ratios, load_samples_csv
from metabolism_low_dim.plot_utils import set_plotting_style


def plot_elemental_stoichiometry(
    outpath: Path,
    nc_ratio: np.ndarray,
    pc_ratio: np.ndarray,
    mass_fractions: np.ndarray,
    empirical_series: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> None:
    protein_fraction = mass_fractions[:, 0]
    nc_percent = 100.0 * nc_ratio
    pc_percent = 100.0 * pc_ratio

    fig, ax = plt.subplots(1, 1, figsize=(3.05, 2.35), constrained_layout=True)

    hb = ax.hexbin(
        nc_percent,
        pc_percent,
        C=protein_fraction,
        reduce_C_function=np.mean,
        gridsize=70,
        mincnt=1,
        cmap="magma",
    )
    ax.set_title("P:C vs N:C")
    ax.set_xlabel("N:C (molar %)")
    ax.set_ylabel("P:C (molar %)")

    redfield_nc_percent = 100.0 * (16.0 / 106.0)
    redfield_pc_percent = 100.0 * (1.0 / 106.0)
    ax.scatter(
        redfield_nc_percent,
        redfield_pc_percent,
        marker="*",
        s=180,
        c="cyan",
        edgecolors="black",
        linewidths=0.8,
        label="Redfield 106:16:1",
        zorder=5,
    )

    if empirical_series:
        markers = ["o", "s", "^", "D", "P", "X", "v"]
        facecolors = ["white", "gold", "deepskyblue", "lime", "tomato", "violet", "wheat"]
        for i, (label, empirical_nc_ratio, empirical_pc_ratio) in enumerate(empirical_series):
            ax.scatter(
                100.0 * empirical_nc_ratio,
                100.0 * empirical_pc_ratio,
                s=32,
                marker=markers[i % len(markers)],
                c=facecolors[i % len(facecolors)],
                edgecolors="black",
                linewidths=0.7,
                alpha=0.95,
                label=label,
                zorder=4,
            )

    ax.legend(loc="best", frameon=True)

    cbar = fig.colorbar(hb, ax=ax, shrink=0.95)
    cbar.set_label("Protein mass fraction")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _default_label_for_path(csv_path: Path) -> str:
    stem = csv_path.stem.lower()
    if "martiny" in stem:
        return "Martiny 2013"
    if "vrede" in stem:
        return "Vrede 2002"
    return csv_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot N:C vs P:C elemental stoichiometry from a samples CSV."
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
    plot_elemental_stoichiometry(
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
