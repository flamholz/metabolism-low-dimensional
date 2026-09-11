"""Plotting utilities for phase diagrams."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np




def plot_phase_diagram(
    outpath: Path,
    nc_ratio: np.ndarray,
    pc_ratio: np.ndarray,
    mass_fractions: np.ndarray,
    empirical_series: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> None:
    protein_fraction = mass_fractions[:, 0]
    nc_percent = 100.0 * nc_ratio
    pc_percent = 100.0 * pc_ratio

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5), constrained_layout=True)

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
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
