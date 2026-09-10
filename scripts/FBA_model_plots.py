#!/usr/bin/env python3
"""
Build a two-panel figure of an FBA model loaded from an SBML file: the 3D
feasible flux polytope (panel A), and, on panel B, how acetate overflow
depends on growth rate under a few different ways of resolving FBA's
degeneracy.

Usage
-----
    python FBA_model_plots.py                 # loads elemental_fba.xml from this folder
    python FBA_model_plots.py my_model.xml    # or point it at another SBML file

Panel A: the feasible polytope
-------------------------------
The steady-state flux space {v : S v = 0, lb <= v <= ub} is a convex polytope.
This model's polytope happens to be 3-dimensional, so it can be drawn exactly
(no information lost) on three chosen flux axes. For each of many directions on
the unit sphere we solve an LP that maximizes that direction over the polytope;
every optimum is a boundary point, and their convex hull is the polytope itself.
If you load a higher-dimensional model, the same picture is still valid but
becomes a *projection* (shadow) onto the three axes rather than the whole object.

Panel B: resolving FBA's degeneracy
------------------------------------
Maximizing growth alone leaves the rest of the flux distribution
underdetermined -- at a given growth rate, acetate efflux can fall anywhere
in a wide feasible range (shaded region, found by flux variability). Two
regularized alternatives pick a single point out of that range by adding a
penalty on flux to a single combined objective, maximize(growth - alpha *
penalty(v)), where the penalty is weighted per-reaction by ENZYME_STEPS (an
approximate count of the real enzymatic steps each lumped reaction stands
in for, as a stand-in for enzyme/proteome cost):
  - L1 penalty (sum of |flux|, LASSO-style): has constant marginal cost, so
    it collapses to an all-or-nothing step as alpha crosses a sparsity
    threshold, with no continuous trade-off in between.
  - L2 penalty (sum of flux^2, ridge-style): has rising marginal cost, so it
    yields a smooth, continuously-shifting optimum -- and reproduces a
    qualitatively realistic overflow-metabolism curve.
Both curves are traced by sweeping alpha alone (build_regularized_problem /
regularized_trace); no growth cap is needed; since the objective already
weighs growth against flux cost, each alpha pins down exactly one optimum.

Requirements: cobra, numpy, scipy, matplotlib, cvxpy  (pip install cobra scipy matplotlib cvxpy)
"""
import os
import sys
from collections.abc import Callable

import numpy as np
import cobra
import cvxpy as cp
from numpy.typing import NDArray
from cobra.util.array import create_stoichiometric_matrix
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_utils import get_palette, set_plotting_style

# ---------------------------------------------------------------- configuration
# Which three fluxes to use as axes. sign = +1 to plot the flux as-is,
# -1 to plot its negation (exchange fluxes are negative for uptake, so -1
# turns an uptake into a positive "uptake rate").
AXES = [
    ("EX_glc__D_e", -1, "glucose uptake flux\n (mmol/gDW/h)"),
    ("EX_ac_e",      1, "acetate efflux\n (mmol/gDW/h)"),
    ("DM_biomass",   1, "biomass flux (h$^{-1}$)")
]
PALETTE        = get_palette()
OBJECTIVE_RXN  = "DM_biomass"   # reaction whose max-growth optimum is starred
ACETATE_RXN    = "EX_ac_e"      # acetate exchange, plotted vs. growth in panel B
N_DIRECTIONS   = 700            # directions sampled on the sphere (more = smoother)
N_GROWTH_PTS   = 40             # growth-rate points swept for the feasible range
VIEW           = dict(elev=22, azim=-60)

# regularization strengths swept to trace each regularized curve (panel B):
# for a given alpha there is exactly one optimum (no growth cap needed), so
# sweeping alpha alone traces the curve directly.
REG_L1_ALPHAS  = np.logspace(-5, -1, 80)     # spans below/above the L1 sparsity threshold (~0.0045)
REG_L2_ALPHAS  = np.logspace(-5, -1.3, 80)   # spans near-vanilla growth down to near-total suppression

# Approximate number of real enzymatic steps each lumped reaction stands in
# for -- used as regularization weights, so that a unit of flux through a
# reaction that hides many real enzymes is penalized more.
# Reactions not listed default to a weight of 1.
ENZYME_STEPS = {
    "HEX": 1, "GLYC": 9, "PPP": 3, "PPP_RECYCLE": 3, "PDH": 1,
    "TCA": 8, "OXPHOS_NADH": 4, "OXPHOS_FADH2": 4, "ACK": 2, "SULFR": 3,
    # macromolecule assembly: these lump whole biosynthetic *processes*, not
    # single enzymes, so the counts below are rougher -- PROTEIN_SYN especially,
    # since translation (ribosome + aaRS + elongation factors) is reused per
    # residue rather than being a fixed pipeline.
    "STORAGE_SYN": 3,     # GlgC, GlgA, GlgB
    "LIPID_SYN": 10,      # FAS II (~8 enzymes) + head-group attachment (~2)
    "CELLWALL_SYN": 8,    # Mur pathway (~6) + PBP cross-linking (~2)
    # nucleotide biosynthesis (~7) + polymerase machinery (~3) + GS/GOGAT for
    # the NH4 that goes into the purine/pyrimidine rings (~2), rather than the
    # reaction's direct-amination shortcut
    "NUCACID_SYN": 12,
    # amino-acid backbone synthesis from a central precursor (~5, family-
    # dependent) + GS/GOGAT ammonia assimilation (~2) + aaRS per residue (5)
    # + ribosome/elongation factors (~3)
    "PROTEIN_SYN": 15,
}


def default_sbml_path() -> str:
    """Return the path to the bundled example model, ``SBML/elemental_fba.xml``,
    resolved relative to this script's location so it works regardless of the
    caller's current working directory.

    Returns
    -------
    str
        Absolute path to the default SBML file.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "SBML", "elemental_fba.xml")


def fibonacci_sphere(n: int) -> NDArray[np.float64]:
    """Generate roughly-uniform unit vectors on the sphere using a Fibonacci
    spiral, used to sample directions for the polytope's support function.

    Parameters
    ----------
    n : int
        Number of directions to sample.

    Returns
    -------
    NDArray[np.float64]
        Array of shape ``(n, 3)``; each row is a unit vector in R^3.
    """
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ])


def print_mass_balance(model: cobra.Model) -> None:
    """Print a mass/charge balance report for the model's internal (non-
    boundary) reactions, flagging any that don't close elementally.

    Parameters
    ----------
    model : cobra.Model
        The model to check. Not modified.
    """
    unbalanced = {}
    for r in model.reactions:
        if r.boundary:
            continue
        bal = r.check_mass_balance()
        if bal:
            unbalanced[r.id] = bal
    if unbalanced:
        print(f"mass/charge balance: {len(unbalanced)} unbalanced internal reaction(s):")
        for rid, bal in unbalanced.items():
            print(f"  {rid}: {bal}")
        print("  (expected: these consume/produce lumped macromolecule pools "
              "-- protein, lipid, nucleic acid, cell wall, storage -- that have "
              "no chemical formula, so they can't balance elementally)")
    else:
        print("mass/charge balance: all internal reactions balanced")


def boundary_points(
    model: cobra.Model,
    axes: list[tuple[str, int, str]],
    n_dir: int,
) -> NDArray[np.float64]:
    """Find a support point of the feasible flux polytope, projected onto
    three chosen flux axes, for each of many sampled directions. Every such
    point lies on the polytope's boundary, so their convex hull approximates
    the (projected) polytope itself.

    Parameters
    ----------
    model : cobra.Model
        The model whose flux polytope is being sampled. Its objective is
        overwritten repeatedly (once per direction) as a side effect.
    axes : list of (str, int, str)
        Three ``(reaction_id, sign, axis_label)`` tuples defining which flux
        each of the three plotted axes tracks; ``sign`` is +1 or -1 and flips
        the flux's displayed direction (e.g. so uptake reads as positive).
    n_dir : int
        Number of directions to sample on the unit sphere.

    Returns
    -------
    NDArray[np.float64]
        Array of shape ``(m, 3)`` of unique boundary points (m <= n_dir,
        since infeasible directions are skipped and duplicates are merged).
    """
    rxns = [model.reactions.get_by_id(a[0]) for a in axes]
    signs = [a[1] for a in axes]
    pts = []
    for d in fibonacci_sphere(n_dir):
        expr = sum(d[k] * signs[k] * rxns[k].flux_expression for k in range(3))
        model.objective = model.problem.Objective(expr, direction="max")
        if model.slim_optimize() is None:      # infeasible for this direction
            continue
        f = model.optimize().fluxes
        pts.append([signs[k] * f[rxns[k].id] for k in range(3)])
    return np.unique(np.round(np.array(pts), 6), axis=0)


def build_regularized_problem(
    model: cobra.Model,
    weights: dict[str, float],
    penalty: Callable[[cp.Expression], cp.Expression],
) -> tuple[cp.Problem, cp.Variable, cp.Parameter, dict[str, int]]:
    """Build the reusable convex problem for *regularized* FBA: a single
    combined objective, ``maximize(growth - alpha * penalty(v))``, rather
    than pFBA's two-step lexicographic version (max growth, then min flux
    without sacrificing it). For a given alpha this has exactly one
    optimum -- no growth cap needed -- so sweeping alpha alone traces the
    curve.

    Parameters
    ----------
    model : cobra.Model
        The model to build the problem from. Read-only (its own objective
        and solver state are not touched; a fresh cvxpy problem is built
        from its stoichiometry and bounds).
    weights : dict of str to float
        Per-reaction regularization weight, keyed by reaction id. Reactions
        not present default to a weight of 1 (see ``ENZYME_STEPS``).
    penalty : Callable[[cp.Expression], cp.Expression]
        The per-flux penalty function, applied elementwise: ``cp.abs`` for
        an L1 (LASSO-style) regularizer, or ``cp.square`` for an L2
        (ridge-style) regularizer.

    Returns
    -------
    problem : cp.Problem
        The cvxpy problem; call ``problem.solve()`` after setting
        ``alpha.value``.
    v : cp.Variable
        Length-``n_reactions`` flux vector, in the same order as
        ``model.reactions``.
    alpha : cp.Parameter
        The regularization strength; set ``alpha.value`` before each solve.
    idx : dict of str to int
        Maps each reaction id to its index into ``v``.
    """
    rxns = list(model.reactions)
    idx = {r.id: i for i, r in enumerate(rxns)}
    S = create_stoichiometric_matrix(model)
    lb = np.array([r.lower_bound for r in rxns])
    ub = np.array([r.upper_bound for r in rxns])
    w = np.array([weights.get(r.id, 1.0) for r in rxns])

    v = cp.Variable(len(rxns))
    alpha = cp.Parameter(nonneg=True)
    growth = v[idx[OBJECTIVE_RXN]]
    constraints = [S @ v == 0, v >= lb, v <= ub]
    objective = cp.Maximize(growth - alpha * cp.sum(cp.multiply(w, penalty(v))))
    problem = cp.Problem(objective, constraints)
    return problem, v, alpha, idx


def regularized_trace(
    model: cobra.Model,
    alphas: NDArray[np.float64],
    penalty: Callable[[cp.Expression], cp.Expression],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sweep alpha and solve the regularized problem (see
    ``build_regularized_problem``) at each value, tracing the single
    resulting (growth, acetate) curve. With the L1 penalty this is a step
    (all-or-nothing, LASSO-style, since constant marginal cost gives no
    interior optimum); with L2 it's smooth, since the penalty's rising
    marginal cost yields an interior optimum that shifts continuously with
    alpha.

    Parameters
    ----------
    model : cobra.Model
        The model to solve on. Not modified.
    alphas : NDArray[np.float64]
        Regularization strengths to sweep, in the order plotted.
    penalty : Callable[[cp.Expression], cp.Expression]
        ``cp.abs`` for L1 or ``cp.square`` for L2; passed through to
        ``build_regularized_problem``.

    Returns
    -------
    growth_vals : NDArray[np.float64]
        Realized growth (``OBJECTIVE_RXN`` flux) at each alpha.
    ac_vals : NDArray[np.float64]
        Realized acetate efflux (``ACETATE_RXN`` flux) at each alpha.
    """
    problem, v, alpha, idx = build_regularized_problem(model, ENZYME_STEPS, penalty)
    growth_index, ac_index = idx[OBJECTIVE_RXN], idx[ACETATE_RXN]

    growth_vals, ac_vals = [], []
    for a in alphas:
        alpha.value = a
        problem.solve(solver=cp.CLARABEL)
        growth_vals.append(v.value[growth_index])
        ac_vals.append(v.value[ac_index])
    return np.array(growth_vals), np.array(ac_vals)


def acetate_feasible_range(
    model: cobra.Model,
    gmax: float,
    n_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Sweep a growth-rate target and find the FBA-feasible range of
    acetate efflux at each one (min/max acetate among all optimal flux
    vectors achieving that growth rate) -- i.e. the full extent of FBA's
    degeneracy, found by flux variability rather than picking any single
    arbitrary vertex.

    Parameters
    ----------
    model : cobra.Model
        The model to solve on. Its ``OBJECTIVE_RXN`` bounds and objective
        are temporarily modified during the sweep, then restored before
        returning.
    gmax : float
        The model's unconstrained maximum growth rate; the sweep runs from
        just above 0 up to this value.
    n_points : int
        Number of growth-rate points to sweep.

    Returns
    -------
    growth_vals : NDArray[np.float64]
        The swept growth-rate values.
    ac_min : NDArray[np.float64]
        Minimum feasible acetate efflux at each growth rate.
    ac_max : NDArray[np.float64]
        Maximum feasible acetate efflux at each growth rate.
    """
    biomass = model.reactions.get_by_id(OBJECTIVE_RXN)
    ac_rxn = model.reactions.get_by_id(ACETATE_RXN)
    orig_bounds = biomass.bounds
    model.objective = OBJECTIVE_RXN

    growth_vals = np.linspace(1e-3, gmax, n_points)
    ac_min, ac_max = [], []
    for g in growth_vals:
        biomass.bounds = (g, g)   # fix growth exactly, then range over acetate
        model.objective = ac_rxn
        model.objective_direction = "min"
        ac_min.append(model.slim_optimize())
        model.objective_direction = "max"
        ac_max.append(model.slim_optimize())

    biomass.bounds = orig_bounds
    return growth_vals, np.array(ac_min), np.array(ac_max)


def main() -> None:
    """Load the SBML model (from the CLI argument, or the bundled example),
    print a mass-balance and dimensionality report, then build and save a
    two-panel figure: the 3D feasible flux polytope (panel A) and the
    FBA-feasible acetate range alongside the L1/L2-regularized growth-vs-
    acetate curves (panel B), to ``figures/elemental_FBA_panels.png`` and
    ``.svg``.
    """
    path = sys.argv[1] if len(sys.argv) > 1 else default_sbml_path()
    if not os.path.exists(path):
        sys.exit(f"SBML file not found: {path}")
    print(f"Loading {path}")
    model = cobra.io.read_sbml_model(path)
    print_mass_balance(model)

    # true dimension of the feasible polytope, for context
    S = cobra.util.array.create_stoichiometric_matrix(model)
    dim = len(model.reactions) - int(np.linalg.matrix_rank(S))
    print(f"reactions={len(model.reactions)}  metabolites={len(model.metabolites)}  "
          f"polytope dimension ~ {dim}"
          + ("" if dim <= 3 else "   (>3: the 3-axis figure is a projection)"))

    P = boundary_points(model, AXES, N_DIRECTIONS)
    hull = ConvexHull(P, qhull_options="QJ")
    print(f"boundary points={len(P)}  vertices={len(hull.vertices)}  faces={len(hull.simplices)}")

    # max-growth optimum (a vertex of the polytope)
    model.objective = OBJECTIVE_RXN
    sol = model.optimize()
    gmax = sol.objective_value
    star = [AXES[k][1] * sol.fluxes[AXES[k][0]] for k in range(3)]
    print("max-growth optimum (on the three axes):", [round(v, 3) for v in star])

    growth_vals, fba_ac_min, fba_ac_max = acetate_feasible_range(model, gmax, N_GROWTH_PTS)
    reg_l1_growth, reg_l1_ac = regularized_trace(model, REG_L1_ALPHAS, cp.abs)
    reg_l2_growth, reg_l2_ac = regularized_trace(model, REG_L2_ALPHAS, cp.square)

    # ---- plot ----
    # sized as two panels (B-C) of a 7"-wide figure -> 4.7" wide overall;
    # font sizes come from the shared style (plotting.mplstyle), tuned for
    # figures at this same scale
    set_plotting_style()

    fig = plt.figure(figsize=(4.7, 2.35))

    # panel A: the 3D feasible polytope
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.add_collection3d(Poly3DCollection(
        [P[s] for s in hull.simplices], alpha=0.4,
        facecolor=PALETTE["light_blue"], edgecolor=PALETTE["blue"], linewidths=0.4))
    V = P[hull.vertices]
    ax1.scatter(V[:, 0], V[:, 1], V[:, 2], s=8, color=PALETTE["dark_blue"], depthshade=False)
    ax1.set_xlabel(AXES[0][2], labelpad=-8)
    ax1.set_ylabel(AXES[1][2], labelpad=-8)
    ax1.set_zlabel(AXES[2][2], labelpad=-8)
    ax1.tick_params(pad=-4)
    ax1.xaxis.set_major_locator(MaxNLocator(3))
    ax1.yaxis.set_major_locator(MaxNLocator(3))
    ax1.zaxis.set_major_locator(MaxNLocator(3))
    ax1.view_init(**VIEW)

    # panel B: FBA-feasible range, plus the regularized L1 and L2 curves
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.fill_between(growth_vals, fba_ac_min, fba_ac_max, color=PALETTE["red"], alpha=0.25,
                      label="FBA-feasible range")
    # L1 sits flat at y=0, coincident with the bottom spine -- draw it above
    # the spines (default zorder ~2.5) so it isn't painted over
    ax2.plot(reg_l1_growth, reg_l1_ac, color=PALETTE["purple"], lw=2.2,
              label="L1 regularized", zorder=10)
    ax2.plot(reg_l2_growth, reg_l2_ac, color=PALETTE["dark_green"], lw=1.2, label="L2 regularized")

    ax2.set_xlabel("biomass flux (h$^{-1}$)")
    ax2.set_ylabel("acetate efflux  (mmol gDW$^{-1}$ h$^{-1}$)")
    ax2.xaxis.set_major_locator(MaxNLocator(3))
    ax2.yaxis.set_major_locator(MaxNLocator(3))
    ax2.legend()
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    fig.subplots_adjust(left=0.03, right=0.98, top=0.98, bottom=0.16, wspace=0.55)

    os.makedirs("figures", exist_ok=True)
    out_png = os.path.join("figures", "elemental_FBA_panels.png")
    out_svg = os.path.join("figures", "elemental_FBA_panels.svg")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
