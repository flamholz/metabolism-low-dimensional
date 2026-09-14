"""Core simulation and stoichiometry math.

Known modeling limitations, left as-is deliberately (not bugs):

- Growth rate, RNA:DNA ratio, nucleic-acid mass fraction, genomic GC
  content, and amino-acid usage are all sampled independently, even though
  real cells couple them (e.g. faster growth is associated with more
  ribosomal RNA; genomic GC content correlates with amino-acid usage
  bias). None of that covariance is modeled here, so the sampled cloud is
  likely more diffuse than what real organisms actually occupy.
- Sugar and lipid mass fractions (`mass_fraction_ranges.csv`) are
  intentionally left with wide, independent ranges rather than a tight
  joint cap. The comparison dataset (Martiny et al. 2013) includes marine
  phytoplankton, which can store large amounts of both carbohydrate and
  lipid under nutrient stress, so an invented cap here would risk
  excluding real biology rather than fixing a bug.
"""

from __future__ import annotations

import numpy as np

from .constants import ATOMIC_MASS, MACROMOLECULES, NA_ELEMENTS, NA_POOLS, RESIDUE_ELEMENTS

PROTEIN_AA_MODES = ("observed", "empirical", "range")
NUCLEIC_ACID_NT_MODES = ("observed", "range")


def sample_mass_fractions_from_ranges(
    rng: np.random.Generator,
    n_samples: int,
    mass_fraction_ranges: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Sample non-sugar pools independently; sugar = 1 - sum.

    Draws are rejected when sugar < 0 (infeasible), so we oversample in batches
    until n_samples valid rows are collected.
    """
    collected: list[np.ndarray] = []
    n_collected = 0
    while n_collected < n_samples:
        batch = max((n_samples - n_collected) * 4, 10_000)
        prot = rng.uniform(*mass_fraction_ranges["protein"], size=batch)
        na = rng.uniform(*mass_fraction_ranges["nucleic_acid"], size=batch)
        membrane_lipid = rng.uniform(*mass_fraction_ranges["membrane_lipid"], size=batch)
        storage_lipid = rng.uniform(*mass_fraction_ranges["storage_lipid"], size=batch)
        metabolite = rng.uniform(*mass_fraction_ranges["metabolite"], size=batch)
        sugar = 1.0 - prot - na - membrane_lipid - storage_lipid - metabolite
        valid = sugar >= 0.0
        rows = np.stack(
            [
                prot[valid],
                sugar[valid],
                na[valid],
                membrane_lipid[valid],
                storage_lipid[valid],
                metabolite[valid],
            ],
            axis=1,
        )
        collected.append(rows)
        n_collected += rows.shape[0]
    return np.concatenate(collected, axis=0)[:n_samples]


def _sample_ranged_elements_with_closure(
    rng: np.random.Generator,
    n_samples: int,
    ranges: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Sample each element's mass fraction independently and uniformly within
    its own range, rejecting and redrawing any row whose total exceeds 1.0.

    Mass fractions of real chemical constituents cannot exceed the whole;
    this enforces that hard constraint (not an empirical/biological
    assumption) since the per-element ranges are set independently and can
    otherwise combine into physically impossible compositions.
    """
    elements = tuple(ranges.keys())
    collected: dict[str, list[np.ndarray]] = {element: [] for element in elements}
    n_collected = 0
    while n_collected < n_samples:
        batch = max((n_samples - n_collected) * 4, 10_000)
        draws = {element: rng.uniform(*ranges[element], size=batch) for element in elements}
        total = sum(draws.values())
        valid = total <= 1.0
        for element in elements:
            collected[element].append(draws[element][valid])
        n_collected += int(valid.sum())
    return {element: np.concatenate(collected[element])[:n_samples] for element in elements}


def _protein_cno_from_frequencies(
    aa_frequencies: np.ndarray,
    aa_codes: tuple[str, ...],
    residue_element_counts: dict[str, dict[str, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute protein C/N/O mass fractions from an (n_samples, len(aa_codes))
    matrix of per-sample amino-acid frequencies (columns in ``aa_codes`` order).
    """
    c_counts = np.array([residue_element_counts[aa]["C"] for aa in aa_codes], dtype=float)
    n_counts = np.array([residue_element_counts[aa]["N"] for aa in aa_codes], dtype=float)
    o_counts = np.array([residue_element_counts[aa]["O"] for aa in aa_codes], dtype=float)
    residue_masses = np.array(
        [
            sum(residue_element_counts[aa][elem] * ATOMIC_MASS[elem] for elem in RESIDUE_ELEMENTS)
            for aa in aa_codes
        ],
        dtype=float,
    )

    mean_residue_mass = aa_frequencies @ residue_masses
    c_mass_fraction = (aa_frequencies @ c_counts) * ATOMIC_MASS["C"] / mean_residue_mass
    n_mass_fraction = (aa_frequencies @ n_counts) * ATOMIC_MASS["N"] / mean_residue_mass
    o_mass_fraction = (aa_frequencies @ o_counts) * ATOMIC_MASS["O"] / mean_residue_mass
    return c_mass_fraction, n_mass_fraction, o_mass_fraction


def sample_protein_cno_from_aa(
    rng: np.random.Generator,
    n_samples: int,
    concentration: float,
    aa_codes: tuple[str, ...],
    residue_element_counts: dict[str, dict[str, int]],
    observed_aa_mean: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample protein C/N/O mass fractions from a synthetic Dirichlet
    distribution of amino-acid frequencies, centered on ``observed_aa_mean``
    with concentration ``concentration`` (higher = tighter around the mean).

    This applies independent per-amino-acid noise around a single mean, so it
    cannot reproduce real inter-genome covariance in amino-acid usage (e.g.
    genomic GC content driving several amino acids' frequencies together).
    See ``sample_protein_cno_from_empirical_aa`` for an alternative that
    resamples whole real genome compositions instead.
    """
    mean = np.array([observed_aa_mean[aa] for aa in aa_codes], dtype=float)
    mean /= mean.sum()
    alpha = mean * concentration
    aa_frequencies = rng.dirichlet(alpha, size=n_samples)
    return _protein_cno_from_frequencies(aa_frequencies, aa_codes, residue_element_counts)


def sample_protein_cno_from_empirical_aa(
    rng: np.random.Generator,
    n_samples: int,
    aa_codes: tuple[str, ...],
    residue_element_counts: dict[str, dict[str, int]],
    empirical_aa_codes: tuple[str, ...],
    empirical_frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample protein C/N/O mass fractions by resampling whole amino-acid
    frequency vectors from real sequenced genomes (with replacement), rather
    than drawing synthetic per-element noise around a single mean.

    Each Monte Carlo sample gets one genome's actual, internally-consistent
    amino-acid composition (see data/moura2013_aa_frequencies_by_genome.csv),
    so this preserves whatever real covariance exists between amino acids
    within a genome -- unlike ``sample_protein_cno_from_aa``, which perturbs
    each amino acid independently. Genomes are resampled with equal
    probability regardless of domain, matching the corpus's real composition
    (predominantly Bacteria).

    Parameters
    ----------
    empirical_aa_codes : tuple of str
        Column order of ``empirical_frequencies``; reordered internally to
        match ``aa_codes`` if the two differ (as long as the code sets match).
    empirical_frequencies : np.ndarray
        Shape ``(n_genomes, len(empirical_aa_codes))``.
    """
    if set(empirical_aa_codes) != set(aa_codes):
        raise ValueError(
            "empirical_aa_codes and aa_codes must contain the same amino acids: "
            f"{set(empirical_aa_codes)} != {set(aa_codes)}"
        )
    column_for = {aa: i for i, aa in enumerate(empirical_aa_codes)}
    reorder = [column_for[aa] for aa in aa_codes]
    empirical_frequencies = empirical_frequencies[:, reorder]

    n_genomes = empirical_frequencies.shape[0]
    genome_idx = rng.integers(0, n_genomes, size=n_samples)
    aa_frequencies = empirical_frequencies[genome_idx]
    return _protein_cno_from_frequencies(aa_frequencies, aa_codes, residue_element_counts)


def sample_nucleic_acid_from_gc(
    rng: np.random.Generator,
    n_samples: int,
    gc_mean: float,
    gc_concentration: float,
    pool_concentration: float,
    na_residue_element_counts: dict[str, dict[str, dict[str, int]]],
    na_pool_mix_mean: dict[str, float],
) -> dict[str, np.ndarray]:
    """Sample nucleic-acid C/N/O/P mass fractions from GC content and RNA:DNA mix.

    For each of the RNA and DNA pools, nucleotide frequencies are derived
    from a per-sample GC content (assuming G/C and A/U(T) each split their
    share of frequency evenly, i.e. a Chargaff's-second-rule-style
    symmetry within each pool -- a standard approximation, not an exact
    biological constraint). RNA and DNA pools are then mixed per sample
    via a Dirichlet draw centered on ``na_pool_mix_mean``.

    Parameters
    ----------
    gc_mean, gc_concentration : float
        Beta-distribution mean and concentration for per-sample GC
        content (shared by both RNA and DNA pools within a given sample).
    pool_concentration : float
        Dirichlet concentration for the RNA:DNA mass-fraction mix,
        centered on ``na_pool_mix_mean`` (higher = tighter around the mean).
    na_residue_element_counts : dict
        Dehydrated (polymer) element counts per pool and nucleotide code,
        as returned by ``data_io.load_na_residue_element_counts``.
    na_pool_mix_mean : dict
        Mean mass-fraction split between the "RNA" and "DNA" pools.

    Returns
    -------
    dict of str to np.ndarray
        Per-sample mass fraction for each of "C", "N", "O", "P".
    """
    gc_alpha = gc_mean * gc_concentration
    gc_beta = (1.0 - gc_mean) * gc_concentration
    gc_content = rng.beta(gc_alpha, gc_beta, size=n_samples)

    gc_nts = {"RNA": ("G", "C"), "DNA": ("G", "C")}

    per_pool_element_mass_fraction: dict[str, dict[str, np.ndarray]] = {}
    for pool in NA_POOLS:
        nt_codes = tuple(na_residue_element_counts[pool].keys())
        nt_freq_list = []
        for nt in nt_codes:
            if nt in gc_nts[pool]:
                nt_freq_list.append(gc_content / 2.0)
            else:
                nt_freq_list.append((1.0 - gc_content) / 2.0)
        nt_frequencies = np.stack(nt_freq_list, axis=1)

        residue_masses = np.array(
            [
                sum(na_residue_element_counts[pool][nt][elem] * ATOMIC_MASS[elem] for elem in NA_ELEMENTS)
                for nt in nt_codes
            ],
            dtype=float,
        )
        mean_residue_mass = nt_frequencies @ residue_masses

        pool_fracs: dict[str, np.ndarray] = {}
        for element in ("C", "N", "O", "P"):
            element_counts = np.array(
                [na_residue_element_counts[pool][nt][element] for nt in nt_codes],
                dtype=float,
            )
            pool_fracs[element] = (
                (nt_frequencies @ element_counts) * ATOMIC_MASS[element] / mean_residue_mass
            )
        per_pool_element_mass_fraction[pool] = pool_fracs

    pool_mean = np.array([na_pool_mix_mean[pool] for pool in NA_POOLS], dtype=float)
    pool_mean /= pool_mean.sum()
    pool_alpha = pool_mean * pool_concentration
    pool_fractions = rng.dirichlet(pool_alpha, size=n_samples)

    sampled = {element: np.zeros(n_samples) for element in ("C", "N", "O", "P")}
    for i, pool in enumerate(NA_POOLS):
        for element in sampled:
            sampled[element] += pool_fractions[:, i] * per_pool_element_mass_fraction[pool][element]
    return sampled


def sample_element_fractions(
    rng: np.random.Generator,
    n_samples: int,
    protein_aa_mode: str,
    protein_aa_concentration: float,
    nucleic_acid_nt_mode: str,
    nucleic_acid_gc_concentration: float,
    nucleic_acid_pool_concentration: float,
    element_ranges: dict[str, dict[str, tuple[float, float]]],
    aa_codes: tuple[str, ...],
    residue_element_counts: dict[str, dict[str, int]],
    observed_aa_mean: dict[str, float],
    na_residue_element_counts: dict[str, dict[str, dict[str, int]]],
    na_gc_mean: float,
    na_pool_mix_mean: dict[str, float],
    empirical_aa_codes: tuple[str, ...] | None = None,
    empirical_aa_frequencies: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Sample per-macromolecule C/N/O/P mass fractions for every pool in MACROMOLECULES.

    Dispatches each macromolecule to whichever sampling method its mode
    selects:

    - protein: ``sample_protein_cno_from_aa`` (Dirichlet noise around
      ``observed_aa_mean``) if ``protein_aa_mode == "observed"``,
      ``sample_protein_cno_from_empirical_aa`` (bootstrap real genomes,
      via ``empirical_aa_codes``/``empirical_aa_frequencies``) if
      ``"empirical"``; ``"range"`` falls through to the same
      independent-range sampling as every other macromolecule. P is
      always drawn independently from ``element_ranges["protein"]["P"]``
      regardless of mode, since amino-acid composition doesn't determine it.
    - nucleic_acid: ``sample_nucleic_acid_from_gc`` if
      ``nucleic_acid_nt_mode == "observed"``; ``"range"`` falls through to
      independent-range sampling.
    - every other macromolecule: always independent-range sampling, via
      ``_sample_ranged_elements_with_closure``.

    Raises
    ------
    ValueError
        If ``protein_aa_mode`` is not one of ``PROTEIN_AA_MODES`` or
        ``nucleic_acid_nt_mode`` is not one of ``NUCLEIC_ACID_NT_MODES``.

    Returns
    -------
    dict of str to dict of str to np.ndarray
        ``result[macro][element]`` is the per-sample mass fraction of
        ``element`` ("C"/"N"/"O"/"P") within macromolecule ``macro``, for
        every ``macro`` in ``MACROMOLECULES``.
    """
    if protein_aa_mode not in PROTEIN_AA_MODES:
        raise ValueError(f"protein_aa_mode must be one of {PROTEIN_AA_MODES}, got {protein_aa_mode!r}")
    if nucleic_acid_nt_mode not in NUCLEIC_ACID_NT_MODES:
        raise ValueError(
            f"nucleic_acid_nt_mode must be one of {NUCLEIC_ACID_NT_MODES}, got {nucleic_acid_nt_mode!r}"
        )

    sampled: dict[str, dict[str, np.ndarray]] = {}

    protein_c: np.ndarray | None = None
    protein_n: np.ndarray | None = None
    protein_o: np.ndarray | None = None
    if protein_aa_mode == "observed":
        protein_c, protein_n, protein_o = sample_protein_cno_from_aa(
            rng=rng,
            n_samples=n_samples,
            concentration=protein_aa_concentration,
            aa_codes=aa_codes,
            residue_element_counts=residue_element_counts,
            observed_aa_mean=observed_aa_mean,
        )
    elif protein_aa_mode == "empirical":
        protein_c, protein_n, protein_o = sample_protein_cno_from_empirical_aa(
            rng=rng,
            n_samples=n_samples,
            aa_codes=aa_codes,
            residue_element_counts=residue_element_counts,
            empirical_aa_codes=empirical_aa_codes,
            empirical_frequencies=empirical_aa_frequencies,
        )

    na_sampled: dict[str, np.ndarray] | None = None
    if nucleic_acid_nt_mode == "observed":
        na_sampled = sample_nucleic_acid_from_gc(
            rng=rng,
            n_samples=n_samples,
            gc_mean=na_gc_mean,
            gc_concentration=nucleic_acid_gc_concentration,
            pool_concentration=nucleic_acid_pool_concentration,
            na_residue_element_counts=na_residue_element_counts,
            na_pool_mix_mean=na_pool_mix_mean,
        )

    for macro in MACROMOLECULES:
        if macro == "protein" and protein_aa_mode in ("observed", "empirical"):
            low, high = element_ranges[macro]["P"]
            sampled[macro] = {
                "C": protein_c,
                "N": protein_n,
                "O": protein_o,
                "P": rng.uniform(low, high, size=n_samples),
            }
        elif macro == "nucleic_acid" and nucleic_acid_nt_mode == "observed":
            sampled[macro] = {element: na_sampled[element] for element in ("C", "N", "O", "P")}
        else:
            sampled[macro] = _sample_ranged_elements_with_closure(
                rng=rng,
                n_samples=n_samples,
                ranges=element_ranges[macro],
            )
    return sampled


def compute_elemental_totals(
    mass_fractions: np.ndarray,
    element_fractions: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Aggregate per-macromolecule element mass fractions into whole-cell totals.

    For each element, sums ``mass_fractions[:, i] *
    element_fractions[macro][element]`` over all macromolecules (in
    ``MACROMOLECULES`` order), giving each sample's whole-cell mass
    fraction of that element.

    Parameters
    ----------
    mass_fractions : np.ndarray
        Shape ``(n_samples, len(MACROMOLECULES))``, columns in
        ``MACROMOLECULES`` order (e.g. as returned by
        ``sample_mass_fractions_from_ranges``).
    element_fractions : dict
        ``element_fractions[macro][element]`` is that macromolecule's
        per-sample mass fraction of the element, as returned by
        ``sample_element_fractions``.

    Returns
    -------
    dict of str to np.ndarray
        Whole-cell mass fraction per sample, for "C", "N", "O", "P".
    """
    totals = {
        "C": np.zeros(mass_fractions.shape[0]),
        "N": np.zeros(mass_fractions.shape[0]),
        "O": np.zeros(mass_fractions.shape[0]),
        "P": np.zeros(mass_fractions.shape[0]),
    }
    for i, macro in enumerate(MACROMOLECULES):
        for element in totals:
            totals[element] += mass_fractions[:, i] * element_fractions[macro][element]
    return totals


def to_molar_ratio(
    numerator_mass_fraction: np.ndarray,
    denominator_mass_fraction: np.ndarray,
    numerator_element: str,
    denominator_element: str,
) -> np.ndarray:
    """Convert a mass-fraction ratio to a molar (mole:mole) ratio.

    Divides each mass fraction by its element's atomic mass before taking
    the ratio: ``(numerator_mass_fraction / ATOMIC_MASS[numerator_element])
    / (denominator_mass_fraction / ATOMIC_MASS[denominator_element])``.
    """
    numerator_moles = numerator_mass_fraction / ATOMIC_MASS[numerator_element]
    denominator_moles = denominator_mass_fraction / ATOMIC_MASS[denominator_element]
    return numerator_moles / denominator_moles
