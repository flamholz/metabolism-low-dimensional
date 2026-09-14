"""Generate this repo's figures.

Run everything with `snakemake --cores 1` from the repo root (every script
here resolves its own paths relative to the repo root or its own file
location, not the caller's cwd, so no `workdir:` directive is needed).
"""

CONSTRAINED_ELEMENTAL_STOICH_PLOT = "figures/constrained_elemental_stoich.png"
CONSTRAINED_ELEMENTAL_STOICH_SAMPLES = "output/elemental_stoich_samples.csv"

rule all:
    input:
        "figures/elemental_FBA_panels.png",
        "figures/elemental_FBA_panels.svg",
        CONSTRAINED_ELEMENTAL_STOICH_PLOT,
        CONSTRAINED_ELEMENTAL_STOICH_SAMPLES,


rule fba_plots:
    input:
        script="scripts/FBA_model_plots.py",
        plot_utils="metabolism_low_dim/plot_utils.py",
        style="metabolism_low_dim/plotting.mplstyle",
        sbml="SBML/elemental_fba.xml",
    output:
        "figures/elemental_FBA_panels.png",
        "figures/elemental_FBA_panels.svg",
    shell:
        "python {input.script} {input.sbml}"


rule transform_vrede2002:
    input:
        script="scripts/transform_vrede2002.py",
        table="data/vrede2002_table3.csv",
    output:
        "data/vrede2002_empirical.csv",
    shell:
        "python {input.script}"


rule transform_makino2003:
    input:
        script="scripts/transform_makino2003.py",
        table="data/makino2003_coli.csv",
    output:
        "data/makino2003_empirical.csv",
    shell:
        "python {input.script}"


rule sample_elemental_stoich:
    input:
        script="scripts/sample_elemental_stoich.py",
        package=[
            "metabolism_low_dim/__init__.py",
            "metabolism_low_dim/constants.py",
            "metabolism_low_dim/data_io.py",
            "metabolism_low_dim/model.py",
        ],
        mass_fraction_ranges="data/mass_fraction_ranges.json",
        element_ranges="data/element_ranges.json",
        aa_residue_element_counts="data/aa_residue_element_counts.json",
        aa_observed_mean="data/moura2013_aa_frequencies.json",
        aa_frequencies_by_genome="data/moura2013_aa_frequencies_by_genome.csv",
        na_residue_element_counts="data/na_residue_element_counts.json",
        na_gc_content="data/na_gc_content.csv",
        na_pool_mix_mean="data/na_pool_mix_mean.csv",
    output:
        CONSTRAINED_ELEMENTAL_STOICH_SAMPLES,
    shell:
        "python {input.script} --n-samples 1000000 --samples-csv " + CONSTRAINED_ELEMENTAL_STOICH_SAMPLES


rule plot_elemental_stoich:
    input:
        script="scripts/plot_elemental_stoich.py",
        package=[
            "metabolism_low_dim/__init__.py",
            "metabolism_low_dim/constants.py",
            "metabolism_low_dim/data_io.py",
            "metabolism_low_dim/plot_utils.py",
            "metabolism_low_dim/plotting.mplstyle",
        ],
        samples_csv=CONSTRAINED_ELEMENTAL_STOICH_SAMPLES,
        martiny="data/martiny_table_s2_latitude_metadata.csv",
        vrede_empirical="data/vrede2002_empirical.csv",
        makino_empirical="data/makino2003_empirical.csv",
    output:
        CONSTRAINED_ELEMENTAL_STOICH_PLOT,
    shell:
        "python {input.script} "
        "--samples-csv {input.samples_csv} "
        "--plot-path " + CONSTRAINED_ELEMENTAL_STOICH_PLOT + " "
        "--empirical-csv {input.vrede_empirical} --empirical-label 'Vrede 2002' "
        "--empirical-csv {input.makino_empirical} --empirical-label 'Makino 2003'"
