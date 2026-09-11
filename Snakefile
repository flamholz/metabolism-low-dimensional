"""Generate this repo's figures.

Run everything with `snakemake --cores 1` from the repo root (every script
here resolves its own paths relative to the repo root or its own file
location, not the caller's cwd, so no `workdir:` directive is needed).
"""

PHASE_DIAGRAM_OUTDIR = "figures/phase_diagram"

rule all:
    input:
        "figures/elemental_FBA_panels.png",
        "figures/elemental_FBA_panels.svg",
        f"{PHASE_DIAGRAM_OUTDIR}/phase_diagram.png",
        f"{PHASE_DIAGRAM_OUTDIR}/phase_diagram_samples.csv",


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


rule phase_diagram:
    input:
        script="scripts/constrained_elemental_stoich.py",
        package=[
            "metabolism_low_dim/__init__.py",
            "metabolism_low_dim/constants.py",
            "metabolism_low_dim/data_io.py",
            "metabolism_low_dim/model.py",
            "metabolism_low_dim/plot_utils.py",
            "metabolism_low_dim/plotting.py",
            "metabolism_low_dim/plotting.mplstyle",
        ],
        mass_fraction_ranges="data/mass_fraction_ranges.csv",
        element_ranges="data/element_ranges.csv",
        aa_residue_element_counts="data/aa_residue_element_counts.csv",
        aa_observed_mean="data/aa_observed_mean.csv",
        na_residue_element_counts="data/na_residue_element_counts.csv",
        na_gc_content="data/na_gc_content.csv",
        na_pool_mix_mean="data/na_pool_mix_mean.csv",
        martiny="data/martiny_table_s2_latitude_metadata.csv",
        vrede_empirical="data/vrede2002_empirical.csv",
        makino_empirical="data/makino2003_empirical.csv",
    output:
        f"{PHASE_DIAGRAM_OUTDIR}/phase_diagram.png",
        f"{PHASE_DIAGRAM_OUTDIR}/phase_diagram_samples.csv",
    shell:
        "python {input.script} "
        "--outdir " + PHASE_DIAGRAM_OUTDIR + " "
        "--empirical-csv {input.vrede_empirical} --empirical-label 'Vrede 2002' "
        "--empirical-csv {input.makino_empirical} --empirical-label 'Makino 2003'"
