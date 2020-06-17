from pathlib import Path
from functools import partial

import click


click.option = partial(click.option, show_default=True)


@click.group(short_help='Plotting of results')
def plot():
    """
    Plot results.
    """
    pass


@plot.command()
@click.option('--inputs', type=click.Path(exists=True), required=True, nargs=0,
              help='CCCV output file(s) to plot.')
@click.argument('inputs', nargs=-1)
@click.option('--out', type=click.Path(), help='Output file path. If not given, `.show()` result.')
@click.option('--title', type=str, default='', help='Plot title.')
def cccv(inputs, out, title):
    """Plot CCCV result(s)."""
    from phenotrex.io.flat import load_cccv_accuracy_file
    from phenotrex.util.plotting import compleconta_plot

    conditions = [Path(str(x)).stem for x in inputs]
    cccv_results = [load_cccv_accuracy_file(x) for x in inputs]
    compleconta_plot(cccv_results=cccv_results, conditions=conditions, title=title, save_path=out)


@plot.command('shap-summary', short_help='Plot summary of SHAP feature contributions.')
@click.argument('fasta_files', type=click.Path(exists=True), nargs=-1)
@click.option('--genotype', type=click.Path(exists=True),
              required=False, help='Input genotype file.')
@click.option('--classifier', required=True, type=click.Path(exists=True),
              help='Path of pickled classifier file.')
@click.option('--out_plot', required=True, type=click.Path(dir_okay=False),
              help='The file to save the generated plot at.')
@click.option('--out_summary', default=None, type=click.Path(dir_okay=False),
              help='The file to save the generated summary at.')
@click.option('--n_max_features', type=int, default=20,
              help='The number of top most important features (by absolute SHAP value) to plot.')
@click.option('--n_samples', default='auto',
              help='The nsamples parameter of SHAP. '
                   'Only used by models which utilize a `shap.KernelExplainer` (e.g. TrexSVM).')
@click.option('--title', type=str, default='', help='Plot title.')
@click.option('--verb', is_flag=True)
def shap_summary(out_plot, out_summary, n_max_features, title, **kwargs):
    import matplotlib as mpl
    mpl.use('Agg')
    from .generic_func import generic_compute_shaps
    from phenotrex.util.plotting import shap_summary_plot

    sh, gr = generic_compute_shaps(**kwargs)
    shap_summary_plot(
        sh=sh,
        n_max_features=n_max_features,
        out_summary_plot=out_plot,
        out_summary_txt=out_summary,
        title=title
    )


@plot.command('shap-force', short_help='Plot SHAP feature contributions per sample.')
@click.argument('fasta_files', type=click.Path(exists=True), nargs=-1)
@click.option('--genotype', type=click.Path(exists=True),
              required=False, help='Input genotype file.')
@click.option('--classifier', required=True, type=click.Path(exists=True),
              help='Path of pickled classifier file.')
@click.option('--out_prefix', required=True, type=str,
              help='The prefix to generated SHAP force plots.')
@click.option('--out_summary', default=None, type=click.Path(dir_okay=False),
              help='The path to save the generated summary at.')
@click.option('--n_max_features', type=int, default=20,
              help='The number of top most important features (by absolute SHAP value) to plot.')
@click.option('--n_samples', default='auto',
              help='The nsamples parameter of SHAP. '
                   'Only used by models which utilize a `shap.KernelExplainer` (e.g. TrexSVM).')
@click.option('--verb', is_flag=True)
def shap_force(out_prefix, out_summary, n_max_features, **kwargs):
    """
    Generate SHAP force plots for each sample (passed either as FASTA files or as genotype file).
    All plots will be saved at the path `{out_prefix}_{sample_identifier}_force_plot.png`.
    All non-existent directories in the out_prefix path will be created as needed.
    """
    import matplotlib as mpl
    mpl.use('Agg')

    from .generic_func import generic_compute_shaps
    from phenotrex.util.plotting import shap_force_plots

    sh, gr = generic_compute_shaps(**kwargs)
    shap_force_plots(
        gr=gr,
        sh=sh,
        out_prefix=out_prefix,
        n_max_features=n_max_features,
        out_individual_summary=out_summary
    )


@plot.command('shap-full', short_help='Create all SHAP plots and summaries.')
@click.argument('fasta_files', type=click.Path(exists=True), nargs=-1)
@click.option('--genotype', type=click.Path(exists=True),
              required=False, help='Input genotype file.')
@click.option('--classifier', required=True, type=click.Path(exists=True),
              help='Path of pickled classifier file.')
@click.option('--force_plot_prefix', type=str, default='force_plots/',
              help='The prefix to generated per-sample SHAP force plots.')
@click.option('--out_force_file', default='per_sample_forces.tsv', type=click.Path(dir_okay=False),
              help='The path to save the generated per-sample SHAP force file.')
@click.option('--out_summary_plot', default='shap_summary.png', type=click.Path(dir_okay=False),
              help='The path to save the generated summary plot at.')
@click.option('--out_summary_file', default='shap_summary.tsv', type=click.Path(dir_okay=False),
              help='The path to save the generated summary file at.')
@click.option('--summary_plot_title', type=str, default='SHAP Summary')
@click.option('--n_max_features', type=int, default=30,
              help='The number of top most important features (by absolute SHAP value) to plot.')
@click.option('--n_samples', default='auto',
              help='The nsamples parameter of SHAP. '
                   'Only used by models which utilize a `shap.KernelExplainer` (e.g. TrexSVM).')
@click.option('--verb', is_flag=True)
def shap_full(
    force_plot_prefix,
    out_force_file,
    out_summary_plot,
    out_summary_file,
    n_max_features,
    summary_plot_title,
    **kwargs
):
    """
    Generate SHAP force plots for each sample (passed either as FASTA files or as genotype file) as
    well as summary plots averaged over all samples.
    """
    import matplotlib as mpl
    mpl.use('Agg')
    from .generic_func import generic_compute_shaps
    from phenotrex.util.plotting import shap_force_plots, shap_summary_plot

    sh, gr = generic_compute_shaps(**kwargs)
    shap_force_plots(
        gr=gr,
        sh=sh,
        out_prefix=force_plot_prefix,
        n_max_features=n_max_features,
        out_individual_summary=out_force_file
    )
    shap_summary_plot(
        sh=sh,
        n_max_features=n_max_features,
        out_summary_plot=out_summary_plot,
        out_summary_txt=out_summary_file,
        title=summary_plot_title
    )
