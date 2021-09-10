import click

from phenotrex.cli.generic_opt import common_deepnog_options


@click.command(context_settings=dict(help_option_names=["-h", "--help"]),
               short_help="Prediction of phenotypes with classifier")
@click.argument('fasta_files', type=click.Path(exists=True), nargs=-1)
@click.option('--genotype', type=click.Path(exists=True),
              required=False, help='Input genotype file.')
@click.option('--classifier', required=True, type=click.Path(exists=True),
              help='Path of pickled classifier file.')
@click.option('--min_proba', type=click.FloatRange(0.5, 1.0), default=0.5,
              help='Class probability threshold for displaying predictions. '
                   'Predictions below the threshold will be given as "N/A".')
@click.option('--out_explain_per_sample', type=click.Path(dir_okay=False),
              help='Write SHAP explanations for each predicted sample to file (optional).')
@click.option('--out_explain_summary', type=click.Path(dir_okay=False),
              help='Write SHAP explanations summarized over all samples (optional).')
@click.option('--n_max_explained_features', type=int, default=50,
              help='Limit output number of features in SHAP explanation files. '
                   'Also used to constrain the complexity of SHAP computations with TrexSVMs.')
@click.option('--shap_n_samples', type=str, default=16000,
              help='The nsamples parameter of SHAP. Only used by models '
                   'which utilize a `shap.KernelExplainer` (e.g. TrexSVM).')
@click.option('--verb', is_flag=True)
@common_deepnog_options
def predict(*args, **kwargs):
    """
    Predict phenotype from a set of (possibly gzipped) DNA or protein FASTA files
    or a single genotype file.
    NB: Genotype computation is highly expensive and performed on the fly on FASTA files.
    For increased speed when predicting multiple phenotypes, create a .genotype file to reuse
    with the command `compute-genotype`.
    """
    from phenotrex.ml.prediction import predict as _predict
    _predict(*args, **kwargs)
