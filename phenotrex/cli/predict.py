import click

from phenotrex.ml.prediction import predict as _predict


@click.command(context_settings=dict(help_option_names=["-h", "--help"]),
               short_help="Prediction of phenotypes with classifier")
@click.argument('fasta_files', type=click.Path(exists=True), nargs=-1)
@click.option('--genotype', type=click.Path(exists=True),
              required=False, help='Input genotype file.')
@click.option('--classifier', required=True, type=click.Path(exists=True),
              help='Path of pickled classifier file.')
@click.option('--verb', is_flag=True)
def predict(*args, **kwargs):
    """
    Predict phenotype from a set of (possibly gzipped) DNA or protein FASTA files
    or a single genotype file.
    NB: Genotype computation is highly expensive and performed on the fly on FASTA files.
    For increased speed when predicting multiple phenotypes, create a .genotype file to reuse
    with the command `compute-genotype`.
    """
    _predict(*args, **kwargs)
