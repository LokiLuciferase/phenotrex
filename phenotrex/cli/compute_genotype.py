import click

from phenotrex.io.flat import write_genotype_file
from phenotrex.transforms import fastas_to_grs


@click.command(context_settings=dict(help_option_names=["-h", "--help"]),
               short_help="Transform FASTA files into a genotype file.")
@click.argument('input', required=True, type=click.Path(exists=True), nargs=-1)
@click.option('--out', type=click.Path(exists=False),
              required=True, help='Path of output genotype file.')
@click.option('--n_threads', type=int, required=False,
              help='Number of threads to use. Default, utilize at most all cores.')
@click.option('--verb', is_flag=True)
def compute_genotype(input, out, n_threads=None, verb=True):
    """
    Given a set of FASTA files, perform protein calling (for DNA FASTA files) and annotation
    of EggNOG5 clusters, and write to a .genotype file.
    """
    write_genotype_file(genotypes=fastas_to_grs(input, verb=verb, n_threads=n_threads), output_file=out)
