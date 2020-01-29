import click
from tqdm.auto import tqdm

from phenotrex.io.flat import write_genotype_file
from phenotrex.transforms import fasta_to_gr


@click.command(context_settings=dict(help_option_names=["-h", "--help"]),
               short_help="Transform FASTA files into a genotype file.")
@click.argument('input', required=False, type=click.Path(exists=True), nargs=-1)
@click.option('--out', type=click.Path(exists=False),
              required=True, help='Path of output genotype file.')
# @click.option('--verb', is_flag=True)  # TODO: re-enable when deepnog tqdm prompt is transient
def compute_genotype(input, out, verb=False):
    """
    Given a set of FASTA files, perform protein calling (for DNA FASTA files) and annotation
    of EggNOG5 clusters, and write to a .genotype file.
    """
    it = tqdm(input, total=len(input), unit='file', desc='Computing genotypes') if len(input) > 1 else input
    annotated = [fasta_to_gr(f, verb=verb) for f in it]
    write_genotype_file(genotypes=annotated, output_file=out)
