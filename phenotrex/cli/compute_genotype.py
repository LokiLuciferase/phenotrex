import click

from phenotrex.cli.generic_opt import common_deepnog_options


@click.command(context_settings=dict(help_option_names=["-h", "--help"]),
               short_help="Transform FASTA files into a genotype file.")
@click.argument('input', required=True, type=click.Path(exists=True), nargs=-1)
@click.option('--out', type=click.Path(exists=False),
              required=True, help='Path of output genotype file.')
@click.option('--threads', type=int, required=False,
              help='Number of parallel threads (default is the number available cores)')
@click.option('--verb', is_flag=True)
@common_deepnog_options
def compute_genotype(input, out, deepnog_threshold, threads=None, verb=True):
    """
    Create a genotype file suitable for learning and inference with `phenotrex`.
    Given a set of (possibly gzipped) DNA or protein FASTA files,
    perform annotation of eggNOG5-tax-2 (bacterial eggNOG5) clusters, and write to a .genotype file.
    """
    from phenotrex.io.flat import write_genotype_file
    from phenotrex.transforms import fastas_to_grs

    write_genotype_file(
        genotypes=fastas_to_grs(
            input, confidence_threshold=deepnog_threshold, verb=verb, n_threads=threads
        ), output_file=out
    )
