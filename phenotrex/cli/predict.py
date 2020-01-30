import click

from phenotrex.io.flat import load_genotype_file, DEFAULT_TRAIT_SIGN_MAPPING
from phenotrex.io.serialization import load_classifier
from phenotrex.transforms.annotation import fastas_to_grs


@click.command(context_settings=dict(help_option_names=["-h", "--help"]),
               short_help="Prediction of phenotypes with classifier")
@click.argument('input', type=click.Path(exists=True), nargs=-1)
@click.option('--genotype', type=click.Path(exists=True),
              required=False, help='Input genotype file.')
@click.option('--classifier', required=True, type=click.Path(exists=True),
              help='Path of pickled classifier file.')
@click.option('--verb', is_flag=True)
def predict(input=tuple(), genotype=None, classifier=None, verb=None):
    """
    Predict phenotype from a set of FASTA files or a single genotype file.
    NB: Genotype computation is highly expensive and performed on the fly on FASTA files.
    For increased speed when predicting multiple phenotypes, create a .genotype file to reuse
    with the command `compute-genotype`.
    """
    if not len(input) and genotype is None:
        raise RuntimeError('Must either supply FASTA file(s) or single genotype file for prediction.')
    grs_from_fasta = fastas_to_grs(input, n_threads=None, verb=verb)
    grs_from_file = load_genotype_file(genotype) if genotype is not None else []
    gr = grs_from_fasta + grs_from_file

    model = load_classifier(filename=classifier, verb=verb)
    preds, probas = model.predict(X=gr)
    translate_output = {trait_id: trait_sign for trait_sign, trait_id in
                        DEFAULT_TRAIT_SIGN_MAPPING.items()}
    print(f"# Trait: {model.trait_name}")
    print("Identifier\tTrait present\tConfidence")
    for record, result, probability in zip(gr, preds, probas):
        print(f"{record.identifier}\t{translate_output[result]}\t{str(round(probability[result], 4))}")
