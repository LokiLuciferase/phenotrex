import click

from pica.io.flat import load_genotype_file, DEFAULT_TRAIT_SIGN_MAPPING
from pica.io.serialization import load_classifier
from pica.cli.generic_opt import universal_options


@click.command(context_settings=dict(help_option_names=["-h", "--help"]),
               short_help="Prediction of phenotypes with classifier")
@click.option('--classifier', required=True, help='Path of pickled classifier file.')
@universal_options
def predict(genotype, classifier, **kwargs):
    """
    Predict phenotype from a genotype file.
    """
    gr = load_genotype_file(genotype)
    model = load_classifier(filename=classifier, **kwargs)
    preds, probas = model.predict(X=gr)
    translate_output = {trait_id: trait_sign for trait_sign, trait_id in
                        DEFAULT_TRAIT_SIGN_MAPPING.items()}
    print("Identifier\tTrait present\tConfidence")
    for record, result, probability in zip(gr, preds, probas):
        print(f"{record.identifier}\t{translate_output[result]}\t{probability[result]}")
