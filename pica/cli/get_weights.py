import click

from pica.io.flat import write_weights_file
from pica.io.serialization import load_classifier

@click.command()
@click.option('--classifier', type=click.Path(exists=True),
              required=True, help='Pickled classifier file.')
@click.option('--out', type=click.Path(), required=True, help='Output file path.')
def get_weights(classifier, out):
    """
    Write the feature weights of a classifier to a flat file.
    """
    clf = load_classifier(filename=classifier, verb=True)
    weights = clf.get_feature_weights()
    write_weights_file(weights_file=out, weights=weights)
