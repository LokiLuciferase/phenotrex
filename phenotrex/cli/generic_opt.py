import click


def universal_options(f):
    """Options required by every command."""
    f = click.option('--genotype', type=click.Path(exists=True),
                     required=True, help='Genotype file path.')(f)
    f = click.option('--verb', is_flag=True)(f)
    return f


def param_options(f):
    """Options to supply parameter files for training and parameter search."""
    f = click.option('--params_file', type=click.Path(exists=True),
                     help='JSON file from which to load train parameters.')(f)
    return f


def common_train_options(f):
    """Train-specific CLI options."""
    f = click.option('--out', type=click.Path(), required=True,
                     help='Output file path of classifier.')(f)
    f = click.option('--weights', is_flag=True, help='Save weights file for trained classifier.')(f)
    f = click.option('--phenotype', type=click.Path(exists=True),
                     required=True, help='Phenotype file path.')(f)
    return f


def common_cv_options(f):
    """CV-specific CLI options."""
    f = click.option('--folds', type=int, default=5, help='Number of folds in CV.')(f)
    f = click.option('--replicates', type=int, default=10, help='Number of replicates for CV.')(f)
    f = click.option('--threads', type=int, default=1, help='Number of threads to use.')(f)
    f = click.option('--phenotype', type=click.Path(exists=True),
                     required=True, help='Phenotype file path.')(f)
    return f


def common_deepnog_options(f):
    """Options added to commands which make use of deepnog."""
    f = click.option(
        '--deepnog_threshold',
        type=click.FloatRange(min=0.0, max=1.0),
        default=None,
        help='The confidence cutoff for feature creation directly from FASTA files.'
             ' Only features identfied in the genome with a confidence greater than'
             ' this are considered.'
    )(f)
    return f
