import click


def universal_options(f):
    """Options required by every command."""
    f = click.option('--genotype', type=click.Path(exists=True),
                     required=True, help='Genotype file path.')(f)
    f = click.option('--verb', is_flag=True)(f)
    return f


def common_train_options(f):
    """Train-specific CLI options."""
    f = click.option('--out', type=click.Path(), required=True, help='Output file path.')(f)
    f = click.option('--weights', is_flag=True, help='Save weights file for trained classifier.')(f)
    f = click.option('--phenotype', type=click.Path(exists=True),
                     required=True, help='Phenotype file path.')(f)
    return f


def common_cv_options(f):
    """CV-specific CLI options."""
    f = click.option('--folds', type=int, default=5, help='Number of folds in CV.')(f)
    f = click.option('--rank', type=str, default='family',
                     help='The taxonomic rank to use for LOGO CV (if groups file provided).')(f)
    f = click.option('--replicates', type=int, default=10, help='Number of replicates for CV.')(f)
    f = click.option('--threads', type=int, default=1, help='Number of threads to use.')(f)
    f = click.option('--out', type=click.Path(), help='Output file path.')(f)
    f = click.option('--groups', type=click.Path(exists=True),
                     help='Split into folds using the groups file.')(f)
    f = click.option('--phenotype', type=click.Path(exists=True),
                     required=True, help='Phenotype file path.')(f)
    return f


def common_cccv_options(f):
    """CCCV-specific CLI options."""
    f = click.option('--conta-steps', type=int, default=20,
                     help='Number of equidistant contamination levels to resample to.')(f)
    f = click.option('--comple-steps', type=int, default=20,
                     help='Number of equidistant completeness levels to resample to.')(f)
    return f
