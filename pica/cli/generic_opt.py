import click


def universal_options(f):
    f = click.option('--genotype', type=click.Path(), required=True)(f)
    f = click.option('--verb', is_flag=True)(f)
    return f


def common_train_options(f):
    """Train-specific CLI options."""
    f = click.option('--out', type=click.Path(), required=True)(f)
    f = click.option('--weights', is_flag=True)(f)
    f = click.option('--phenotype', type=click.Path(), required=True)(f)
    return f


def common_cv_options(f):
    """CV- and CCCV-spcific CLI options."""
    f = click.option('--folds', type=int, default=5, help='Number of folds in CV.')(f)
    f = click.option('--replicates', type=int, default=10, help='Number of replicates for CV.')(f)
    f = click.option('--threads', type=int, default=1, help='Number of threads to use.')(f)
    f = click.option('--out', type=click.Path())(f)
    f = click.option('--phenotype', type=click.Path(), required=True)(f)
    return f


def common_cccv_options(f):
    """CCCV-specific CLI options."""
    f = click.option('--conta-steps', type=int, default=20,
                     help='Number of equidistant contamination levels to resample to.')(f)
    f = click.option('--comple-steps', type=int, default=20,
                     help='Number of equidistant completeness levels to resample to.')(f)
    return f
