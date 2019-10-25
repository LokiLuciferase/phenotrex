from functools import partial

import click

from pica.cli.generic_opt import (universal_options, common_train_options, common_cv_options,
                                  common_cccv_options)
from pica.cli.generic_func import generic_train, generic_cv, generic_cccv, generic_logocv

click.option = partial(click.option, show_default=True)

def svm_options(f):
    """SVM-specific CLI options."""
    f = click.option('--c', type=float, default=5.)(f)
    f = click.option('--penalty', type=click.Choice(['l1', 'l2']), default='l2')(f)
    f = click.option('--tol', type=float, default=1.)(f)
    f = click.option('--n_features', default=None, type=int,
                     help='Number of features aimed at by RFECV. If None, do not perform RFECV.')(f)
    # TODO: add more settable options, or allow arbitrary ones
    return f


@click.group()
def svm():
    """Perform training and evaluation functions with SVM classifier."""
    pass


@svm.command()
@universal_options
@common_train_options
@svm_options
def train(*args, **kwargs):
    """Perform training on SVM model."""
    generic_train('svm', *args, **kwargs)


@svm.command()
@universal_options
@common_cv_options
@svm_options
def cv(*args, **kwargs):
    """Perform CV on SVM model."""
    generic_cv('svm', *args, **kwargs)


@svm.command()
@universal_options
@common_cv_options
@common_cccv_options
@svm_options
def cccv(*args, **kwargs):
    """Perform CCCV on SVM model."""
    generic_cccv('svm', *args, **kwargs)


@svm.command()
@universal_options
@common_cv_options
@svm_options
def logocv(*args, **kwargs):
    """Perform LOGO CV on SVM model."""
    pass
