from pathlib import Path
from functools import partial

import click

from pica.cli.generic_opt import universal_options, common_cv_options, param_options
from pica.cli.generic_func import generic_cv
from pica.cli.clf_opt import xgb_options, svm_options

click.option = partial(click.option, show_default=True)

@click.group()
def cv():
    """
    Perform nested, stratified crossvalidation. print obtained score.
    Optionally, save misclassifications file.
    """
    pass

def cv_options(f):
    """Options specific for CV (not used in CCCV)."""
    f = click.option('--optimize_out', type=click.Path(), default=Path('./params.json'),
                     help='The file path at which to save optimized parameters.')(f)
    f = click.option('--optimize', is_flag=True,
                     help='Whether to perform parameter search using default search parameter range.')(f)
    f = click.option('--rank', type=str, default='family',
                     help='The taxonomic rank to use for LOGO CV (if groups file provided).')(f)
    f = click.option('--groups', type=click.Path(exists=True),
                     help='Split into folds using the groups file.')(f)
    return f

@cv.command('xgb')
@universal_options
@common_cv_options
@cv_options
@param_options
@xgb_options
def xgb(*args, **kwargs):
    """Perform CV on XGB model."""
    generic_cv('xgb', *args, **kwargs)

@cv.command('svm')
@universal_options
@common_cv_options
@cv_options
@param_options
@svm_options
def svm(*args, **kwargs):
    """Perform CV on SVM model."""
    generic_cv('svm', *args, **kwargs)
