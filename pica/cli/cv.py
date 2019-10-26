from functools import partial

import click

from pica.cli.generic_opt import universal_options, common_cv_options
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

@cv.command('xgb')
@universal_options
@common_cv_options
@xgb_options
def xgb(*args, **kwargs):
    """Perform CV on XGB model."""
    generic_cv('xgb', *args, **kwargs)

@cv.command('svm')
@universal_options
@common_cv_options
@svm_options
def svm(*args, **kwargs):
    """Perform CV on SVM model."""
    generic_cv('svm', *args, **kwargs)
