from functools import partial

import click

from pica.cli.generic_opt import universal_options, common_cv_options, common_cccv_options, param_options
from pica.cli.generic_func import generic_cccv
from pica.cli.clf_opt import xgb_options, svm_options

click.option = partial(click.option, show_default=True)


@click.group()
def cccv():
    """
    Perform nested, stratified crossvalidation over a range of simulated completeness
    and contamination values for the training data.
    Optionally, save returned scores for each grid point.
    """
    pass


@cccv.command('xgb')
@universal_options
@common_cv_options
@param_options
@common_cccv_options
@xgb_options
def xgb(*args, **kwargs):
    """Perform Completeness/Contamination CV on XGB model."""
    generic_cccv('xgb', *args, **kwargs)


@cccv.command('svm')
@universal_options
@common_cv_options
@param_options
@common_cccv_options
@svm_options
def svm(*args, **kwargs):
    """Perform Completeness/Contamination CV on SVM model."""
    generic_cccv('svm', *args, **kwargs)
