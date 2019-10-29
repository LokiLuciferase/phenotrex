from functools import partial

import click

from pica.cli.generic_opt import universal_options, common_train_options, param_options
from pica.cli.generic_func import generic_train
from pica.cli.clf_opt import xgb_options, svm_options

click.option = partial(click.option, show_default=True)


@click.group()
def train():
    """
    Perform training and evaluation functions.
    Requires a .genotype and a .phenotype file.
    """
    pass


@train.command()
@universal_options
@common_train_options
@param_options
@xgb_options
def xgb(*args, **kwargs):
    """Train XGB model."""
    generic_train('xgb', *args, **kwargs)


@train.command()
@universal_options
@common_train_options
@param_options
@svm_options
def svm(*args, **kwargs):
    """Train SVM model."""
    generic_train('svm', *args, **kwargs)
