from functools import partial

import click

from pica.cli.generic_opt import universal_options, common_train_options, common_cv_options
from pica.cli.generic_func import generic_train, generic_cv, generic_cccv, generic_logocv

click.option = partial(click.option, show_default=True)

def xgb_options(f):
    """XGB-specific CLI options."""
    f = click.option('--max_depth', type=int, default=4)(f)
    # TODO: add more settable options, or allow arbitrary ones
    return f

@click.group()
def xgb():
    """Perform training and evaluation functions with XGB classifier."""
    pass

@xgb.command()
@universal_options
@common_train_options
@xgb_options
def train(*args, **kwargs):
    """Train XGB model."""
    generic_train('xgb', *args, **kwargs)

@xgb.command()
@universal_options
@common_cv_options
@xgb_options
def cv(*args, **kwargs):
    """Perform CV on XGB model."""
    generic_cv('xgb', *args, **kwargs)

@xgb.command()
@universal_options
@common_cv_options
@xgb_options
def cccv(*args, **kwargs):
    """Perform CCCV on XGB model."""
    generic_cccv('xgb', *args, **kwargs)

@xgb.command()
@universal_options
@common_cv_options
@xgb_options
def logocv(*args, **kwargs):
    """Perform LOGO CV on XGB model."""
    generic_logocv('xgb', *args, **kwargs)
