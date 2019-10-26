from functools import partial

import click

click.option = partial(click.option, show_default=True)

def xgb_options(f):
    """XGB-specific CLI options."""
    f = click.option('--max_depth', type=int, default=4, help='Maximum tree depth.')(f)
    # TODO: add more settable options, or allow arbitrary ones
    return f

def svm_options(f):
    """SVM-specific CLI options."""
    f = click.option('--c', type=float, default=5., help='SVM parameter C.')(f)
    f = click.option('--penalty', type=click.Choice(['l1', 'l2']), default='l2',
                     help='Regularization strategy.')(f)
    f = click.option('--tol', type=float, default=1., help='Stopping tolerance.')(f)
    f = click.option('--n_features', default=None, type=int,
                     help='Number of features aimed at by RFECV. If None, do not perform RFECV.')(f)
    # TODO: add more settable options, or allow arbitrary ones
    return f
