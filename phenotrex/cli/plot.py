from pathlib import Path
from functools import partial

import click

from phenotrex.io.flat import load_cccv_accuracy_file
from phenotrex.util.plotting import compleconta_plot

click.option = partial(click.option, show_default=True)


@click.group(short_help='Plotting of results')
def plot():
    """
    Plot results.
    """
    pass


@plot.command()
@click.option('--inputs', type=click.Path(exists=True), required=True, nargs=0,
              help='CCCV output file(s) to plot.')
@click.argument('inputs', nargs=-1)
@click.option('--out', type=click.Path(), help='Output file path. If not given, `plt.show()` result.')
@click.option('--title', type=str, default='', help='Plot title.')
def cccv(inputs, out, title):
    """Plot CCCV result(s)."""
    conditions = [Path(str(x)).stem for x in inputs]
    cccv_results = [load_cccv_accuracy_file(x) for x in inputs]
    compleconta_plot(cccv_results=cccv_results, conditions=conditions, title=title, save_path=out)
