import click

from pica.cli.svm import svm
from pica.cli.xgb import xgb
from pica.cli.predict import predict

@click.group()
def cli():
    pass

cli.add_command(svm)
cli.add_command(xgb)
cli.add_command(predict)

if __name__ == '__main__':
    cli()
