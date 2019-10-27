import click

from pica.cli.train import train
from pica.cli.cv import cv
from pica.cli.cccv import cccv
from pica.cli.predict import predict
from pica.cli.get_weights import get_weights
from pica.cli.plot import plot

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    pass

cli.add_command(train)
cli.add_command(cv)
cli.add_command(cccv)
cli.add_command(predict)
cli.add_command(get_weights)
cli.add_command(plot)

def main():
    print("""
    ██████╗ ██╗  ██╗███████╗███╗   ██╗ ██████╗       ████████╗██████╗ ███████╗██╗  ██╗
    ██╔══██╗██║  ██║██╔════╝████╗  ██║██╔═══██╗      ╚══██╔══╝██╔══██╗██╔════╝╚██╗██╔╝
    ██████╔╝███████║█████╗  ██╔██╗ ██║██║   ██║█████╗   ██║   ██████╔╝█████╗   ╚███╔╝
    ██╔═══╝ ██╔══██║██╔══╝  ██║╚██╗██║██║   ██║╚════╝   ██║   ██╔══██╗██╔══╝   ██╔██╗
    ██║     ██║  ██║███████╗██║ ╚████║╚██████╔╝         ██║   ██║  ██║███████╗██╔╝ ██╗
    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝          ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
        """)
    cli()

if __name__ == '__main__':
    main()
