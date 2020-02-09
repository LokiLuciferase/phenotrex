#
# Created by Lukas Lüftinger on 05/02/2019.
#
try:
    from .annotation import fastas_to_grs
except ModuleNotFoundError:
    from phenotrex.util.helpers import fail_missing_dependency as fastas_to_grs
__all__ = ['fastas_to_grs']
