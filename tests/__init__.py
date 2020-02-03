from pathlib import Path

from phenotrex.transforms import fastas_to_grs
try:
    fastas_to_grs()
except ImportError:
    FROM_FASTA = False
except TypeError:
    FROM_FASTA = True

DATA_PATH = (Path(__file__).parent/'test_data')
