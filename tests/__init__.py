from pathlib import Path

from phenotrex.transforms import fastas_to_grs
try:
    fastas_to_grs()
except ImportError:
    FROM_FASTA = False
except TypeError:
    FROM_FASTA = True

DATA_PATH = (Path(__file__).parent/'test_data')
GENOMIC_PATH = DATA_PATH/'genomic'
MODELS_PATH = DATA_PATH/'models'
FLAT_PATH = DATA_PATH/'flat'
