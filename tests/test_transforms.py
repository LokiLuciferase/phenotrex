from phenotrex.transforms.resampling import TrainingRecordResampler
from phenotrex.io.flat import load_training_files

from . import DATA_PATH


def test_resampling():
    td, *_ = load_training_files(DATA_PATH/'Sulfate_reducer.genotype',
                                 DATA_PATH/'Sulfate_reducer.phenotype')

    trr = TrainingRecordResampler(random_state=2, verb=True)
    trr.fit(td)
    trr.get_resampled(td[0], comple=.5, conta=.5)
