import pytest

from phenotrex.transforms.resampling import TrainingRecordResampler
from phenotrex.transforms.annotation import fasta_to_gr
from phenotrex.io.flat import load_training_files
from phenotrex.structure.records import GenotypeRecord

from . import DATA_PATH

predict_files = [
    DATA_PATH/'GCA_000692775_1_trunc2.fna',
    DATA_PATH/'GCA_000692775_1_trunc2.faa',
]


def test_resampling():
    td, *_ = load_training_files(DATA_PATH/'Sulfate_reducer.genotype',
                                 DATA_PATH/'Sulfate_reducer.phenotype')

    trr = TrainingRecordResampler(random_state=2, verb=True)
    trr.fit(td)
    trr.get_resampled(td[0], comple=.5, conta=.5)


@pytest.mark.parametrize('infile', predict_files, ids=['fna', 'faa'])
def test_compute_genotype(infile):
    gr = fasta_to_gr(infile, verb=False)
    assert isinstance(gr, GenotypeRecord)
    assert len(gr.features) > 0
