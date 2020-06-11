from pathlib import Path

from tempfile import TemporaryDirectory
from phenotrex.io.flat import load_training_files
from phenotrex.util.plotting import compleconta_plot
from phenotrex.ml.clf.xgbm import TrexXGB
from phenotrex.io.serialization import save_classifier, load_classifier
from phenotrex.util.external_data import Eggnog5TextAnnotator

from .targets import cccv_scores_trex
from . import FLAT_PATH, MODELS_PATH


trait_name = 'T3SS_trunc'


class TestUtil:
    def get_training_data(self):
        td, *_ = load_training_files(
            FLAT_PATH/trait_name/f'{trait_name}.genotype',
            FLAT_PATH/trait_name/f'{trait_name}.phenotype'
        )
        return td

    def test_cc_plot(self):
        with TemporaryDirectory() as tmpdir:
            plot = Path(tmpdir)/'plot.png'
            compleconta_plot(
                list(cccv_scores_trex['SVM'].values()),
                conditions=list(cccv_scores_trex['SVM'].keys()),
                save_path=str(plot)
            )
            assert plot.is_file()

    def test_ml_dump(self):
        with TemporaryDirectory() as tmpdir:
            mp = Path(tmpdir)/f'{trait_name}.pkl'
            td = self.get_training_data()
            xgb = TrexXGB(random_state=2)
            xgb.train(td)
            save_classifier(xgb, mp)
            assert mp.is_file()

    def test_ml_load(self):
        td = self.get_training_data()
        xgb = load_classifier(MODELS_PATH/trait_name/f'{trait_name}.xgb.pkl')
        preds = xgb.predict(td)
        assert preds is not None

    def test_download_eggnog5_annot(self):
        assert Eggnog5TextAnnotator().annotate(2, 'COG3520')
