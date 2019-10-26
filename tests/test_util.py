from pathlib import Path

from tempfile import TemporaryDirectory
from pica.io.flat import load_training_files
from pica.util.plotting import compleconta_plot
from pica.ml.classifiers import TrexXGB
from pica.io.serialization import save_classifier, load_classifier

from .targets import cccv_scores_trex
from . import DATA_PATH


class TestUtil:
    def get_training_data(self):
        td, *_ = load_training_files(DATA_PATH / 'Sulfate_reducer.genotype',
                                     DATA_PATH / 'Sulfate_reducer.phenotype')
        return td

    def test_cc_plot(self):
        with TemporaryDirectory() as tmpdir:
            plot = Path(tmpdir)/'plot.png'
            compleconta_plot(list(cccv_scores_trex['SVM'].values()),
                             conditions=list(cccv_scores_trex['SVM'].keys()),
                             save_path=str(plot))
            assert plot.is_file()

    def test_ml_dump(self):
        with TemporaryDirectory() as tmpdir:
            mp = Path(tmpdir)/'Sulfate_reducer_xgb.pkl'
            td = self.get_training_data()
            xgb = TrexXGB(random_state=2)
            xgb.train(td)
            save_classifier(xgb, mp)
            assert mp.is_file()

    def test_ml_load(self):
        td = self.get_training_data()
        xgb = load_classifier(DATA_PATH/'Sulfate_reducer_xgb.pkl')
        preds = xgb.predict(td)
        assert preds is not None
