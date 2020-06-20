from pathlib import Path

import pytest
from tempfile import TemporaryDirectory
from phenotrex.cli.generic_func import generic_compute_shaps
from phenotrex.io.flat import load_training_files
from phenotrex.util.plotting import compleconta_plot, shap_summary_plot, shap_force_plots
from phenotrex.ml.clf.xgbm import TrexXGB
from phenotrex.io.serialization import save_classifier, load_classifier
from phenotrex.util.external_data import Eggnog5TextAnnotator

from .targets import cccv_scores_trex
from . import FLAT_PATH, MODELS_PATH, GENOMIC_PATH, FROM_FASTA


trait_name = 'T3SS_trunc'
fasta_files = []


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

    @pytest.mark.skipif(not FROM_FASTA, reason='Missing optional dependencies')
    def test_shap_force_plots(self):
        with TemporaryDirectory() as tmpdir:
            sh, gr = generic_compute_shaps(
                classifier=MODELS_PATH/trait_name/f'{trait_name}.xgb.pkl',
                fasta_files=list(GENOMIC_PATH.iterdir()),
                n_samples=50,
                verb=True,
                genotype=None
            )
            shap_force_plots(gr=gr, sh=sh, n_max_features=20, out_prefix=Path(tmpdir)/'force_plots')
            print(list(Path(tmpdir).iterdir()))
            assert len(list(Path(tmpdir).glob('force_plots*'))) > 0

    @pytest.mark.skipif(not FROM_FASTA, reason='Missing optional dependencies')
    def test_shap_summary_plot(self):
        with TemporaryDirectory() as tmpdir:
            sh, gr = generic_compute_shaps(
                classifier=MODELS_PATH/trait_name/f'{trait_name}.xgb.pkl',
                fasta_files=list(GENOMIC_PATH.iterdir()),
                n_samples=50,
                verb=True,
                genotype=None
            )
            shap_summary_plot(
                sh=sh,
                n_max_features=20,
                out_summary_plot=Path(tmpdir)/'plot.png',
                out_summary_txt=Path(tmpdir)/'summary.txt',
                title=''
            )
            assert (Path(tmpdir)/'plot.png').is_file()
            assert (Path(tmpdir)/'summary.txt').is_file()

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
