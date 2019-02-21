#
# Created by Lukas Lüftinger on 2/17/19.
#
import pytest

from tests.targets import first_genotype_accession, first_phenotype_accession, cv_scores, cccv_scores
from pica.io.io import load_training_files
from pica.ml.svm import PICASVM

RANDOM_STATE = 2

trait_names = [
    "Sulfate_reducer",
    #"Aerobe",
    #"sporulation",
]

cv_folds = [#3,
            5,
            #10,
            ]

scoring_methods = [#"accuracy",
                   "balanced_accuracy",
                   #"f1"
                   ]



class TestPICASVM:
    @pytest.mark.parametrize("trait_name",
                             [pytest.param("Sulfate_reducer", id="Sulfate_reducer",),
                              pytest.param("Aerobe", id="Aerobe", marks=[pytest.mark.xfail])])  # file not found
    def test_load_training_files(self, trait_name):
        """
        Test training data loading. Check/catch invalid file formats.
        :param trait_name:
        :return:
        """
        full_path_genotype = f"tests/test_svm/{trait_name}.genotype"
        full_path_phenotype = f"tests/test_svm/{trait_name}.phenotype"
        tr, gr, pr = load_training_files(
            genotype_file=full_path_genotype,
            phenotype_file=full_path_phenotype,
            verb=True)
        assert gr[0].identifier == first_genotype_accession[trait_name]
        assert pr[0].identifier == first_phenotype_accession[trait_name]
        return tr, gr, pr

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    def test_train(self, trait_name):
        """
        Test PICASVM training. Using different traits.
        :param trait_name:
        :return:
        """
        tr, gr, pr = self.test_load_training_files(trait_name)
        svm = PICASVM(verb=True, random_state=RANDOM_STATE)
        trained_svm = svm.train(records=tr)

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("cv", cv_folds, ids=[str(x) for x in cv_folds])
    @pytest.mark.parametrize("scoring", scoring_methods, ids=scoring_methods)
    def test_crossvalidate(self, trait_name, cv, scoring):
        """
        Test default crossvalidation of PICASVM class. Using several different traits, cv folds, and scoring methods.
        Compares with dictionary cv_scores.
        :param trait_name:
        :param cv:
        :param scoring:
        :return:
        """
        tr, gr, pr = self.test_load_training_files(trait_name)
        svm = PICASVM(verb=True, random_state=RANDOM_STATE)
        assert cv_scores[trait_name][cv][scoring] == svm.crossvalidate(records=tr, cv=cv, scoring=scoring)[:2]

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    def test_compleconta_cv(self, trait_name):
        """
        Perform compleconta-cv for each trait name using PICASVM class.
        :param trait_name:
        :return:
        """
        tr, gr, pr = self.test_load_training_files(trait_name)
        svm = PICASVM(verb=True, random_state=RANDOM_STATE)
        assert cccv_scores[trait_name] == svm.crossvalidate_cc(records=tr, cv=5, comple_steps=3, conta_steps=3)
