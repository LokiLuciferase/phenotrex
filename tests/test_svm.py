#
# Created by Lukas Lüftinger on 2/17/19.
#
import pytest

from tests.targets import first_genotype_accession, first_phenotype_accession, cv_scores, cccv_scores
from tests.targets import num_of_features_compressed, num_of_features_uncompressed, num_of_features_selected
from pica.io.io import load_training_files
from pica.ml.svm import PICASVM
from pica.util.helpers import get_x_y_tn

import numpy as np

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

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    def test_compress_vocabulary(self, trait_name):
        """
        Perform feature compression tests
        :param trait_name:
        :return:
        """
        tr, gr, pr = self.test_load_training_files(trait_name)
        svm = PICASVM(verb=True, random_state=RANDOM_STATE)
        svm.compress_vocabulary(records=tr)
        vec = svm.cv_pipeline.named_steps["vec"]
        vec._validate_vocabulary()

        # check if vocabulary is set properly
        assert vec.fixed_vocabulary_

        # check if length of vocabulary is matching
        assert len(vec.vocabulary_) == num_of_features_uncompressed[trait_name]

        X, y, tn = get_x_y_tn(tr)
        X_trans = vec.transform(X)

        # check if number of unique features is matching
        assert X_trans.shape[1] == num_of_features_compressed[trait_name]

        # check if all samples still have at least one feature present
        one_is_zero = False
        non_zero = X_trans.nonzero()
        for x in non_zero:
            if len(x) == 0:
                one_is_zero = True

        assert not one_is_zero

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    def test_recursive_feature_elimination(self, trait_name):
        """
        Perform feature compression tests
        :param trait_name:
        :return:
        """
        tr, gr, pr = self.test_load_training_files(trait_name)
        svm = PICASVM(verb=True, random_state=RANDOM_STATE)
        svm.recursive_feature_elimination(records=tr, n_steps=5, n_features=None)
        vec = svm.cv_pipeline.named_steps["vec"]
        vec._validate_vocabulary()

        # check if vocabulary is set properly
        assert vec.fixed_vocabulary_

        # check if length of vocabulary is matching
        assert len(vec.vocabulary_) == num_of_features_selected[trait_name]

        X, y, tn = get_x_y_tn(tr)
        X_trans = vec.transform(X)

        # check if number of unique features is matching
        assert X_trans.shape[1] == num_of_features_selected[trait_name]

        # check if all samples still have at least one feature present
        one_is_zero = False
        non_zero = X_trans.nonzero()
        for x in non_zero:
            if len(x) == 0:
                one_is_zero = True

        assert not one_is_zero
