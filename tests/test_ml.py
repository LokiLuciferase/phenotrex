import pytest

import numpy as np

from tests.targets import first_genotype_accession, first_phenotype_accession, cv_scores_trex, \
    cccv_scores_trex
from tests.targets import num_of_features_compressed, num_of_features_uncompressed
from pica.io.io import load_training_files
from pica.ml.classifiers.svm import TrexSVM
from pica.ml.classifiers.xgbm import TrexXGB
from pica.util.helpers import get_x_y_tn

from pica.ml.feature_select import recursive_feature_elimination, compress_vocabulary

from . import DATA_PATH

RANDOM_STATE = 2

trait_names = [
    "Sulfate_reducer",
    # "Aerobe",
    # "sporulation",
]

classifiers = [
    TrexSVM,
    TrexXGB  # TODO: add data for TrexXGB
]
classifier_ids = [
    'SVM',
    'XGB'
]

cv_folds = [5, ]

# TODO: both in PICASVM as well as in TrexClassifier, scoring cannot be changed atm
scoring_methods = ["balanced_accuracy",
                   # "accuracy",
                   # "f1"
                   ]


class TestTrexClassifier:
    @pytest.mark.parametrize("trait_name",
                             [pytest.param("Sulfate_reducer", id="Sulfate_reducer", ),
                              pytest.param("Aerobe", id="Aerobe",
                                           marks=[pytest.mark.xfail])])  # file not found
    def test_load_training_files(self, trait_name):
        """
        Test training data loading. Check/catch invalid file formats.
        :param trait_name:
        :return:
        """
        full_path_genotype = DATA_PATH / f"{trait_name}.genotype"
        full_path_phenotype = DATA_PATH / f"{trait_name}.phenotype"
        training_records, genotype, phenotype, group = load_training_files(
            genotype_file=full_path_genotype,
            phenotype_file=full_path_phenotype,
            verb=True)
        assert genotype[0].identifier == first_genotype_accession[trait_name]
        assert phenotype[0].identifier == first_phenotype_accession[trait_name]
        return training_records, genotype, phenotype, group

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_train(self, trait_name, classifier):
        """
        Test TrexClassifier training. Using different traits.
        :param trait_name:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        _ = clf.train(records=training_records)

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("cv", cv_folds, ids=[str(x) for x in cv_folds])
    @pytest.mark.parametrize("scoring", scoring_methods, ids=scoring_methods)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_crossvalidate(self, trait_name, cv, scoring, classifier):
        """
        Test default crossvalidation of PICASVM class. Using several different traits, cv folds, and scoring methods.
        Compares with dictionary cv_scores.
        :param trait_name:
        :param cv:
        :param scoring:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        score_pred = clf.crossvalidate(records=training_records, cv=cv, scoring=scoring)[:2]
        if classifier.identifier in cv_scores_trex:
            score_target = cv_scores_trex[classifier.identifier][trait_name][cv][scoring]
            np.testing.assert_almost_equal(actual=score_pred, desired=score_target, decimal=1)

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_parameter_search(self, trait_name, classifier):
        """
        Test randomized parameter search.

        :param trait_name:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        clf_opt = clf.parameter_search(records=training_records, n_iter=5, return_optimized=True)

        # test retraining
        clf_opt.train(training_records)

    @pytest.mark.xfail()
    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_compleconta_cv(self, trait_name, classifier):
        """
        Perform compleconta-cv for each trait name using TrexClassifier class.
        :param trait_name:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        cccv_scores = clf.crossvalidate_cc(records=training_records, cv=5, comple_steps=3,
                                           conta_steps=3)
        if classifier.identifier in cccv_scores_trex:
            assert cccv_scores_trex[classifier.identifier][trait_name] == cccv_scores

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_compress_vocabulary(self, trait_name, classifier):
        """
        Perform feature compression tests
        :param trait_name:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        compress_vocabulary(records=training_records, pipeline=clf.cv_pipeline)
        vec = clf.cv_pipeline.named_steps["vec"]
        vec._validate_vocabulary()

        # check if vocabulary is set properly
        assert vec.fixed_vocabulary_

        # check if length of vocabulary is matching
        assert len(vec.vocabulary_) == num_of_features_uncompressed[trait_name]

        X, y, tn = get_x_y_tn(training_records)
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
    @pytest.mark.parametrize("n_features", [10_000])
    def test_recursive_feature_elimination(self, trait_name, n_features):
        """
        Perform feature compression tests only for SVM; counterindicated for XGB.
        :param trait_name:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        svm = TrexSVM(verb=True, random_state=RANDOM_STATE)
        recursive_feature_elimination(records=training_records,
                                      pipeline=svm.cv_pipeline,
                                      step=0.01,
                                      n_features=n_features,
                                      )
        vec = svm.cv_pipeline.named_steps["vec"]
        vec._validate_vocabulary()

        # check if vocabulary is set properly
        assert vec.fixed_vocabulary_

        # check if length of vocabulary is matching
        assert len(vec.vocabulary_) >= n_features

        X, y, tn = get_x_y_tn(training_records)
        X_trans = vec.transform(X)

        # check if number of unique features is matching
        assert X_trans.shape[1] >= n_features

        # check if all samples still have at least one feature present
        one_is_zero = False
        non_zero = X_trans.nonzero()
        for x in non_zero:
            if len(x) == 0:
                one_is_zero = True

        assert not one_is_zero
