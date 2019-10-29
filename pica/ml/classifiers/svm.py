#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
from typing import Dict

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from pica.ml.trex_classifier import TrexClassifier
from pica.util.logging import get_logger


class TrexSVM(TrexClassifier):
    """
    Class which encapsulates a sklearn Pipeline of CountVectorizer (for vectorization of features) and
    sklearn.svm.LinearSVC.
    Provides train() and crossvalidate() functionality equivalent to train.py and crossvalidateMT.py.

    :param random_state: A integer randomness seed for a Mersienne Twister (see np.random.RandomState)
    :param kwargs: Any additional named arguments are passed to the XGBClassifier constructor.
    """

    identifier = 'SVM'

    def __init__(self, C: float = 5., penalty: str = "l2", tol: float = 1.,
                 random_state: int = None, verb=False,
                 *args, **kwargs):
        super().__init__(random_state=random_state, verb=verb)
        self.C = C
        self.penalty = penalty
        self.tol = tol
        self.default_search_params = {
            'C': np.logspace(-6, 4, 30).round(8),
            'tol': np.logspace(0, -5, 10).round(8),
            'max_iter': np.logspace(2, 4.3, 20).astype(int)
        }
        self.logger = get_logger(__name__, verb=verb)

        if self.penalty == "l1":
            self.dual = False
        else:
            self.dual = True

        classifier = LinearSVC(C=self.C, tol=self.tol, penalty=self.penalty, dual=self.dual,
                               class_weight="balanced", random_state=self.random_state, **kwargs)

        self.pipeline = Pipeline(steps=[
            ("vec", self.vectorizer),
            ("clf", CalibratedClassifierCV(classifier, method="sigmoid", cv=5))
        ])
        self.cv_pipeline = Pipeline(steps=[
            ("vec", self.vectorizer),
            ("clf", classifier)
        ])

    def _get_coef_(self, pipeline: Pipeline = None) -> np.array:
        r"""
        Interface function to get `coef\_` from classifier used in the pipeline specified
        this might be useful if we switch the classifier, most of them already have a `coef\_` attribute


        :param pipeline: pipeline from which the classifier should be used
        :return: `coef\_` for feature weight report
        """
        if not pipeline:
            pipeline = self.pipeline

        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "coef_"):
            return_weights = clf.coef_
        else:  # assume calibrated classifier
            weights = np.array([c.base_estimator.coef_[0] for c in clf.calibrated_classifiers_])
            return_weights = np.median(weights, axis=0)
        return return_weights

    def get_feature_weights(self) -> Dict:
        """
        Extract the weights for features from pipeline.

        :return: sorted Dict of feature name: weight
        """
        # TODO: find different way to feature weights that is closer to the real weight used for classification
        # get weights directly from the CalibratedClassifierCV object.
        # Each classifier has numpy array .coef_ of which we simply take the mean
        # this is not necessary the actual weight used in the final classifier, but enough to determine importance
        if self.trait_name is None:
            self.logger.error("Pipeline is not fitted. Cannot retrieve weights.")
            return {}

        mean_weights = self._get_coef_()

        # get original names of the features from vectorization step, they might be compressed
        names = self.pipeline.named_steps["vec"].get_feature_names()

        # decompress
        weights = {feature: mean_weights[i] for feature, i in names}

        # sort by absolute value
        sorted_weights = {feature: weights[feature] for feature in sorted(weights, key=lambda key: abs(weights[key]),
                                                                          reverse=True)}
        # TODO: weights should be adjusted if multiple original features were grouped together. probably not needed
        #  if we rely on feature selection in near future
        return sorted_weights
