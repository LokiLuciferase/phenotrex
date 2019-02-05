#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
from time import time
from typing import List, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer

from pica.data_structures.records import TrainingRecord, GenotypeRecord
from pica.util.logging import get_logger


# TODO: For now NO crossvalidation over completeness-contamination gradients, no feature grouping and no plotting.
class PICASVM:
    def __init__(self,
                 C: float = 5,
                 penalty: str = "l2",
                 tol: float = 1,
                 verb=False,
                 *args, **kwargs):
        """
        Class which encapsulates a sklearn Pipeline of CountVectorizer (for vectorization of features) and
        LinearSVC wrapped in CalibratedClassifierCV for provision of probabilities via Platt scaling.
        Provides train() and crossvalidate() functionality equivalent to train.py and crossvalidateMT.py.
        :param C: Penalty parameter C of the error term. See LinearSVC documentation.
        :param penalty: Specifies the norm used in the penalization. See LinearSVC documentation.
        :param tol: Tolerance for stopping criteria. See LinearSVC documentation.
        :param kwargs: Any additional named arguments are passed to the LinearSVC constructor.
        """
        self.trait_name = None
        self.C = C
        self.penalty = penalty
        self.tol = tol
        self.logger = get_logger(__name__, verb=verb)
        self.pipeline = Pipeline(steps=[
            ("vec", CountVectorizer()),
            ("clf", CalibratedClassifierCV(LinearSVC(C=self.C,
                                                     tol=self.tol,
                                                     penalty=self.penalty,
                                                     **kwargs),
                                           method="sigmoid", cv=5))])

    @staticmethod
    def __get_x_y_tn(records: List[TrainingRecord]):
        """
        Get separate X and y from list of TrainingRecord.
        Also infer trait name from first TrainingRecord.
        :param records: a List[TrainingRecord]
        :return: separate lists of features and targets, and the trait name
        """
        trait_name = records[0].trait_name
        X = [" ".join(x.features) for x in records]
        y = [x.trait_sign for x in records]
        return X, y, trait_name

    def train(self, records: List[TrainingRecord], **kwargs):
        """
        Fit CountVectorizer and train LinearSVC on a list of TrainingRecord.
        :param records: a List[TrainingRecord] for fitting of CountVectorizer and training of LinearSVC.
        :param kwargs: additional named arguments are passed to the fit() method of Pipeline.
        :returns: Whether the Pipeline has been fitted on the records.
        """
        self.logger.info("Begin training classifier.")
        X, y, tn = self.__get_x_y_tn(records)
        if self.trait_name is not None:
            self.logger.warning("Pipeline is already fitted. Refusing to fit again.")
            return False
        self.trait_name = tn
        self.pipeline.fit(X=X, y=y, **kwargs)
        self.logger.info("Classifier training completed.")
        return True

    def crossvalidate(self, records: List[TrainingRecord], cv=5,
                      scoring: str = "balanced_accuracy", **kwargs) -> Tuple[float, float, float, float]:
        """
        Perform cv-fold crossvalidation
        :param records: List[TrainingRecords] to perform crossvalidation on.
        :param scoring: Scoring function of crossvalidation. Default: Balanced Accuracy.
        :param cv: Number of folds in crossvalidation. Default: 5
        :param kwargs: Unused
        :return: A list of mean score, score SD, mean fit time and fit time SD.
        """
        self.logger.info("Begin cross-validation on training data.")
        t1 = time()
        X, y, tn = self.__get_x_y_tn(records)
        crossval = cross_validate(estimator=self.pipeline, X=X, y=y, scoring=scoring, cv=cv)
        fit_times, score_times, scores = [crossval.get(x) for x in ("fit_time", "score_time", "test_score")]
        fit_time_mean, fit_time_sd = float(np.mean(fit_times)), float(np.std(fit_times))
        score_time_mean, score_time_sd = float(np.mean(score_times)), float(np.std(score_times))
        score_mean, score_sd = float(np.mean(scores)), float(np.std(scores))
        t2 = time()
        self.logger.info(f"Cross-validation completed.")
        self.logger.info(f"Average fit time: {np.round(fit_time_mean, 2)} seconds.")
        self.logger.info(f"Total duration of cross-validation: {np.round(t2 - t1, 2)} seconds.")
        return score_mean, score_sd, fit_time_mean, fit_time_sd

    def predict(self, X: List[GenotypeRecord]) -> Tuple[List[str], np.ndarray]:
        """
        Predict trait sign and probability of each class for each supplied GenotypeRecord.
        :param X: A List of GenotypeRecord for each of which to predict the trait sign
        :return: a Tuple of predictions and probabilities of each class for each GenotypeRecord in X.
        """
        features: List[str] = [" ".join(x.features) for x in X]
        preds = self.pipeline.predict(X=features)
        probas = self.pipeline.predict_proba(X=features)  # class probabilities via Platt scaling
        return preds, probas
