#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
from time import time
from typing import List, Tuple, Dict

import six
import copy
from operator import itemgetter

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer

from pica.struct.records import TrainingRecord, GenotypeRecord
from pica.transforms.resampling import TrainingRecordResampler
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
            ("vec", CustomVectorizer(binary=True, dtype=np.bool)),
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

    def crossvalidate(self, records: List[TrainingRecord], cv: int = 5,
                      scoring: str = "balanced_accuracy", n_jobs=-1,
                      # TODO: add more complex scoring/reporting, e.g. AUC
                      demote=False, **kwargs) -> Tuple[float, float, float, float]:
        """
        Perform cv-fold crossvalidation
        :param records: List[TrainingRecords] to perform crossvalidation on.
        :param scoring: Scoring function of crossvalidation. Default: Balanced Accuracy.
        :param cv: Number of folds in crossvalidation. Default: 5
        :param n_jobs: Number of parallel jobs. Default: -1 (All processors used)
        :param kwargs: Unused
        :return: A list of mean score, score SD, mean fit time and fit time SD.
        """
        log_function = self.logger.debug if demote else self.logger.info
        log_function("Begin cross-validation on training data.")
        t1 = time()
        X, y, tn = self.__get_x_y_tn(records)
        crossval = cross_validate(estimator=self.pipeline, X=X, y=y, scoring=scoring, cv=cv, n_jobs=n_jobs)
        fit_times, score_times, scores = [crossval.get(x) for x in ("fit_time", "score_time", "test_score")]
        # score_time_mean, score_time_sd = float(np.mean(score_times)), float(np.std(score_times))
        fit_time_mean, fit_time_sd = float(np.mean(fit_times)), float(np.std(fit_times))
        score_mean, score_sd = float(np.mean(scores)), float(np.std(scores))
        t2 = time()
        log_function(f"Cross-validation completed.")
        log_function(f"Average fit time: {np.round(fit_time_mean, 2)} seconds.")
        log_function(f"Total duration of cross-validation: {np.round(t2 - t1, 2)} seconds.")
        return score_mean, score_sd, fit_time_mean, fit_time_sd

    def completeness_cv(self, records: List[TrainingRecord], cv: int = 5, samples: int = 10,
                        comple_steps: int = 20, conta_steps: int = 20,
                        scoring: str = "balanced_accuracy", n_jobs=-1, **kwargs) -> Dict[
        float, Dict[float, List[float]]]:
        """
        Perform cross-validation while resampling training features,
        simulating differential completeness and contamination.
        :param records: List[TrainingRecords] to perform crossvalidation on.
        :param cv: Number of folds in crossvalidation. Default: 5
        :param samples: # TODO: add functionality
        :param scoring: Scoring function of crossvalidation. Default: Balanced Accuracy.
        :param n_jobs: Number of parallel jobs. Default: -1 (All processors used)
        :param kwargs: Unused
        :return: A dict of cross validation scores and SD at defined completeness/contamination levels.
        """
        # TODO: is parallelization over folds or over compleconta levels more efficient?
        self.logger.info("Creating and fitting TrainingRecordResampler.")
        resampler = TrainingRecordResampler(random_state=2, verb=False)
        resampler.fit(records=records)
        self.logger.info("TrainingRecordResampler ready. Begin cross-validation over comple/conta values...")
        t1 = time()
        cv_scores = {}
        # iterate over comple/conta levels
        # TODO: make comple and conta steps settable
        for comple in np.arange(0, 1.05, 0.05):
            comple = np.round(comple, 2)
            self.logger.info(f"Comple: {comple}")
            cv_scores[comple] = {}
            for conta in np.arange(0, 1.05, 0.05):
                conta = np.round(conta, 2)
                try:
                    self.logger.info(f"\tConta: {conta}")
                    resampled_set = [resampler.get_resampled(x, comple, conta) for x in records]
                    cv_scores[comple][conta] = self.crossvalidate(resampled_set, demote=True)  # disable spam
                except ValueError:  # error due to inability to perform cv (no features)
                    self.logger.warning("Cross-validation failed for Completeness {comple} and Contamination {conta}."
                                        "\nThis is likely due to too small feature set at low comple/conta levels.")
                    cv_scores[comple][conta] = (np.nan, np.nan, np.nan, np.nan)
        t2 = time()
        self.logger.info(f"Resampling CV completed in {round((t2 - t1) / 60, 2)} mins.")
        return cv_scores

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

    def compress_vocabulary(self, records: List[TrainingRecord]):
        """
        Method to group features, that store redundant information.
        To avoid overfitting and speed up process.
        Might be replaced or complemented by a feature selection method in future

        compressing features is optional, for the test dataset it appears to be a lot slower:
        2.5 min compression + 2.1 sec cv. vs 3.3 sec cv without compression. TODO: Look for a faster way to compression?

        :return: nothing, sets the vocabulary for CountVectorizer step
        """

        t1 = time()

        X, y, tn = self.__get_x_y_tn(records) # we actually only need X
        vec = CountVectorizer(binary=True, dtype=np.bool)
        vec.fit(X)
        names = vec.get_feature_names()
        X_trans = vec.transform(X)

        size = len(names)
        self.logger.info(f"{size} Features found, starting compression")
        seen = {}
        vocabulary = {}
        for i in range(len(names)):   # num of features
            column = X_trans.A[:, i]
            key = tuple(column)
            found_id = seen.get(key)
            if not found_id:
                seen[key] = i
                vocabulary[names[i]] = i
            else:
                vocabulary[names[i]] = found_id
        size_after = len(seen)
        t2 = time()

        self.logger.info(f"Features compressed to {size_after} unique features in {np.round(t2 - t1, 2)} seconds.")

        # set vocabulary to vectorizer
        self.pipeline.named_steps["vec"].vocabulary = vocabulary
        self.pipeline.named_steps["vec"].fixed_vocabulary_ = True

    def get_feature_weights(self):
        """
        Extract the weights for features from pipeline/model
        :return: tuple of lists: feature names and weights
        """
        # get weights directly from the CalibratedClassifierCV object.
        # Each classifier has numpy array .coef_ of which we simply take the mean
        # this is not necessary the actual weight used in the final classifier, but enough to determine importance

        clf = self.pipeline.named_steps["clf"]
        num_features = len(clf.calibrated_classifiers_[0].base_estimator.coef_[0])
        mean_weights = np.zeros(num_features)
        for calibrated_classifier in clf.calibrated_classifiers_:
            weights = calibrated_classifier.base_estimator.coef_[0]
            mean_weights += weights

        mean_weights /= len(clf.calibrated_classifiers_)

        # get original names of the features from vectorization step, they might be compressed
        names = self.pipeline.named_steps["vec"].get_feature_names()

        # decompress
        weights=[]
        name_list=[]
        for feature, i in names:
            name_list.append(feature)
            weights.append(mean_weights[i]) # use the group weight for all members currently
            # TODO: weights should be adjusted if multiple original features were grouped together.

        return name_list, weights

    @staticmethod
    def validate_(X_test, y_test, estimator):
        """
        part of the crossvalidation (or compleconta) where only validation is performed
        :param X_test: test-records X values
        :param y_test: test-records y values
        :param estimator: classifier previously trained
        :return: score #TODO: expand, what else can be useful?
        """
        preds = estimator.predict(X_test)
        tot_per_y = np.array([y_test.count(x) for x in range(len(set(y_test)))])
        count_per_y = np.zeros(len(set(y_test)))
        for p, y in zip(preds, y_test):
            if p == y:
                count_per_y[y] += 1

        score = count_per_y / tot_per_y
        # mba = np.mean(score)
        # std = np.std(score)
        return score

    def crossvalidate2(self, records: List[TrainingRecord], cv: int = 5,
                      scoring: str = "balanced_accuracy", n_jobs=-1,
                      # TODO: add more complex scoring/reporting, e.g. AUC
                      demote=False, **kwargs) -> Tuple[float, float]:
        """
        Perform cv-fold crossvalidation, own function replacing module from sklearn.
        TODO: modify to simulate compleconta, crossvalidation could be still used from sklearn build in, I guess.
        TODO: parallelization
        :param records: List[TrainingRecords] to perform crossvalidation on.
        :param scoring: Scoring function of crossvalidation. Default: Balanced Accuracy.
        :param cv: Number of folds in crossvalidation. Default: 5
        :param n_jobs: Number of parallel jobs. Default: -1 (All processors used) #TODO not currently implemented
        :param kwargs: Unused
        :return: A list of mean score, score SD, mean fit time and fit time SD.
        """
        log_function = self.logger.debug if demote else self.logger.info
        log_function("Begin cross-validation on training data.")
        X, y, tn = self.__get_x_y_tn(records)
        skf = StratifiedKFold(n_splits=cv)

        t1 = time()
        classifier = copy.deepcopy(self.pipeline)
        scores=[]
        for train_index, test_index in skf.split(X, y):
            # separate in test and training set lists:
            X_test=[X[i] for i in test_index]
            X_train=[X[i] for i in train_index]
            y_test=[y[i] for i in test_index]
            y_train=[y[i] for i in train_index]

            # fit the copy of the pipeline
            classifier.fit(X=X_train, y=y_train, **kwargs)

            # validate and keep scores
            score = self.validate_(X_test, y_test, classifier)
            scores=np.append(scores, score)

        score_mean, score_sd = float(np.mean(scores)), float(np.std(scores))
        t2 = time()
        log_function(f"Cross-validation completed.")
        log_function(f"Total duration of cross-validation: {np.round(t2 - t1, 2)} seconds.")
        return score_mean, score_sd




class CustomVectorizer(CountVectorizer):
    """
    modified from CountVectorizer to override the _validate_vocabulary function, which invoked an error because
    multiple indices of the dictionary contained the same feature index. However, this is we intend.
    Other functions had to be adopted to allow decompression: get_feature_names,
    """


    def _validate_vocabulary(self):
        """
        overriding the validation which does not accept multiple feature-names to encode for one feature
        """
        if self.vocabulary:
            self.vocabulary_ = dict(self.vocabulary)
            self.fixed_vocabulary_ = True
        else:
            self.fixed_vocabulary_ = False


    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        # return value is different from normal CountVectorizer output: maintain dict instead of returning a list
        return sorted(six.iteritems(self.vocabulary_),
                                     key=itemgetter(1))

