#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
from time import time
from typing import List, Tuple, Dict

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE

from pica.struct.records import TrainingRecord, GenotypeRecord
from pica.ml.cccv import CompleContaCV
from pica.util.logging import get_logger
from pica.util.helpers import get_x_y_tn


class PICASVM:
    def __init__(self,
                 C: float = 5,
                 penalty: str = "l2",
                 tol: float = 1,
                 random_state: int = None,
                 verb=False,
                 *args, **kwargs):
        """
        Class which encapsulates a sklearn Pipeline of CountVectorizer (for vectorization of features) and
        LinearSVC wrapped in CalibratedClassifierCV for provision of probabilities via Platt scaling.
        Provides train() and crossvalidate() functionality equivalent to train.py and crossvalidateMT.py.
        :param C: Penalty parameter C of the error term. See LinearSVC documentation.
        :param penalty: Specifies the norm used in the penalization. See LinearSVC documentation.
        :param tol: Tolerance for stopping criteria. See LinearSVC documentation.
        :param random_state: A integer randomness seed for a Mersienne Twister (see np.random.RandomState)
        :param kwargs: Any additional named arguments are passed to the LinearSVC constructor.
        """
        self.trait_name = None
        self.cccv_result = None  # result of compleconta-crossvalidation saved in object so it gets pickled

        self.C = C
        self.penalty = penalty
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)
        self.logger = get_logger(__name__, verb=verb)
        self.verb = verb

        vectorizer = CustomVectorizer(binary=True, dtype=np.bool)
        classifier = LinearSVC(C=self.C, tol=self.tol, penalty=self.penalty, random_state=self.random_state, **kwargs)

        self.pipeline = Pipeline(steps=[
            ("vec", vectorizer),
            ("clf", CalibratedClassifierCV(classifier, method="sigmoid", cv=5))
        ])
        self.cv_pipeline = Pipeline(steps=[
            ("vec", vectorizer),
            ("clf", classifier)
        ])


    def train(self, records: List[TrainingRecord], **kwargs):
        """
        Fit CountVectorizer and train LinearSVC on a list of TrainingRecord.
        :param records: a List[TrainingRecord] for fitting of CountVectorizer and training of LinearSVC.
        :param kwargs: additional named arguments are passed to the fit() method of Pipeline.
        :returns: Whether the Pipeline has been fitted on the records.
        """
        # TODO: run compress vocabulary before?

        self.logger.info("Begin training classifier.")
        X, y, tn = get_x_y_tn(records)
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
                      # TODO: StratifiedKFold
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

        # TODO: run compress vocabulary before?

        log_function = self.logger.debug if demote else self.logger.info
        log_function("Begin cross-validation on training data.")
        t1 = time()
        X, y, tn = get_x_y_tn(records)
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
        Method to group features, that store redundant information, to avoid overfitting and speed up process (in some
        cases). Might be replaced or complemented by a feature selection method in future versions.

        Compressing vocabulary is optional, for the test dataset it took 30 seconds, while the time saved later on is not
        significant.

        :param records: a list of TrainingRecord objects.
        :return: nothing, sets the vocabulary for CountVectorizer step
        """

        t1 = time()

        X, y, tn = get_x_y_tn(records)  # we actually only need X
        vec = self.cv_pipeline.named_steps["vec"]
        if not vec.vocabulary:
            vec.fit(X)
            names = vec.get_feature_names()
        else:
            names = list(vec.vocabulary.keys())
        X_trans = vec.transform(X)

        size = len(names)
        self.logger.info(f"{size} Features found, starting compression")
        seen = {}
        new_vocabulary = {}
        for i in range(len(names)):
            column = X_trans.getcol(i).nonzero()[0]
            key = tuple(column)
            found_id = seen.get(key)
            if not found_id:
                seen[key] = i
                new_vocabulary[names[i]] = i
            else:
                new_vocabulary[names[i]] = found_id
        size_after = len(seen)
        t2 = time()

        self.logger.info(f"Features compressed to {size_after} unique features in {np.round(t2 - t1, 2)} seconds.")

        # set vocabulary to vectorizer
        self.pipeline.named_steps["vec"].vocabulary = new_vocabulary
        self.pipeline.named_steps["vec"].fixed_vocabulary_ = True

    def recursive_feature_elimination(self, records: List[TrainingRecord], n_steps: int = 5, n_features: int = None):
        """
        Function to apply RFE to limit the vocabulary used by the CustomVectorizer, optional step.
        :param records: list of TrainingRecords, entire training set.
        :param n_steps: number of elimination steps maximal
        :param n_features: number of features to select (if None: half of the provided features)
        :return:
        """

        t1 = time()

        self.logger.info(f"Starting recursive feature elimination")
        estimator = self.cv_pipeline.named_steps["clf"]
        selector = RFE(estimator, step=n_steps, n_features_to_select=n_features)

        X, y, tn = get_x_y_tn(records)
        vectorizer = self.cv_pipeline.named_steps["vec"]

        # get previous vocabulary (might be already compressed)
        if not vectorizer.vocabulary:
            vectorizer.fit(X)
            names = vectorizer.get_feature_names()
            previous_vocabulary = {names[i]: i for i in range(len(names))}
        else:
            previous_vocabulary = vectorizer.vocabulary
        X = vectorizer.transform(X)

        selector = selector.fit(X=X, y=y)

        original_size = len(previous_vocabulary)
        support = selector.get_support()
        support = support.nonzero()[0]
        size_after = len(support)
        new_id = {support[x]: x for x in range(len(support))}
        vocabulary = {feature: new_id[i] for feature, i in previous_vocabulary.items() if new_id.get(i)}

        t2 = time()

        self.logger.info(f"{size_after} features were selected of {original_size} using Recursive Feature Eliminiation"
                         f" in {np.round(t2 - t1, 2)} seconds.")

        # set vocabulary to vectorizer
        self.cv_pipeline.named_steps["vec"].vocabulary = vocabulary
        self.cv_pipeline.named_steps["vec"].fixed_vocabulary_ = True

    def coef_(self) -> np.array:
        """
        coef_ function to get weights from

        :return: coef_ for feature selection
        """

        clf = self.pipeline.named_steps["clf"]
        num_features = len(clf.calibrated_classifiers_[0].base_estimator.coef_[0])
        mean_weights = np.zeros(num_features)
        for calibrated_classifier in clf.calibrated_classifiers_:
            weights = calibrated_classifier.base_estimator.coef_[0]
            mean_weights += weights

        mean_weights /= len(clf.calibrated_classifiers_)

        return mean_weights


    def get_feature_weights(self) -> Tuple[List, List]:
        """
        Extract the weights for features from pipeline.
        :return: tuple of lists: feature names and weights
        """
        # TODO: find different way to feature weights that is closer to the real weight used for classification
        # get weights directly from the CalibratedClassifierCV object.
        # Each classifier has numpy array .coef_ of which we simply take the mean
        # this is not necessary the actual weight used in the final classifier, but enough to determine importance
        if self.trait_name is None:
            self.logger.error("Pipeline is not fitted. Cannot retrieve weights.")
            return [], []

        mean_weights = self.coef_()

        # get original names of the features from vectorization step, they might be compressed
        names = self.pipeline.named_steps["vec"].get_feature_names()

        # decompress
        weights = []
        name_list = []
        for feature, i in names:
            name_list.append(feature)
            weights.append(mean_weights[i])  # use the group weight for all members currently
            # TODO: weights should be adjusted if multiple original features were grouped together.

        return name_list, weights

    def crossvalidate_cc(self, records: List[TrainingRecord], cv: int = 5,
                           comple_steps: int = 20, conta_steps: int = 20,
                           n_jobs: int = -1, repeats: int = 10):
        """
        Instantiates an CompleContaCV object, and calls its run_cccv method with records. Returns its result.
        :param records: List[TrainingRecord] on which completeness_contamination_CV is to be performed
        :param cv: number of folds
        :param comple_steps: number of equidistand completeness levels
        :param conta_steps: number of equidistand contamination levels
        :param n_jobs: number of parallel jobs (-1 for n_cpus)
        :param repeats: Number of times the crossvalidation is repeated
        :return: A dictionary with mean balanced accuracies for each combination: dict[comple][conta]=mba
        """

        cccv = CompleContaCV(pipeline=self.cv_pipeline, cv=cv,
                             comple_steps=comple_steps, conta_steps=conta_steps,
                             n_jobs=n_jobs, repeats=repeats,
                             random_state=self.random_state, verb=self.verb)
        score_dict = cccv.run(records=records)
        self.cccv_result = score_dict
        return score_dict


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
        return sorted(self.vocabulary_.items(), key=lambda x: x[1])  # no stdlib dependency when using lambda
