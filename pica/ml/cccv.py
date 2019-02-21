#
# Created by Lukas Lüftinger on 14/02/2019.
#
import os
import copy
from time import time
from typing import List, Tuple, Dict, Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from pica.struct.records import TrainingRecord
from pica.transforms.resampling import TrainingRecordResampler
from pica.util.logging import get_logger
from pica.util.helpers import get_x_y_tn


class CompleContaCV:
    def __init__(self, pipeline: Pipeline, scoring_function: Callable = balanced_accuracy_score, cv: int = 5,
                 comple_steps: int = 20, conta_steps: int = 20,
                 n_jobs: int = -1, repeats: int = 10, random_state: np.random.RandomState = None, verb: bool = False):
        """
        A class containing all custom completeness/contamination cross-validation functionality.
        :param scoring_function: Sklearn-like scoring function of crossvalidation. Default: Balanced Accuracy.
        :param cv: Number of folds in crossvalidation. Default: 5
        :param comple_steps: number of steps between 0 and 1 (relative completeness) to be simulated
        :param conta_steps: number of steps between 0 and 1 (relative contamination level) to be simulated
        :param n_jobs: Number of parallel jobs. Default: -1 (All processors used)
        :param repeats: Number of times the crossvalidation is repeated
        :param random_state: An integer random seed or instance of np.random.RandomState
        """
        self.pipeline = pipeline
        self.cv = cv
        self.scoring_method = scoring_function
        self.comple_steps = comple_steps
        self.conta_steps = conta_steps
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.repeats = repeats
        self.random_state = random_state if type(random_state) is np.random.RandomState else np.random.RandomState(random_state)
        self.logger = get_logger(__name__, verb=verb)

    def _validate_subset(self, records: List[TrainingRecord], estimator: Pipeline):
        """
        Use a fitted Pipeline to predict scores on resampled test data.
        part of the compleconta crossvalidation where only validation is performed.
        :param records: test records as a List of TrainingRecord objects
        :param estimator: classifier previously trained as a sklearn.Pipeline object
        :return: score
        """
        X, y, tn = get_x_y_tn(records)
        preds = estimator.predict(X)
        score = self.scoring_method(y, preds)
        return score

    def _replicates(self, records: List[TrainingRecord], cv: int = 5,
                                    comple_steps: int = 20, conta_steps: int = 20,
                                    repeats: int = 10):
        """
        Generator function to yield test/training sets which will be fed into subprocesses for _completeness_cv
        :param records: the complete set of TrainingRecords
        :param cv: number of folds in the crossvalidation to be performed
        :param comple_steps: number of steps between 0 and 1 (relative completeness) to be simulated
        :param conta_steps: number of steps between 0 and 1 (relative contamination level) to be simulated
        :param repeats: number of repeats for the entire crossvalidation
        :return: parameter list to submit to worker process
        """
        for r in range(repeats):
            X, y, tn = get_x_y_tn(records)
            skf = StratifiedKFold(n_splits=cv, random_state=self.random_state)
            fold = 0
            for train_index, test_index in skf.split(X, y):
                fold += 1
                # separate in training set lists:
                X_train = [X[i] for i in train_index]
                y_train = [y[i] for i in train_index]
                test_records = [records[i] for i in test_index]
                starting_message = f"Starting comple/conta replicate {r + 1}/{repeats}: fold {fold}"
                yield [test_records, X_train, y_train, comple_steps, conta_steps, self.logger.level, starting_message]

    def _completeness_cv(self, param, **kwargs) -> Dict[float, Dict[float, float]]:
        """
        Perform completeness/contamination simulation and testing for one fold. This is a separate function only called
        by run_cccv which spawns subprocesses using a ProcessPoolExecutor from concurrent.futures
        :param param: List [test_records, X_train, y_train, comple_steps, conta_steps, starting_message]
        workaround to get multiple parameters into this function. (using processor.map) #TODO find nicer solution?
        """
        # unpack parameters
        test_records, X_train, y_train, comple_steps, conta_steps, verb, starting_message = param

        # needed to create a new logger, self.logger not accessible from a different process
        logger = get_logger(__name__, verb=verb)
        logger.info(starting_message)

        classifier = copy.deepcopy(self.pipeline)
        classifier.fit(X=X_train, y=y_train, **kwargs)

        # initialize the resampler with the test_records only, so the samples are unknown to the classifier
        resampler = TrainingRecordResampler(random_state=self.random_state, verb=False)
        resampler.fit(records=test_records)
        cv_scores = {}
        comple_increment = 1 / comple_steps
        conta_increment = 1 / conta_steps
        for comple in np.arange(0, 1.05, comple_increment):
            comple = np.round(comple, 2)
            cv_scores[comple] = {}
            for conta in np.arange(0, 1.05, conta_increment):
                conta = np.round(conta, 2)
                resampled_set = [resampler.get_resampled(x, comple, conta) for x in test_records]
                cv_scores[comple][conta] = self._validate_subset(resampled_set, classifier)
        return cv_scores

    def run(self, records: List[TrainingRecord]):
        """
        :param records: List[TrainingRecords] to perform compleconta-crossvalidation on.
        :return: A dictionary with mean balanced accuracies for each combination: dict[comple][conta]=mba
        """
        # TODO: run compress_vocabulary before?

        self.logger.info("Begin completeness/contamination matrix crossvalidation on training data.")
        t1 = time()
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            cv_scores = executor.map(self._completeness_cv,
                                     self._replicates(records, self.cv,
                                                                      self.comple_steps, self.conta_steps,
                                                                      self.repeats))
        t2 = time()
        mba = {}
        cv_scores_list = [x for x in cv_scores]

        for comple in cv_scores_list[0].keys():
            mba[comple] = {}
            for conta in cv_scores_list[0][comple].keys():
                single_result = [cv_scores_list[r][comple][conta] for r in range(self.repeats * self.cv)]
                mean_over_fold_and_replicates = np.mean(single_result)
                std_over_fold_and_replicates = np.std(single_result)
                mba[comple][conta] = {"score_mean": mean_over_fold_and_replicates,
                                      "score_sd": std_over_fold_and_replicates}
        self.logger.info(f"Total duration of cross-validation: {np.round(t2 - t1, 2)} seconds.")
        return mba
