from time import time
from typing import List

from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from phenotrex.structure.records import TrainingRecord
from phenotrex.util.logging import get_logger
from phenotrex.util.helpers import get_x_y_tn_ft

import numpy as np

DEFAULT_STEP_SIZE = 0.0025
DEFAULT_SCORING_FUNCTION = 'balanced_accuracy'


def recursive_feature_elimination(
    records: List[TrainingRecord],
    pipeline: Pipeline,
    step: float = DEFAULT_STEP_SIZE,
    n_features: int = None,
    random_state: np.random.RandomState = None
):
    """
    Function to apply RFE to limit the vocabulary used by the CustomVectorizer, optional step.

    :param records: list of TrainingRecords, entire training set.
    :param pipeline: the pipeline which vocabulary should be modified
    :param step: rate of features to eliminate at each step. the lower the number, the more steps
    :param n_features: number of features to select (if None: half of the provided features)
    :param random_state: random state for deterministic results
    :return: number of features used
    """
    t1 = time()

    X, y, tn, ft = get_x_y_tn_ft(records)
    vec = pipeline.named_steps["vec"]
    estimator = pipeline.named_steps["clf"]

    if not vec.vocabulary:
        vec.fit(X)
    previous_vocabulary = vec.vocabulary_

    if not n_features:
        n_features = len(previous_vocabulary) // 2

    X_trans = vec.transform(X)

    logger = get_logger(__name__, verb=True)
    split = StratifiedKFold(shuffle=True, n_splits=5, random_state=random_state)
    selector = RFECV(
        estimator,
        step=step,
        min_features_to_select=n_features,
        cv=split,
        n_jobs=5,
        scoring=DEFAULT_SCORING_FUNCTION
    )
    selector = selector.fit(X=X_trans, y=y)

    original_size = len(previous_vocabulary)
    support = selector.get_support()
    support = support.nonzero()[0]
    new_id = {support[x]: x for x in range(len(support))}
    vocabulary = {
        feature: new_id[i]
        for feature, i in previous_vocabulary.items()
        if new_id.get(i) is not None
    }
    size_after = selector.n_features_

    t2 = time()

    logger.info(
        f"{size_after}/{original_size} features selected using Recursive Feature Eliminiation."
        f" in {np.round(t2 - t1, 2)} seconds.")

    # set vocabulary to vectorizer
    pipeline.named_steps["vec"].vocabulary = vocabulary
    pipeline.named_steps["vec"].vocabulary_ = vocabulary
    pipeline.named_steps["vec"].fixed_vocabulary_ = True

    return size_after
