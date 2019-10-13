from time import time
from typing import List, Tuple, Dict

import xgboost as xgb
from sklearn.pipeline import Pipeline

from pica.ml.trex_classifier import TrexClassifier
from pica.util.logging import get_logger


class TrexXGB(TrexClassifier):
    """
    Class which encapsulates a sklearn Pipeline of CountVectorizer (for vectorization of features) and
    xgb.sklearn.GradientBoostingClassifier.
    Provides train() and crossvalidate() functionality equivalent to train.py and crossvalidateMT.py.

    :param random_state: A integer randomness seed for a Mersienne Twister (see np.random.RandomState)
    :param kwargs: Any additional named arguments are passed to the XGBClassifier constructor.
    """
    def __init__(self, max_depth: int = 4, learning_rate: float = 0.05,
                 n_estimators: int = 30, gamma: float = 0, min_child_weight: int = 1,
                 subsample: float = 0.7, colsample_bytree: float = 0.3,
                 n_jobs: int = -1, random_state: int = None, verb=False, *args, **kwargs):
        super().__init__(random_state=random_state, verb=verb)
        self.logger = get_logger(__name__, verb=True)

        classifier = xgb.sklearn.XGBClassifier(missing=0, max_depth=max_depth,
                                               learning_rate=learning_rate,
                                               n_estimators=n_estimators, gamma=gamma,
                                               min_child_weight=min_child_weight,
                                               subsample=subsample, colsample_bytree=colsample_bytree,
                                               n_jobs=n_jobs, verbose=verb,
                                               **kwargs)

        self.pipeline = Pipeline(steps=[
            ("vec", self.vectorizer),
            ("clf", classifier)
        ])

        self.cv_pipeline = self.pipeline

    def get_feature_weights(self) -> Dict:
        if self.trait_name is None:
            self.logger.error("Pipeline is not fitted. Cannot retrieve weights.")
            return {}
        # get original names of the features from vectorization step, they might be compressed
        names = self.pipeline.named_steps["vec"].get_feature_names()
        weights = sorted(list(self.pipeline.named_steps['clf'].get_booster().get_fscore().items()),
                         key=lambda x: x[1], reverse=True)
        sorted_weights = {names[int(y[0].replace('f', ''))][0]: y[1] for x, y in zip(names, weights)}
        return sorted_weights
