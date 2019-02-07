#
# Created by Lukas Lüftinger on 05/02/2019.
#
from typing import List

import numpy as np
from numpy.random import RandomState
from sklearn.utils import resample

from pica.util.logging import get_logger
from pica.struct.records import TrainingRecord


class TrainingRecordResampler:
    def __init__(self,
                 random_state: float = None,
                 verb: bool = False):
        """
        Instantiates an object which can generate versions of a TrainingRecord
        resampled to defined completeness and contamination levels.
        Requires prior fitting with full List[TrainingRecord] to get sources of contamination for both classes.
        :param random_state: Randomness seed to use while resampling
        :param verb: Toggle verbosity
        """
        self.logger = get_logger(initname=self.__class__.__name__, verb=verb)
        self.random_state = RandomState(random_state)  # use numpy RandomState here. sklearn is ok with that.
        self.conta_source_pos = None
        self.conta_source_neg = None
        self.fitted = False

    def fit(self, records: List[TrainingRecord]):
        """
        Fit TrainingRecordResampler on full TrainingRecord list
        to determine set of positive and negative features for contamination resampling.
        :param records: the full List[TrainingRecord] on which ml training will commence.
        :return: True if fitting was performed, else False.
        """
        if self.fitted:
            self.logger.warning("TrainingRecordSampler already fitted on full TrainingRecord data."
                                " Refusing to fit again.")
            return False
        total_neg_featureset = []
        total_pos_featureset = []
        for record in records:
            if record.trait_sign == 1:
                total_pos_featureset += record.features
            elif record.trait_sign == 0:
                total_neg_featureset += record.features
            else:
                raise RuntimeError("Unexpected record sign found. Aborting.")
        self.conta_source_pos = np.array(total_pos_featureset)
        self.conta_source_neg = np.array(total_neg_featureset)
        self.fitted = True
        return True

    def get_resampled(self, record: TrainingRecord,
                      comple: float = 1,
                      conta: float = 0) -> TrainingRecord:
        """
        Resample a TrainingRecord to defined completeness and contamination levels. Comple=1, Conta=1 will double set size.
        :param comple: completeness of returned TrainingRecord features. Range: 0 - 1
        :param conta: contamination of returned TrainingRecord features. Range: 0 - 1
        :param record: the input TrainingRecord
        :return: a resampled TrainingRecord.
        """
        if not self.fitted:
            raise RuntimeError("TrainingRecordResampler is not fitted on full TrainingRecord set. Aborting.")
        if not 0 <= comple <= 1 or not 0 <= conta <= 1:
            raise RuntimeError("Invalid comple/conta settings. Must be between 0 and 1.")

        features = record.features
        n_features_comple = int(np.floor(len(features) * comple))
        n_features_conta = int(np.floor(len(features) * conta))  # TODO: calculate after or before incompleting?

        # make incomplete
        incomplete_features = resample(features,
                                       replace=False,
                                       n_samples=n_features_comple,
                                       random_state=self.random_state)
        self.logger.info(f"Reduced features of TrainingRecord {record.identifier} "
                         f"from {len(features)} to {n_features_comple}.")

        # make contaminations
        record_class = record.trait_sign
        if record.trait_sign == 1:
            conta_source = self.conta_source_neg
        elif record.trait_sign == 0:
            conta_source = self.conta_source_pos
        else:
            raise RuntimeError(f"Unexpected record sign found: {record.trait_sign}. Aborting.")
        conta_features = list(self.random_state.choice(a=conta_source,
                                                       size=n_features_conta,
                                                       replace=False))
        # TODO: what if not enough conta features?
        self.logger.info(f"Enriched features of TrainingRecord {record.identifier} "
                         f"with {len(conta_features)} features from {'positive' if record_class == 0 else 'negative'} set.")

        new_record = TrainingRecord(identifier=record.identifier,
                                    trait_name=record.trait_name,
                                    trait_sign=record.trait_sign,
                                    features=incomplete_features + conta_features)
        return new_record
