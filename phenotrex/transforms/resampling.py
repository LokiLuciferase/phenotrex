#
# Created by Lukas Lüftinger on 05/02/2019.
#
from typing import List

import numpy as np
from numpy.random import RandomState
from sklearn.utils import resample

from phenotrex.util.logging import get_logger
from phenotrex.structure.records import TrainingRecord


class TrainingRecordResampler:
    """
    Instantiates an object which can generate versions of a TrainingRecord
    resampled to defined completeness and contamination levels.
    Requires prior fitting with full List[TrainingRecord]
    to get sources of contamination for both classes.

    :param random_state: Randomness seed to use while resampling
    :param verb: Toggle verbosity
    """
    def __init__(
        self,
        random_state: float = None,
        verb: bool = False
    ):
        self.logger = get_logger(initname=self.__class__.__name__, verb=verb)
        self.random_state = random_state if type(random_state) is RandomState else RandomState(random_state)
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
                total_pos_featureset.append(record.features)
            elif record.trait_sign == 0:
                total_neg_featureset.append(record.features)
            else:
                raise RuntimeError("Unexpected record sign found. Aborting.")
        self.conta_source_pos = np.array(total_pos_featureset)
        self.conta_source_neg = np.array(total_neg_featureset)
        self.fitted = True
        return True

    def get_resampled(
        self,
        record: TrainingRecord,
        comple: float = 1.,
        conta: float = 0.
    ) -> TrainingRecord:
        """
        Resample a TrainingRecord to defined completeness and contamination levels.
        Comple=1, Conta=1 will double set size.

        :param comple: completeness of returned TrainingRecord features. Range: 0 - 1
        :param conta: contamination of returned TrainingRecord features. Range: 0 - 1
        :param record: the input TrainingRecord
        :return: a resampled TrainingRecord.
        """
        if not self.fitted:
            raise RuntimeError(
                "TrainingRecordResampler is not fitted on full TrainingRecord set. Aborting."
            )
        if not 0 <= comple <= 1 or not 0 <= conta <= 1:
            raise RuntimeError("Invalid comple/conta settings. Must be between 0 and 1.")

        features = record.features
        n_features_comple = int(np.floor(len(features) * comple))

        # make incomplete
        incomplete_features = resample(
            features, replace=False, n_samples=n_features_comple, random_state=self.random_state
        )
        self.logger.info(
            f"Reduced features of TrainingRecord {record.identifier} "
            f"from {len(features)} to {n_features_comple}."
        )
        # make contaminations
        record_class = record.trait_sign
        if record.trait_sign == 1:
            # guard against very small sample errors after StratifiedKFold
            if self.conta_source_neg.shape[0] == 1:
                source_set_id = 0
            else:
                source_set_id = self.random_state.randint(0, self.conta_source_neg.shape[0] - 1)
            conta_source = list(self.conta_source_neg[source_set_id])
        elif record.trait_sign == 0:
            if self.conta_source_pos.shape[0] == 1:
                source_set_id = 0
            else:
                source_set_id = self.random_state.randint(0, self.conta_source_pos.shape[0] - 1)
            conta_source = list(self.conta_source_pos[source_set_id])
        else:
            raise RuntimeError(f"Unexpected record sign found: {record.trait_sign}. Aborting.")

        n_features_conta = min(len(conta_source), int(np.floor(len(conta_source) * conta)))
        conta_features = list(self.random_state.choice(
            a=conta_source, size=n_features_conta, replace=False
        ))
        # TODO: what if not enough conta features?
        self.logger.info(
            f"Enriched features of TrainingRecord {record.identifier} "
            f"with {len(conta_features)} features from "
            f"{'positive' if record_class == 0 else 'negative'} set."
        )
        new_record = TrainingRecord(
            identifier=record.identifier,
            trait_name=record.trait_name,
            trait_sign=record.trait_sign,
            feature_type=record.feature_type,
            features=incomplete_features + conta_features,
            group_name=None,
            group_id=None
        )
        return new_record
