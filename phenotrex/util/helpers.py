#
# Created by Lukas Lüftinger on 14/02/2019.
#
from typing import List, Tuple

import numpy as np

from phenotrex.structure.records import TrainingRecord


def get_x_y_tn(records: List[TrainingRecord]) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Get separate X and y from list of TrainingRecord.
    Also infer trait name from first TrainingRecord.

    :param records: a List[TrainingRecord]
    :return: separate lists of features and targets, and the trait name
    """
    trait_name = records[0].trait_name
    X = np.array([" ".join(x.features) for x in records])
    y = np.array([x.trait_sign for x in records])
    return X, y, trait_name


def get_groups(records: List[TrainingRecord]) -> np.ndarray:
    """
    Get groups from list of TrainingRecords

    :param records:
    :return: list for groups
    """
    group_ids = np.array([x.group_id for x in records])
    return group_ids
