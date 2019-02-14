#
# Created by Lukas Lüftinger on 14/02/2019.
#
from typing import List, Tuple

from pica.struct.records import TrainingRecord


def get_x_y_tn(records: List[TrainingRecord]) -> Tuple[List, List, str]:
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
