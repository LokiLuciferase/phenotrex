#
# Created by Lukas Lüftinger on 2/5/19.
#
from typing import List
from dataclasses import dataclass

"""Data structures containing Genotype and Phenotype information."""


@dataclass
class GenotypeRecord:
    """ TODO add docstring """
    identifier: str
    features:   List[str]


@dataclass
class PhenotypeRecord:
    """ TODO add docstring """
    identifier: str
    trait_name: str
    trait_sign: int


@dataclass
class GroupRecord:
    """ TODO add docstring """
    identifier: str
    group_name: str
    group_id: int


@dataclass
class TrainingRecord(GenotypeRecord, PhenotypeRecord, GroupRecord):
    """ TODO add docstring """
    pass
