#
# Created by Lukas Lüftinger on 2/5/19.
#
from typing import List
from dataclasses import dataclass

"""Data structures containing Genotype and Phenotype information."""


@dataclass
class GenotypeRecord:
    identifier: str
    features:   List[str]


@dataclass
class PhenotypeRecord:
    identifier: str
    trait_name: str
    trait_sign: int


@dataclass
class TrainingRecord(GenotypeRecord, PhenotypeRecord):
    pass
