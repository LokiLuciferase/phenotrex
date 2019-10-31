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
    features: List[str]

    def __repr__(self):
        return f"{self.identifier} n_features={len(self.features)}"


@dataclass
class PhenotypeRecord:
    """ TODO add docstring """
    identifier: str
    trait_name: str
    trait_sign: int

    def __repr__(self):
        return f"{self.identifier} trait({self.trait_name})={self.trait_sign}"


@dataclass
class GroupRecord:
    """ TODO add docstring """
    identifier: str
    group_name: str
    group_id: int

    def __repr__(self):
        return f"{self.identifier} group({self.group_name})={self.group_id}"


@dataclass
class TrainingRecord(GenotypeRecord, PhenotypeRecord, GroupRecord):
    """ TODO add docstring """
    pass

    def __repr__(self):
        gr_repr = GenotypeRecord.__repr__(self).split(' ')[1]
        pr_repr = PhenotypeRecord.__repr__(self).split(' ')[1]
        gro_repr = GroupRecord.__repr__(self).split(' ')[1]
        return f"id={self.identifier} {' '.join([gr_repr, pr_repr, gro_repr])}"
