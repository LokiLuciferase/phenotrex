#
# Created by Lukas Lüftinger on 2/5/19.
#
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class GenotypeRecord:
    """
    Structure containing the genomic features to which learning is applied, for each sample
    referenced by `identifier`.
    """
    identifier: str
    features: List[str]

    def __repr__(self):
        return f"{self.identifier} n_features={len(self.features)}"


@dataclass
class PhenotypeRecord:
    """
    Structure containing ground truth class values (0 for trait absent, 1 for trait present) for the
    trait `trait_name` in sample `identifier`.
    """
    identifier: str
    trait_name: str
    trait_sign: int

    def __repr__(self):
        return f"{self.identifier} trait({self.trait_name})={self.trait_sign}"


@dataclass
class GroupRecord:
    """
    Structure containing grouping information for each sample for Leave-one-group-out CV.
    """
    identifier: str
    group_name: Optional[str]
    group_id: Optional[int]

    def __repr__(self):
        return f"{self.identifier} group({self.group_name})={self.group_id}"


@dataclass
class TrainingRecord(GenotypeRecord, PhenotypeRecord, GroupRecord):
    """
    Structure which collates information from Genotype-, Phenotype- and GroupRecords, creating
    a single observation suitable as machine learning input for each sample.
    """
    def __repr__(self):
        gr_repr = GenotypeRecord.__repr__(self).split(' ')[1]
        pr_repr = PhenotypeRecord.__repr__(self).split(' ')[1]
        gro_repr = GroupRecord.__repr__(self).split(' ')[1]
        return f"id={self.identifier} {' '.join([gr_repr, pr_repr, gro_repr])}"
