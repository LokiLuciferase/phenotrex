#
# Created by Lukas Lüftinger on 2/5/19.
#
import logging
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import json

from pica.util.logging import get_logger
from pica.struct.records import GenotypeRecord, PhenotypeRecord, GroupRecord, TrainingRecord

DEFAULT_TRAIT_SIGN_MAPPING = {"YES": 1, "NO": 0}


def load_genotype_file(input_file: str) -> List[GenotypeRecord]:
    """
    Loads a genotype .tsv file and returns a list of GenotypeRecord for each entry.
    :param input_file: The path to the input genotype file.
    :return: List[GenotypeRecord] of records in the genotype file
    """
    with open(input_file) as genotype_file:
        genotype_records = []
        for line in genotype_file:
            identifier, *features = line.strip().split("\t")
            genotype_records.append(GenotypeRecord(identifier=identifier,
                                                   features=features))
    dupcount = Counter([x.identifier for x in genotype_records])
    if dupcount.most_common()[0][1] > 1:
        raise RuntimeError(f"Duplicate entries found in genotype file: {dupcount}")
    return sorted(genotype_records, key=lambda x: x.identifier)


def load_phenotype_file(input_file: str, sign_mapping: Dict[str, int]=None) -> List[PhenotypeRecord]:
    """
    Loads a phenotype .tsv file and returns a list of PhenotypeRecord for each entry.
    :param input_file: The path to the input phenotype file.
    :param sign_mapping: an optional Dict to change mappings of trait sign. Default: {"YES": 1, "NO": 0}
    :return: List[PhenotypeRecord] of records in the phenotype file
    """
    with open(input_file) as phenotype_file:
        identifiers = []
        trait_signs = []
        _, trait_name = phenotype_file.readline().strip().split("\t")
        for line in phenotype_file:
            identifier, trait_sign = line.strip().split("\t")
            identifiers.append(identifier)
            trait_signs.append(trait_sign)

    dupcount = Counter(identifiers)
    if dupcount.most_common()[0][1] > 1:
        raise RuntimeError(f"Duplicate entries found in genotype file: {dupcount}")

    if sign_mapping is None:
        sign_mapping = DEFAULT_TRAIT_SIGN_MAPPING

    trait_signs = [sign_mapping.get(x, None) for x in trait_signs]
    phenotype_records = [PhenotypeRecord(identifier=x,
                                         trait_name=trait_name,
                                         trait_sign=y) for x, y in zip(identifiers, trait_signs)]
    ret = sorted(phenotype_records, key=lambda x: x.identifier)

    return ret


def load_groups_file(input_file: str, selected_rank: str = None) -> List[GroupRecord]:
    """
    Loads a .tsv file which contains group or taxid for each sample in the other training files.
    Automatically classifies the
    :param input_file: path to the file that is processed
    :param selected_rank: the standard rank that is selected (optional) if not set, the inputfile is assumed to contain groups,
     i.e. each unique entry of the ID will be a new group
    :return: a list of GroupRecords
    """

    with open(input_file) as groups_file:
        identifiers = []
        group_ids = []
        for line in groups_file:
            identifier, group_id = line.strip().split("\t")
            identifiers.append(identifier)
            group_ids.append(group_id)

    dupcount = Counter(identifiers)
    if dupcount.most_common()[0][1] > 1:
        raise RuntimeError(f"Duplicate entries found in groups file: {dupcount}")

    if selected_rank:
        try:
            from pica.util.taxonomy import get_taxonomic_group_mapping
            group_name_mapping, group_id_mapping = get_taxonomic_group_mapping(group_ids=group_ids,
                                                                               selected_rank=selected_rank)
            group_records = [GroupRecord(identifier=x, group_id=group_id_mapping[y], group_name=group_name_mapping[y])
                             for x, y in zip(identifiers, group_ids)]

        except ImportError:
            raise RuntimeError("A required package was not found. ete3 is required to support taxonomic ids for"
                               " grouping. Please install or divide your samples into groups manually")

    else:
        group_id_mapping = {x: group_id for group_id, x in enumerate(set(group_ids))}
        group_records = [GroupRecord(identifier=x, group_id=group_id_mapping[y], group_name=y)
                         for x, y in zip(identifiers, group_ids)]

    ret = sorted(group_records, key=lambda x: x.identifier)

    return ret


def collate_training_data(genotype_records: List[GenotypeRecord], phenotype_records: List[PhenotypeRecord],
                          group_records: List[GroupRecord],
                          universal_genotype: bool = False, verb: bool = False) -> List[TrainingRecord]:
    """
    Returns a list of TrainingRecord from two lists of GenotypeRecord and PhenotypeRecord.
    To be used for training and CV of PICASVM.
    Checks if 1:1 mapping of phenotypes and genotypes exists,
    and if all PhenotypeRecords pertain to same trait.
    :param genotype_records: List[GenotypeRecord]
    :param phenotype_records: List[PhenotypeRecord]
    :param group_records: List[GroupRecord] optional, if leave one group out is the split strategy
    :param universal_genotype: Whether to use an universal genotype file.
    :param verb: toggle verbosity.
    :return: List[TrainingRecord]
    """
    logger = get_logger(__name__, verb=verb)
    gr_dict = {x.identifier: x for x in genotype_records}
    pr_dict = {x.identifier: x for x in phenotype_records}
    gp_dict = {x.identifier: x for x in group_records}
    traits = set(x.trait_name for x in phenotype_records)
    if universal_genotype:
        if not set(gr_dict.keys()).issuperset(set(pr_dict.keys())):
            raise RuntimeError("Not all identifiers of phenotype records were found in the universal genotype."
                               "Cannot collate to TrainingRecords.")
    else:
        different_identifiers = set(gr_dict.keys()).symmetric_difference(set(pr_dict.keys()))
        if different_identifiers:
            logger.error(f"Identifiers not present in all record types: {different_identifiers}")
            raise RuntimeError("Different identifiers found among genotype and phenotype records. "
                               "Cannot collate to TrainingRecords.")
        if group_records:
            if len(gp_dict) != len(pr_dict):
                raise RuntimeError("Group and phenotype/genotype records are of unequal length."
                                   "Cannot collate to TrainingRecords.")
            if set(gp_dict.keys()) != set(pr_dict.keys()):
                raise RuntimeError("Different identifiers found among groups and phenotype/genotype records. "
                                   "Cannot collate to TrainingRecords.")

    if len(traits) > 1:
        raise RuntimeError("More than one traits have been found in phenotype records. "
                           "Cannot collate to TrainingRecords.")

    ret = [TrainingRecord(identifier=pr_dict[x].identifier,
                          trait_name=pr_dict[x].trait_name,
                          trait_sign=pr_dict[x].trait_sign,
                          features=gr_dict[x].features,
                          group_name=gp_dict[x].group_name,
                          group_id=gp_dict[x].group_id) for x in pr_dict.keys()]
    logger.info(f"Collated genotype and phenotype records into {len(ret)} TrainingRecord.")
    return ret


def load_training_files(genotype_file: str, phenotype_file: str, groups_file: str = None, selected_rank: str = None,
                        universal_genotype: bool = False, verb=False) -> Tuple[List[TrainingRecord],
                                                                               List[GenotypeRecord],
                                                                               List[PhenotypeRecord],
                                                                               List[GroupRecord]]:
    """
    Convenience function to load phenotype and genotype file together, and return a list of TrainingRecord.
    :param genotype_file: The path to the input genotype file.
    :param phenotype_file: The path to the input phenotype file.
    :param groups_file: The path to the input groups file.
    :param selected_rank: The selected standard rank to use for taxonomic grouping
    :param universal_genotype: Whether to use an universal genotype file.
    :param verb: toggle verbosity.
    :return: Tuple[List[TrainingRecord], List[GenotypeRecord], List[PhenotypeRecord]]
    """
    logger = get_logger(__name__, verb=verb)
    gr = load_genotype_file(genotype_file)
    pr = load_phenotype_file(phenotype_file)
    if groups_file:
        gp = load_groups_file(groups_file, selected_rank=selected_rank)
    else:
        # if not set, each sample gets its own group (not used currently)
        gp = [GroupRecord(identifier=x.identifier, group_name=x.identifier, group_id=y) for y, x in enumerate(pr)]
    logger.info("Genotype and Phenotype records successfully loaded from file.")
    return collate_training_data(gr, pr, gp, universal_genotype=universal_genotype, verb=verb), gr, pr, gp


def write_weights_file(weights_file: str, weights: Dict):
    """
    Function to write the weights to specified file in tab-separated fashion with header
    :param weights_file: The path to the file to which the output will be written
    :param weights: sorted dictionary storing weights with feature names as indices
    :return: nothing
    """

    header = ["Rank", "Feature_name", "Weight"]

    with open(weights_file, "w") as output_file:
        output_file.write("%s\n" % "\t".join(header))
        for rank, (name, weight) in enumerate(weights.items()):
            output_file.write(f"{rank+1}\t{name.upper()}\t{weight}\n")

def write_cccv_accuracy_file(output_file: str, cccv_results):
    """
    Function to write the cccv accuracies in the exact format that phendb uses as input
    :param output_file: file
    :param cccv_results:
    :return: nothing
    """

    write_list = []

    for completeness, data in cccv_results.items():
        for contamination, nested_data in data.items():
            write_item = {
                "mean_balanced_accuracy": nested_data["score_mean"],
                "stddev_balanced_accuracy": nested_data["score_sd"],
                "contamination": contamination,
                "completeness": completeness
            }
            write_list.append(write_item)
    with open(output_file, "w") as outf_handler:
        json.dump(write_list, outf_handler, indent="\t")


def write_misclassifications_file(output_file: str, records: List[TrainingRecord], misclassifications,
                                  use_groups: bool = False):
    """
    Function to write the misclassifications file
    :param output_file: name of the outputfile
    :param records: List of trainingRecord objects
    :param misclassifications: List of percentages of misclassifications
    :param use_groups: toggles average over groups and groups output
    :return:
    """

    identifier_list = [record.identifier for record in records]
    trait_sign_list = [record.trait_sign for record in records]
    if use_groups:
        group_names = [record.group_name for record in records]
        identifier_list = list(set(group_names))
        grouped_mcs = []
        grouped_signs = []
        for group in identifier_list:
            group_mcs = [mcs for mcs, group_name in zip(misclassifications, group_names) if group == group_name]
            group_sign = [trait_sign for trait_sign, group_name in zip(trait_sign_list, group_names)
                          if group == group_name]
            grouped_mcs.append(np.mean(group_mcs))
            grouped_signs.append(np.mean(group_sign))

        trait_sign_list = grouped_signs
        misclassifications = grouped_mcs

    sorted_tuples = sorted(zip(identifier_list, trait_sign_list, misclassifications),
                           key=lambda k: k[2], reverse=True)
    header = ["Identifier", "Trait present", "Mis-classifications [frac.]"]
    trait_translation = {y: x for x, y in DEFAULT_TRAIT_SIGN_MAPPING.items()}
    with open(output_file, "w") as outf:
        outf.write("%s\n" % "\t".join(header))
        for identifier, trait_sign, mcs in sorted_tuples:
            outf.write(f'{identifier}\t{trait_translation.get(trait_sign, "MIXED")}\t{mcs}\n')
