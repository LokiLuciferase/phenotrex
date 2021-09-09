#
# Created by Lukas Lüftinger on 2/5/19.
#
from typing import List, Dict, Tuple, Optional
from collections import Counter
import json
import gzip

import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC, HasStopCodon, _verify_alphabet
import numpy as np

from phenotrex.util.logging import get_logger
from phenotrex.structure.records import GenotypeRecord, PhenotypeRecord, GroupRecord, TrainingRecord

DEFAULT_TRAIT_SIGN_MAPPING = {"YES": 1, "NO": 0}


def _is_gzipped(f: str) -> bool:
    try:
        with gzip.open(f) as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def load_fasta_file(input_file: str) -> Tuple[str, List]:
    """
    Load a fasta file into a list of SeqRecords.

    :param input_file: The path to the input fasta file.
    :returns: A tuple of the sequence type ('protein' or 'dna'), and the list of SeqRecords.
    """
    if _is_gzipped(input_file):
        openfunc = gzip.open
        bit = 'rt'
    else:
        openfunc = open
        bit = 'r'
    with openfunc(input_file, bit) as handle:
        seqs = [x.upper() for x in SeqIO.parse(handle=handle, format='fasta',
                                               alphabet=IUPAC.ambiguous_dna)]
        if not all(_verify_alphabet(x.seq) for x in seqs):
            handle.seek(0)
            seqs = [x.upper() for x in SeqIO.parse(handle=handle, format='fasta',
                                                   alphabet=HasStopCodon(IUPAC.extended_protein))]
            if not all(_verify_alphabet(x.seq) for x in seqs):
                raise ValueError('Invalid input file (neither DNA nor protein FASTA).')
            return 'protein', seqs
        return 'dna', seqs


def load_genotype_file(input_file: str) -> List[GenotypeRecord]:
    """
    Loads a genotype .tsv file and returns a list of GenotypeRecord for each entry.

    :param input_file: The path to the input genotype file.
    :return: List[GenotypeRecord] of records in the genotype file
    """
    with open(input_file) as genotype_file:
        metadata = dict()
        genotype_lines = []
        genotype_records = []
        for line in genotype_file:
            if line.strip().startswith('#'):
                k, v = line[1:].strip().split(':', maxsplit=1)
                metadata[k] = v
            else:
                genotype_lines.append(line)

        metadata = {**{'feature_type': 'legacy'}, **metadata}

        for line in genotype_lines:
            identifier, *features = line.strip().split("\t")
            genotype_records.append(
                GenotypeRecord(
                    identifier=identifier,
                    feature_type=metadata['feature_type'],
                    features=features
                ))

    dupcount = Counter([x.identifier for x in genotype_records])
    if dupcount.most_common()[0][1] > 1:
        raise RuntimeError(f"Duplicate entries found in genotype file: {dupcount}")
    return sorted(genotype_records, key=lambda x: x.identifier)


def load_phenotype_file(
    input_file: str, sign_mapping: Dict[str, int] = None
) -> List[PhenotypeRecord]:
    """
    Loads a phenotype .tsv file and returns a list of PhenotypeRecord for each entry.

    :param input_file: The path to the input phenotype file.
    :param sign_mapping: an optional Dict to change mappings of trait sign.
                         Default: {"YES": 1, "NO": 0}
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
    Group-Ids may be ncbi-taxon-ids or arbitrary group names.
    Taxon-Ids are only used if a standard rank is selected,
    otherwise user-specified group-ids are assumed.
    Automatically classifies the [TODO missing text?]

    :param input_file: path to the file that is processed
    :param selected_rank: the standard rank that is selected (optional) if not set,
                        the input file is assumed to contain groups,
                        i.e., each unique entry of the ID will be a new group
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
            from phenotrex.util.taxonomy import get_taxonomic_group_mapping
            group_name_mapping, group_id_mapping = get_taxonomic_group_mapping(
                group_ids=group_ids, selected_rank=selected_rank
            )
            group_records = [GroupRecord(identifier=x, group_id=group_id_mapping[y],
                                         group_name=group_name_mapping[y])
                             for x, y in zip(identifiers, group_ids)]

        except ImportError:
            raise RuntimeError(
                "A required package was not found. ete3 is required to support taxonomic ids for"
                " grouping. Please install or divide your samples into groups manually")

    else:
        group_id_mapping = {x: group_id for group_id, x in enumerate(set(group_ids))}
        group_records = [GroupRecord(identifier=x, group_id=group_id_mapping[y], group_name=y)
                         for x, y in zip(identifiers, group_ids)]

    ret = sorted(group_records, key=lambda x: x.identifier)

    return ret


def load_params_file(params_file: str) -> Dict:
    """
    Load a JSON file of training parameters.

    :param params_file: The input file.
    :return: A dictionary of training parameters.
    """
    with open(params_file, 'r') as fin:
        return json.load(fin)


def write_genotype_file(genotypes: List[GenotypeRecord], output_file: str):
    """
    Saves a list of GenotypeRecords to a .genotype file.

    :param genotypes: The genotypes to write to a file.
    :param output_file: The output file path.
    """
    feature_types = list(set(x.feature_type for x in genotypes))
    if len(feature_types) > 1:
        raise ValueError(
            'Cannot write GenotypeRecords with different feature_types to the same genotype file.'
        )
    with open(output_file, 'w') as genotype_file:
        genotype_file.write(f'#feature_type:{feature_types[0]}\n')
        for g in genotypes:
            genotype_file.write('\t'.join([g.identifier, *g.features, '\n']))


def write_params_file(params_file: str, params: Dict):
    """
    Write a JSON file of training parameters.

    :param params_file: The output file path.
    :param params: A dictionary of training parameters.
    :return: A dictionary of training parameters.
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NumpyEncoder, self).default(obj)

    with open(params_file, 'w') as fout:
        json.dump(params, fp=fout, indent=2, cls=NumpyEncoder)
        fout.write('\n')


def collate_training_data(
    genotype_records: List[GenotypeRecord],
    phenotype_records: List[PhenotypeRecord],
    group_records: List[GroupRecord],
    verb: bool = False
) -> List[TrainingRecord]:
    """
    Returns a list of TrainingRecord from two lists of GenotypeRecord and PhenotypeRecord.
    To be used for training and CV of TrexClassifier.
    Checks if 1:1 mapping of phenotypes and genotypes exists,
    and if all PhenotypeRecords pertain to same trait.

    :param genotype_records: List[GenotypeRecord]
    :param phenotype_records: List[PhenotypeRecord]
    :param group_records: List[GroupRecord] optional, if leave one group out is the split strategy
    :param verb: toggle verbosity.
    :return: A list of TrainingRecords.
    """
    logger = get_logger(__name__, verb=verb)
    gr_dict = {x.identifier: x for x in genotype_records}
    pr_dict = {x.identifier: x for x in phenotype_records}
    gp_dict = {x.identifier: x for x in group_records}
    traits = set(x.trait_name for x in phenotype_records)
    if not set(gr_dict.keys()).issuperset(set(pr_dict.keys())):
        raise RuntimeError(
            "Not all identifiers of phenotype records were found in the phenotype file. "
            "Cannot collate to TrainingRecords."
        )
    if not set(gp_dict.keys()).issuperset(set(pr_dict.keys())):
        raise RuntimeError(
            "Not all identifiers of phenotype records were found in the groups file. "
            "Cannot collate to TrainingRecords."
        )
    if len(traits) > 1:
        raise RuntimeError(
            "More than one trait has been found in phenotype records. "
            "Cannot collate to TrainingRecords."
        )
    ret = [
        TrainingRecord(
            identifier=pr_dict[x].identifier,
            trait_name=pr_dict[x].trait_name,
            trait_sign=pr_dict[x].trait_sign,
            feature_type=gr_dict[x].feature_type,
            features=gr_dict[x].features,
            group_name=gp_dict[x].group_name,
            group_id=gp_dict[x].group_id
        ) for x in pr_dict.keys()
    ]
    logger.info(f"Collated genotype and phenotype records into {len(ret)} TrainingRecord.")
    return ret


def load_training_files(
    genotype_file: str,
    phenotype_file: str,
    groups_file: str = None,
    selected_rank: str = None,
    verb=False
) -> Tuple[
    List[TrainingRecord], List[GenotypeRecord], List[PhenotypeRecord], List[GroupRecord]
]:
    """
    Convenience function to load phenotype, genotype and optionally groups file together,
    and return a list of TrainingRecord.

    :param genotype_file: The path to the input genotype file.
    :param phenotype_file: The path to the input phenotype file.
    :param groups_file: The path to the input groups file. Optional.
    :param selected_rank: The selected standard rank to use for taxonomic grouping
    :param verb: toggle verbosity.
    :return: The collated TrainingRecords as well as single genotype, phenotype and group records
    """
    logger = get_logger(__name__, verb=verb)
    gr = load_genotype_file(genotype_file)
    pr = load_phenotype_file(phenotype_file)
    if groups_file:
        gp = load_groups_file(groups_file, selected_rank=selected_rank)
    else:
        # if not given, each sample gets its own group (not used currently)
        gp = [
            GroupRecord(identifier=x.identifier, group_name=x.identifier, group_id=y)
            for y, x in enumerate(pr)
        ]
    tr = collate_training_data(gr, pr, gp, verb=verb)
    logger.info("Records successfully loaded from file.")
    return tr, gr, pr, gp


def write_weights_file(weights_file: str, weights: Dict, annots: List[Optional[str]] = None):
    """
    Function to write the weights to specified file in tab-separated fashion with header

    :param weights_file: The path to the file to which the output will be written
    :param weights: sorted dictionary storing weights with feature names as indices
    :param annots: annotations for the features names. Optional.
    :return: nothing
    """
    names, weight_vals = zip(*list(weights.items()))
    out = pd.DataFrame({'Feature Name': names, 'Weight': weight_vals})
    if annots is not None:
        out['Feature Annotation'] = annots
    out.index.name = 'Rank'
    out = out.reset_index(drop=False)
    out['Rank'] += 1
    out.to_csv(weights_file, sep='\t', index=False)


def write_cccv_accuracy_file(output_file: str, cccv_results):
    """
    Function to write the cccv accuracies in the exact format that phendb uses as input.

    :param output_file: file
    :param cccv_results:
    :return:
    """
    write_list = []
    for completeness, data in cccv_results.items():
        for contamination, nested_data in data.items():
            write_item = {
                "mean_balanced_accuracy"  : nested_data["score_mean"],
                "stddev_balanced_accuracy": nested_data["score_sd"],
                "contamination"           : contamination,
                "completeness"            : completeness
            }
            write_list.append(write_item)
    with open(output_file, "w") as outf_handler:
        json.dump(write_list, outf_handler, indent="\t")
        outf_handler.write('\n')


def load_cccv_accuracy_file(cccv_file: str) -> Dict:
    """
    Function to load cccv accuracies from phendb format.

    :param cccv_file: The CCCV results file.
    :return: A Dict of CCCV results in the internal CCCV results format.
    """
    cccv_results = {}
    with open(cccv_file) as fin:
        loaded = json.load(fin)
    for row in loaded:
        score_mean, score_sd, conta, comple = row.values()
        comple_dict = cccv_results.setdefault(comple, {})
        comple_dict[conta] = {
            'score_mean': score_mean,
            'score_sd': score_sd
        }
    return cccv_results


def write_misclassifications_file(
    output_file: str, records: List[TrainingRecord], misclassifications, use_groups: bool = False
):
    """
    Function to write the misclassifications file.

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
            group_mcs = [mcs for mcs, group_name in zip(misclassifications, group_names)
                         if group == group_name]
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
