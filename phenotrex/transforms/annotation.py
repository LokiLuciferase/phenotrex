import os
from typing import List
from pathlib import Path
from pkg_resources import resource_filename
from tempfile import NamedTemporaryFile
from subprocess import check_call, DEVNULL
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from Bio.SeqIO import SeqRecord, parse
from deepnog.deepnog import load_nn, predict, set_device, create_df
from deepnog.dataset import ProteinDataset
from tqdm.auto import tqdm

from phenotrex.io.flat import load_fasta_file
from phenotrex.structure.records import GenotypeRecord

PRODIGAL_BIN_PATH = resource_filename('phenotrex', 'bin/prodigal')
DEEPNOG_WEIGHTS_PATH = resource_filename('deepnog', 'parameters/eggNOG5/2/deepencoding.pth')
DEEPNOG_ARCH = 'deepencoding'
EGGNOGDB_VERS = '5.0'

class PreloadedProteinDataset(ProteinDataset):
    """Hack ProteinDataset to load from list directly."""
    def __init__(self, protein_list: List[SeqRecord]):
        try:
            super().__init__(file='')
        except ValueError:
            pass  # >:D
        self.iter = (x for x in protein_list)


def fastas_to_grs(fasta_files: List[str], verb: bool = False,
                 n_threads: int = None) -> List[GenotypeRecord]:
    """
    Perform GenotypeRecord calculation for a list of FASTA files. Apply process-based parallelism
    since gene annotation scales well with cores.

    :param fasta_files: a list of DNA and/or protein FASTA files to be converted into GenotypeRecords.
    :param verb: Whether to display progress of annotation with tqdm.
    :param n_threads: Number of threads to run in parallel. Default, use up to all available CPU cores.
    :returns: A list of GenotypeRecords corresponding with supplied FASTA files.
    """
    n_threads = min(os.cpu_count(), n_threads) if n_threads is not None else os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        if len(fasta_files) > 1 and verb:
            annotations = tqdm(executor.map(fasta_to_gr, fasta_files),
                               total=len(fasta_files), desc='deepnog', unit='file')
        else:
            annotations = executor.map(fasta_to_gr, fasta_files)
    return list(annotations)


def fasta_to_gr(fasta_file: str, verb: bool = False) -> GenotypeRecord:
    """
    Given a fasta file, determine whether gene calling is required (DNA fasta) or if deepnog can be
    applied directly (protein fasta). If required, perform prodigal gene call and return
    GenotypeRecord (output of deepnog) of the file.

    :param fasta_file: A DNA or protein fasta file to be converted into GenotypeRecord.
    :param verb: Whether to display progress of annotation with tqdm.
    :returns: A single GenotypeRecord representing the sample.
    """
    fname = Path(str(fasta_file)).name
    seqtype, seqs = load_fasta_file(fasta_file)
    if seqtype == 'protein':
        return annotate_with_deepnog(fname, seqs, verb)
    else:
        return annotate_with_deepnog(fname, call_proteins(fasta_file), verb)


def call_proteins(fna_file: str) -> List[SeqRecord]:
    """
    Perform protein calling with prodigal.

    :param fna_file: A nucleotide fasta file.
    :returns: a list of SeqRecords suitable for annotation with deepnog.
    """
    with NamedTemporaryFile(mode='w+') as tmp_f:
        check_call([PRODIGAL_BIN_PATH, '-i', fna_file, '-a', tmp_f.name],
                   stderr=DEVNULL, stdout=DEVNULL)
        tmp_f.seek(0)
        return list(parse(tmp_f, 'fasta'))


def annotate_with_deepnog(identifier: str, protein_list: List[SeqRecord],
                          verb: bool = True) -> GenotypeRecord:
    """
    Perform calling of EggNOG5 clusters on a list of SeqRecords belonging to a sample, using deepnog.

    :param identifier: The name associated with the sample.
    :param protein_list: A list of SeqRecords containing protein sequences.
    :param verb: Whether to use tqdm for progress calculation
    :returns: a GenotypeRecord suitable for use with phenotrex.
    """
    device = set_device('auto')
    torch.set_num_threads(1)

    model_dict = torch.load(DEEPNOG_WEIGHTS_PATH, map_location=device)
    model = load_nn(DEEPNOG_ARCH, model_dict, device)
    class_labels = model_dict['classes']
    dataset = PreloadedProteinDataset(protein_list)

    preds, confs, ids, indices = predict(model, dataset, device,
                                         batch_size=1,
                                         num_workers=1,
                                         verbose=3 if verb else 0)
    threshold = None
    if hasattr(model, 'threshold'):
        threshold = model.threshold
    df = create_df(class_labels, preds, confs, ids, indices,
                   threshold=threshold, device=device, verbose=0)
    cogs = [x for x in df.prediction.unique() if x]
    return GenotypeRecord(identifier=identifier, features=cogs)
