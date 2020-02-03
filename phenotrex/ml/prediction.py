from phenotrex.io.flat import load_genotype_file, DEFAULT_TRAIT_SIGN_MAPPING
from phenotrex.io.serialization import load_classifier
try:
    from phenotrex.transforms import fastas_to_grs
except ModuleNotFoundError:
    from phenotrex.util.helpers import fail_missing_dependency as fastas_to_grs

def predict(fasta_files=tuple(), genotype=None, classifier=None, verb=False):
    """
    Predict phenotype from a set of (possibly gzipped) DNA or protein FASTA files
    or a single genotype file.
    NB: Genotype computation is highly expensive and performed on the fly on FASTA files.
    For increased speed when predicting multiple phenotypes, create a .genotype file to reuse
    with the command `compute-genotype`.

    :param fasta_files: An iterable of fasta file paths
    :param genotype: A genotype file path
    :param classifier: A pickled classifier file path
    :param verb: Whether to show progress of fasta file annotation.
    """
    if not len(fasta_files) and genotype is None:
        raise RuntimeError('Must either supply FASTA file(s) or single genotype file for prediction.')
    if len(fasta_files):
        grs_from_fasta = fastas_to_grs(fasta_files, n_threads=None, verb=verb)
    else:
        grs_from_fasta = []

    grs_from_file = load_genotype_file(genotype) if genotype is not None else []
    gr = grs_from_fasta + grs_from_file

    model = load_classifier(filename=classifier, verb=verb)
    preds, probas = model.predict(X=gr)
    translate_output = {trait_id: trait_sign for trait_sign, trait_id in
                        DEFAULT_TRAIT_SIGN_MAPPING.items()}
    print(f"# Trait: {model.trait_name}")
    print("Identifier\tTrait present\tConfidence")
    for record, result, probability in zip(gr, preds, probas):
        print(f"{record.identifier}\t{translate_output[result]}\t{str(round(probability[result], 4))}")
