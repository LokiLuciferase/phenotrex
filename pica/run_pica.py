#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse

from pica.io.io import load_training_files, load_genotype_file, write_weights_file, DEFAULT_TRAIT_SIGN_MAPPING,\
    write_misclassifications_file, write_cccv_accuracy_file
from pica.ml.classifiers.svm import TrexSVM
from pica.util.serialization import save_classifier, load_classifier
from pica.util.logging import get_logger


def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")

    # train
    sp_train_descr = """Train PICA model with .phenotype and .genotype files."""
    sp_train = subparsers.add_parser("train", description=sp_train_descr)
    sp_train.add_argument("-w", "--weights", action="store_true",
                          help="Write feature ranks and weights in a separate tsv file named <output file>.rank")
    sp_train.add_argument("-o", "--out", required=True, type=str,
                          help="Filename of output file.")

    # crossvalidate
    sp_crossvalidate_descr = """Crossvalidate on data from .phenotype and .genotype files."""
    sp_crossvalidate = subparsers.add_parser("crossvalidate", description=sp_crossvalidate_descr)
    sp_crossvalidate.add_argument("--cv", type=int, default=5,
                                  help="Number of folds in cross-validation.")
    sp_crossvalidate.add_argument("-o", "--out", required=False, type=str,
                                  help="Filename of output file showing mis-classifications. (optional)")
    sp_crossvalidate.add_argument("--replicates", type=int, default=10,
                                  help="Number of replicates for the cross-validation.")
    sp_crossvalidate.add_argument("--threads", type=int, default=-1, help="Number of threads to use.")

    # leave one group out (taxonomy)
    sp_logo_descr = """Leave-one-group-out-validation on data from .phenotype, .genotype and .groups or .taxid files.
     If no groups file is provided, leave-one-out-validation is performed"""
    sp_logo = subparsers.add_parser("logo", description=sp_logo_descr)
    sp_logo.add_argument("--groups", type=str, help="Inputfile that specifies the taxids or groups")
    sp_logo.add_argument("--rank", required=False, type=str,
                         help="Taxonomic rank on which the separation should be done (optional), if non specified:"
                              "use groups without taxonomy")
    sp_logo.add_argument("-o", "--out", required=False)
    sp_logo.add_argument("--threads", type=int, default=-1, help="Number of threads to use.")

    # compleconta_cv
    sp_compleconta_cv_descr = """Crossvalidate for each step of completeness/contamination of the input data."""
    sp_compleconta_cv = subparsers.add_parser("cccv", description=sp_compleconta_cv_descr)
    sp_compleconta_cv.add_argument("--cv", type=int, default=5,
                                   help="Number of folds in cross-validation.")
    sp_compleconta_cv.add_argument("--comple-steps", type=int, default=20,
                                   help="Number of equidistant completeness levels to resample to.")
    sp_compleconta_cv.add_argument("--conta-steps", type=int, default=20,
                                   help="Number of equidistant contamination levels to resample to.")
    sp_compleconta_cv.add_argument("--replicates", type=int, default=10,
                                   help="Number of replicates for the cross-validation.")
    sp_compleconta_cv.add_argument("--threads", type=int, default=-1,
                                   help="Number of threads to be used for this calculation.")
    sp_compleconta_cv.add_argument("-o", "--out", required=True, type=str,
                                   help="Filename of output file.")

    # required for all previous commands
    for name, subp in subparsers.choices.items():
        subp.add_argument("-p", "--phenotype", required=True,
                          help=".phenotype .tsv file.")
        subp.add_argument("-c", "--svm_c", default=5, type=float,
                          help="SVM parameter C.")
        subp.add_argument("-t", "--tol", default=1,
                          help="SVM stopping tolerance.")
        subp.add_argument("-r", "--reg", default="l2", choices=["l1", "l2"],
                          help="Regularization strategy.")
        subp.add_argument("-f", "--reduce_features", action="store_true",
                          help="Apply reduction of feature space before training operation")
        subp.add_argument("--num_of_features", default=1000, type=int,
                          help="Number of features aimed by recursive feature elimination")
    # predict
    sp_predict_descr = """Predict trait sign of .genotype file contents"""
    sp_predict = subparsers.add_parser("predict", description=sp_predict_descr)
    sp_predict.add_argument("-c", "--classifier", required=True,
                            help="pickled PICA classifier to predict with.")

    # required for ALL commands
    for name, subp in subparsers.choices.items():
        subp.add_argument("-v", "--verb", default=False, action="store_true",
                          help="Toggle verbosity")
        subp.add_argument("-g", "--genotype", required=True,
                          help=".genotype .tsv file.")

    sp_weights_decr = """Write feature weights from existing classifier to a specified output file"""
    sp_weights = subparsers.add_parser("weights", description=sp_weights_decr)
    sp_weights.add_argument("-c", "--classifier", required=True,
                            help="pickled PICA classifier")
    sp_weights.add_argument("-o", "--out", type=str, required=True, help="Filename of output file")

    return parser.parse_args()


def call(args):
    """discern subcommand and execute with collected args"""
    logger = get_logger("PICA", verb=True)
    sn = args.subparser_name
    if sn in ("train", "crossvalidate", "cccv"):
        training_records, _, _, _ = load_training_files(genotype_file=args.genotype, phenotype_file=args.phenotype,
                                                        verb=args.verb)
        svm = TrexSVM(C=args.svm_c, penalty=args.reg, tol=args.tol, verb=args.verb)

        if sn == "train":
            svm.train(records=training_records, reduce_features=args.reduce_features, n_features=args.num_of_features)
            if args.weights:
                weights = svm.get_feature_weights()
                weights_file_name = f"{args.out}.rank"
                write_weights_file(weights_file=weights_file_name, weights=weights)
            save_classifier(obj=svm, filename=args.out, overwrite=False, verb=args.verb)

        elif sn == "crossvalidate":
            cv = svm.crossvalidate(records=training_records,
                                   cv=args.cv,
                                   n_replicates=args.replicates,
                                   n_jobs=args.threads,
                                   reduce_features=args.reduce_features,
                                   n_features=args.num_of_features)
            mean_balanced_accuracy, mba_sd, misclassifications = cv
            logger.info(f"Mean balanced accuracy: {mean_balanced_accuracy} +/- {mba_sd}")

            # write misclassifications output to file if specified
            if args.out:
                write_misclassifications_file(args.out, records=training_records, misclassifications=misclassifications)

        elif sn == "cccv":
            cccv = svm.crossvalidate_cc(records=training_records, cv=args.cv,
                                        comple_steps=args.comple_steps,
                                        conta_steps=args.conta_steps, n_jobs=args.threads,
                                        n_replicates=args.replicates, reduce_features=args.reduce_features,
                                        n_features=args.num_of_features)
            # write output in JSON-format as old pica did
            # TODO: add a graphical output?
            write_cccv_accuracy_file(args.out, cccv)

    elif sn == "logo":
        training_records, _, _, _ = load_training_files(genotype_file=args.genotype,
                                                        phenotype_file=args.phenotype,
                                                        groups_file=args.groups,
                                                        selected_rank=args.rank,
                                                        verb=args.verb)
        svm = TrexSVM(C=args.svm_c, penalty=args.reg, tol=args.tol, verb=args.verb)
        cv = svm.crossvalidate(records=training_records, n_replicates=1, groups=True, n_jobs=args.threads,
                               reduce_features=args.reduce_features, n_features=args.num_of_features)
        mean_balanced_accuracy, mba_sd, misclassifications = cv
        logger.info(f"Mean balanced accuracy: {mean_balanced_accuracy} +/- {mba_sd}")

        # write misclassifications output to file if specified
        if args.out:
            logger.info(f"Fractions of misclassifications per sample/group are written file: {args.out}")
            write_misclassifications_file(args.out, records=training_records, misclassifications=misclassifications,
                                          use_groups=True)

    elif sn == "predict":
        genotype_records = load_genotype_file(args.genotype)
        svm = load_classifier(filename=args.classifier, verb=True)
        results, probabilities = svm.predict(X=genotype_records)

        translate_output = {trait_id: trait_sign for trait_sign, trait_id in DEFAULT_TRAIT_SIGN_MAPPING.items()}

        sys.stdout.write("Identifier\tTrait present\tConfidence\n")
        for record, result, probability in zip(genotype_records, results, probabilities):
            sys.stdout.write(f"{record.identifier}\t{translate_output[result]}\t{probability[result]}\n")

    elif sn == "weights":
        svm = load_classifier(filename=args.classifier, verb=True)
        weights = svm.get_feature_weights()
        write_weights_file(weights_file=args.out, weights=weights)

    else:
        logger.warning("Unknown subcommand. See -h or --help for available commands.")
        sys.exit(1)


def main():
    args = get_args()
    call(args)


if __name__ == "__main__":
    main()
