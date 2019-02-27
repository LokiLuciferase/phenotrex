#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
import json

from pica.io.io import load_training_files, load_genotype_file, write_weights_file
from pica.ml.svm import PICASVM
from pica.util.serialization import save_ml, load_ml
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
    sp_compleconta_cv.add_argument("--threads", type=int, default=4,
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
        subp.add_argument("--num_of_features", default=10000, type=int,
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
        training_records, _, _ = load_training_files(args.genotype, args.phenotype, verb=args.verb)
        svm = PICASVM(C=args.svm_c, penalty=args.reg, tol=args.tol, verb=args.verb)

        if sn == "train":
            svm.train(records=training_records, reduce_features=args.reduce_features, n_features=args.num_of_features)
            if args.weights:
                weights = svm.get_feature_weights()
                weights_file_name = f"{args.out}.rank"
                write_weights_file(weights_file=weights_file_name, weights=weights)
            save_ml(obj=svm, filename=args.out, overwrite=False, verb=args.verb)

        elif sn == "crossvalidate":
            cv = svm.crossvalidate(records=training_records, cv=args.cv, n_replicates=args.replicates,
                                   reduce_features=args.reduce_features, n_features=args.num_of_features)
            mean_balanced_accuracy, mba_sd, misclassifications = cv
            logger.info(f"Mean balanced accuracy: {mean_balanced_accuracy} +/- {mba_sd}")

            # write misclassifications output to file if specified
            if args.out:
                identifier_list = [record.identifier for record in training_records]
                trait_sign_list = [record.trait_sign for record in training_records]
                sorted_tuples = sorted(zip(identifier_list, trait_sign_list, misclassifications),
                                       key=lambda k: k[2], reverse=True)
                header = ["Identifier", "Trait present", "Mis-classifications [frac.]"]
                trait_translation = {0: "NO", 1: "YES"}
                with open(args.out, "w") as output_file:
                    output_file.write("%s\n" % "\t".join(header))
                    for identifier, trait_sign, mcs in sorted_tuples:
                        output_file.write(f"{identifier}\t{trait_translation[trait_sign]}\t{mcs}\n")


        elif sn == "cccv":
            cccv = svm.crossvalidate_cc(records=training_records, cv=args.cv,
                                        comple_steps=args.comple_steps,
                                        conta_steps=args.conta_steps, n_jobs=args.threads,
                                        n_replicates=args.replicates, reduce_features=args.reduce_features,
                                        n_features=args.num_of_features)
            # write output in JSON-format as old pica did
            # TODO: add a graphical output?
            json.dump(cccv, args.out, indent="\t")

    elif sn == "predict":
        genotype_records = load_genotype_file(args.genotype)
        svm = load_ml(filename=args.classifier, verb=True)
        print(svm.predict(X=genotype_records))  # TODO: make proper output/file out

    elif sn == "weights":
        svm = load_ml(filename=args.classifier, verb=True)
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
