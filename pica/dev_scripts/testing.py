#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
from pica.io.io import load_phenotype_and_genotype_file, collate_training_data
from pica.ml.svm import PICASVM
from pica.util.serialization import save_ml, load_ml


if __name__ == "__main__":
    g, p = load_phenotype_and_genotype_file(genotype_file="pica/dev_scripts/Sulfate_reducer.genotype",
                                            phenotype_file="pica/dev_scripts/Sulfate_reducer.phenotype")
    td = collate_training_data(g, p, verb=True)  # make training data set from genotype and phenotype records

    svm = PICASVM(verb=True, random_state=2)
    svm.train(records=td)
    print("Predictions:", svm.predict(g[-10:]))  # maybe change output format
    print("Crossvalidation:", svm.crossvalidate(records=td))
    svm.train(records=td)  # error due to being fitted already
    save_ml(svm, filename="/scratch/lueftinger/test_save.bin", verb=True, overwrite=True)
    new_svm = load_ml("/scratch/lueftinger/test_save.bin", verb=True)
    print("Predictions:", new_svm.predict(g[-10:]))
