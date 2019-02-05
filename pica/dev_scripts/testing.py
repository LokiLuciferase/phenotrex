#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
from pica.io.io import load_training_files
from pica.ml.svm import PICASVM
from pica.util.serialization import save_ml, load_ml


if __name__ == "__main__":
    td, gr, pr = load_training_files(genotype_file="pica/dev_scripts/Sulfate_reducer.genotype",
                             phenotype_file="pica/dev_scripts/Sulfate_reducer.phenotype",
                             verb=True) # make training data set from genotype and phenotype files
    svm = PICASVM(verb=True, random_state=2)
    svm.train(records=td)
    print("Predictions:", svm.predict(gr[-10:]))  # maybe change output format
    print("Crossvalidation:", svm.crossvalidate(records=td))
    svm.train(records=td)  # error due to being fitted already
    save_ml(svm, filename="test_save.bin", verb=True, overwrite=True)
    new_svm = load_ml("test_save.bin", verb=True)
    print("Predictions:", new_svm.predict(gr[-10:]))

