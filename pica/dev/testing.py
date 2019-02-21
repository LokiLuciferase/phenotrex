#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#

from pica.io.io import load_training_files
from pica.ml.svm import PICASVM
from pica.util.plotting import compleconta_plot


if __name__ == "__main__":

    cccv_results = []
    names = ["Psychro_or_tolerant", "Sulfate_reducer", "Aerobe_old"]#, "Aerobe_new"]
    for name in names:
        td, gr, pr = load_training_files(genotype_file=f"tests/test_svm/{name}.genotype",
                                         phenotype_file=f"tests/test_svm/{name}.phenotype",
                                         verb=True)  # make training data set from genotype and phenotype files
        svm = PICASVM(verb=True)
        svm.crossvalidate_cc(records=td, comple_steps=20, conta_steps=20)
        cccv_results.append(svm.cccv_result)
        compleconta_plot(svm.cccv_result)
    compleconta_plot(cccv_results, conditions=names)
