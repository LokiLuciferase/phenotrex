#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#

from pica.io.io import load_training_files
from pica.ml.svm import PICASVM
from pica.util.plotting import compleconta_plot


if __name__ == "__main__":

    cccv_results = []
    #names = ["Psychro_or_tolerant", "Sulfate_reducer", "Aerobe_old"]#, "Aerobe_new"]
    names = ["T3SS"]
    for name in names:
        td, gr, pr, gp = load_training_files(genotype_file=f"tests/test_svm/{name}.genotype",
                                             phenotype_file=f"tests/test_svm/{name}.phenotype",
                                             groups_file=f"tests/test_svm/{name}.taxid", selected_rank="auto",
                                             verb=True)  # make training data set from genotype and phenotype files
        svm = PICASVM(verb=True)
        #svm.train(records=td, reduce_features=True)
        #print(svm.get_feature_weights())

        svm.crossvalidate(records=td, n_jobs=4, reduce_features=True, n_features=10000, groups=True, n_replicates=1)
        #svm.crossvalidate_cc(records=td, comple_steps=10, conta_steps=10, n_jobs=4, repeats=10, reduce_features=True)
        cccv_results.append(svm.cccv_result)
        compleconta_plot(svm.cccv_result)
    #compleconta_plot(cccv_results, conditions=names)
