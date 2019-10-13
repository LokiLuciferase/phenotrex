#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#

from pica.io.io import load_training_files
from pica.ml.classifiers.svm import TrexSVM
from pica.ml.classifiers.xgbm import TrexXGB
from pica.util.plotting import compleconta_plot


def test_svm():
    cccv_results = []
    #names = ["Psychro_or_tolerant", "Sulfate_reducer", "Aerobe_old"]  #, "Aerobe_new"]
    names = ["T3SS"]
    for name in names:
        td, gr, pr, gp = load_training_files(genotype_file=f"tests/test_svm/{name}.genotype",
                                             phenotype_file=f"tests/test_svm/{name}.phenotype",
                                             groups_file=f"tests/test_svm/{name}.taxid", selected_rank="auto",
                                             verb=True)  # make training data set from genotype and phenotype files
        svm = TrexSVM(verb=True)
        # svm.train(records=td, reduce_features=True)
        # print(svm.get_feature_weights())
        #svm.crossvalidate(records=td, n_jobs=4, reduce_features=True, n_features=10000, groups=True, n_replicates=1)
        svm.crossvalidate_cc(records=td, comple_steps=10, conta_steps=10,
                             n_jobs=6, reduce_features=True)
        cccv_results.append(svm.cccv_result)
        compleconta_plot(svm.cccv_result)


def test_xgb():
    cccv_results = []
    #names = ["Psychro_or_tolerant", "Sulfate_reducer", "Aerobe_old"]  #, "Aerobe_new"]
    names = ["T3SS"]
    for name in names:
        td, gr, pr, gp = load_training_files(genotype_file=f"tests/test_svm/{name}.genotype",
                                             phenotype_file=f"tests/test_svm/{name}.phenotype",
                                             groups_file=f"tests/test_svm/{name}.taxid", selected_rank="auto",
                                             verb=True)  # make training data set from genotype and phenotype files
        xgb = TrexXGB(verb=True, subsample=1, colsample_bytree=1, n_estimators=40, max_depth=5)
        # svm.train(records=td, reduce_features=True)
        # print(svm.get_feature_weights())
        #svm.crossvalidate(records=td, n_jobs=4, reduce_features=True, n_features=10000, groups=True, n_replicates=1)
        xgb.crossvalidate_cc(records=td, comple_steps=10, conta_steps=10,
                             n_jobs=6, reduce_features=False)  # not required when running XGB
        cccv_results.append(xgb.cccv_result)
        compleconta_plot(xgb.cccv_result)


if __name__ == "__main__":
    test_svm()
    test_xgb()
