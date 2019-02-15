#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
import time
import logging
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np

from pica.io.io import load_training_files
from pica.ml.svm import PICASVM
from pica.transforms.resampling import TrainingRecordResampler
from pica.util.serialization import save_ml, load_ml


if __name__ == "__main__":
    td, gr, pr = load_training_files(genotype_file="pica/dev/Sulfate_reducer.genotype",
                                     phenotype_file="pica/dev/Sulfate_reducer.phenotype",
                                     verb=True)  # make training data set from genotype and phenotype files
    svm = PICASVM(verb=logging.DEBUG, random_state=2)
    # svm.train(records=td)
    # print("Predictions:", svm.predict(gr[-10:]))  # maybe change output format
    # print("Crossvalidation:", svm.crossvalidate(records=td))
    # svm.train(records=td)  # error due to being fitted already
    # save_ml(svm, filename="test_save.bin", verb=True, overwrite=True)
    # new_svm = load_ml("test_save.bin", verb=True)
    # print("Predictions:", new_svm.predict(gr[-10:]))
    #
    # lone_record = td[-1]
    # trr = TrainingRecordResampler(verb=False, random_state=2)  # when using random_state, this is fully reproducible
    # trr.fit(td)
    # resampled_record = trr.get_resampled(lone_record, comple=0.9, conta=0.1)
    #
    # print("New feature set size:", len(resampled_record.features))
    # print("Features in new but not in old feature set:",
    #       len(set(resampled_record.features) - set(lone_record.features)))
    #
    # # make single input function to compare speed of regular map with parallel map
    # # resample to super shitty quality
    # resampled_single = lambda x: trr.get_resampled(x, comple=0.2, conta=0.1)
    # t1 = time.time()
    # resampled_list = list(map(resampled_single, td))
    # t2 = time.time()
    #
    # print("Sequential duration of single resampling of full TrainingRecord list:", t2 - t1)
    #
    # # try Thread parallelization:
    # # slightly worse than non-parallelized. too much overhead.
    # t3 = time.time()
    # with ThreadPoolExecutor() as executor:
    #     resampled_list_parallel = list(executor.map(resampled_single, td))
    # t4 = time.time()
    #
    # print("Parallel duration of single resampling of full TrainingRecord list:", t4 - t3)
    #
    #
    # # train pica with bad quality training set:
    # third_svm = PICASVM(verb=True, random_state=2)
    # print("Crossvalidation:", svm.crossvalidate(records=resampled_list))  # crossvalidation bacc gets correctly shitty


    # try cv over comple/conta, unoptimized (parallelized over folds of single CVs, via sklearn)
    # svm.crossvalidate(td)
    #
    # svm.compress_vocabulary(td)
    # svm.crossvalidate(td)
    #svm.train(td)
    #cv_dict = svm.completeness_cv(td, cv=5)


    #print(cv_dict)  # takes 9 min on 6 Ryzen 5 cores

    # def print_weights():
    #     names, weights=svm.get_feature_weights()
    #     weights, names = zip(*sorted(zip(weights,names),reverse=True))
    #     count=0
    #     for name, weight in zip(names, weights):
    #         count+=1
    #         print(count,name.upper(),weight)
    #         if count == 20:
    #             break
    #
    #
    # print_weights()

    svm.crossvalidate_cc(records=td, comple_steps=3, conta_steps=3)
    pprint(svm.cccv_result)
