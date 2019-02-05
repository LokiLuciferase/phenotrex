#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from pica.io.io import load_training_files
from pica.ml.svm import PICASVM
from pica.transforms.resampling import TrainingRecordResampler
from pica.util.serialization import save_ml, load_ml


if __name__ == "__main__":
    td, gr, pr = load_training_files(genotype_file="pica/dev_scripts/Sulfate_reducer.genotype",
                                     phenotype_file="pica/dev_scripts/Sulfate_reducer.phenotype",
                                     verb=True)  # make training data set from genotype and phenotype files
    svm = PICASVM(verb=True, random_state=2)
    svm.train(records=td)
    print("Predictions:", svm.predict(gr[-10:]))  # maybe change output format
    print("Crossvalidation:", svm.crossvalidate(records=td))
    svm.train(records=td)  # error due to being fitted already
    save_ml(svm, filename="test_save.bin", verb=True, overwrite=True)
    new_svm = load_ml("test_save.bin", verb=True)
    print("Predictions:", new_svm.predict(gr[-10:]))

    lone_record = td[-1]
    trr = TrainingRecordResampler(verb=False, random_state=2)  # when using random_state, this is fully reproducible
    trr.fit(td)
    resampled_record = trr.get_resampled(lone_record, comple=0.9, conta=0.1)

    print("New feature set size:", len(resampled_record.features))
    print("Features in new but not in old feature set:",
          len(set(resampled_record.features) - set(lone_record.features)))

    # make single input function to compare speed of regular map with parallel map
    resampled_curried = lambda x: trr.get_resampled(x, 0.9, 0.1)
    t1 = time.time()
    resampled_list = list(map(resampled_curried, td))
    t2 = time.time()

    print("Sequential duration of single resampling of full TrainingRecord list:", t2 - t1)

    # try Thread parallelization:
    # slightly worse than non-parallelized. too much overhead.
    t3 = time.time()
    with ThreadPoolExecutor() as executor:
        resampled_list_parallel = list(executor.map(resampled_curried, td))
    t4 = time.time()

    print("Parallel duration of single resampling of full TrainingRecord list:", t4 - t3)
