#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
import numpy as np

from pica.io.io import load_training_files
from pica.ml.svm import PICASVM
from pica.transforms.resampling import TrainingRecordResampler
from pica.util.serialization import save_ml, load_ml


if __name__ == "__main__":
    td, gr, pr = load_training_files(genotype_file="pica/dev_scripts/Sulfate_reducer.genotype",
                             phenotype_file="pica/dev_scripts/Sulfate_reducer.phenotype",
                             verb=True) # make training data set from genotype and phenotype files
    # svm = PICASVM(verb=True, random_state=2)
    # svm.train(records=td)
    # print("Predictions:", svm.predict(gr[-10:]))  # maybe change output format
    # print("Crossvalidation:", svm.crossvalidate(records=td))
    # svm.train(records=td)  # error due to being fitted already
    # save_ml(svm, filename="test_save.bin", verb=True, overwrite=True)
    # new_svm = load_ml("test_save.bin", verb=True)
    # print("Predictions:", new_svm.predict(gr[-10:]))

    lone_record = td[-1]
    trr = TrainingRecordResampler(verb=True, random_state=None)  # when using random_state, this is fully reproducible
    trr.fit(td)
    resampled_record = trr.get_resampled(lone_record, comple=0.9, conta=0.1)

    print("New feature set size:", len(resampled_record.features))
    print("Features in new but not in old feature set:",
          len(set(resampled_record.features) - set(lone_record.features)))
