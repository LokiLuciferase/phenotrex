#
# Created by Lukas Lüftinger on 2/5/19.
#
import os
import logging

from sklearn.externals import joblib

from pica.util.logging import get_logger

def save_ml(obj, filename: str, overwrite=False, verb=False):
    logger = get_logger(initname=__name__, loglevel=logging.INFO if verb else logging.WARNING)
    basefolder = os.path.dirname(filename)
    if not os.path.exists(basefolder):
        raise RuntimeError(f"Output folder does not exist: {basefolder}")
    if os.path.isfile(filename):
        if overwrite:
            logger.warning("Overwriting existing file.")
        else:
            raise RuntimeError("Output file exists.")
    logger.info("Begin saving classifier...")
    joblib.dump(obj, filename=filename)
    logger.info("Classifier saved.")

def load_ml(filename: str, verb=False):
    logger = get_logger(initname=__name__, loglevel=logging.INFO if verb else logging.WARNING)
    if not os.path.isfile(filename):
        raise RuntimeError(f"Input file does not exist: {filename}")
    obj = joblib.load(filename)
    logger.info("Successfully loaded classifier.")
    return obj
