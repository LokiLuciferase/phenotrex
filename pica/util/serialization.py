#
# Created by Lukas Lüftinger on 2/5/19.
#
import os
import logging

from sklearn.externals import joblib

from pica.util.logging import get_logger


def save_ml(obj, filename: str, overwrite=False, verb=False):
    """
    Save a PICA ml classifier as a pickled Python3 class. e.g. a fitted PICASVM object
    :param obj: the Python3 object to be saved.
    :param filename: Output filename
    :param overwrite: Overwrite existing files with same name
    :param verb: Toggle verbosity
    """
    logger = get_logger(initname=__name__, verb=verb)
    basefolder = os.path.dirname(os.path.abspath(filename))
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
    """
    Load a pickled PICA ml classifier to a usable object. e.g. a fitted PICASVM object
    :param filename: Input filename
    :param verb: Toggle verbosity
    :return: a unpickled PICA ml classifier
    """
    logger = get_logger(initname=__name__, verb=verb)
    if not os.path.isfile(filename):
        raise RuntimeError(f"Input file does not exist: {filename}")
    obj = joblib.load(filename)
    logger.info("Successfully loaded classifier.")
    return obj
