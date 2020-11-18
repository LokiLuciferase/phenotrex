#
# Created by Lukas Lüftinger on 2/5/19.
#
import os
import sys

import joblib

import phenotrex
from phenotrex.util.logging import get_logger


def save_classifier(obj, filename: str, overwrite=False, verb=False):
    """
    Save a TrexClassifier as a pickled object.

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


def load_classifier(filename: str, verb=False):
    """
    Load a pickled TrexClassifier to a usable object.

    :param filename: Input filename
    :param verb: Toggle verbosity
    :return: a unpickled PICA ml classifier
    """
    logger = get_logger(initname=__name__, verb=verb)
    if not os.path.isfile(filename):
        raise RuntimeError(f"Input file does not exist: {filename}")
    obj = joblib.load(filename)
    logger.info(f"Successfully loaded classifier (feature_type={obj.feature_type}).")
    return obj
