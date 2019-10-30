#
# Created by Lukas Lüftinger on 2/5/19.
#
import logging

logging.addLevelName(logging.DEBUG, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
logging.addLevelName(logging.INFO, "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.INFO))
logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))


def get_logger(initname, verb=False):
    """
    This function provides a logger to all scripts used in this project.

    :param initname: The name of the logger to show up in log.
    :param verb: Toggle verbosity
    :return: the finished Logger object.
    """
    logger = logging.getLogger(initname)
    if type(verb) is bool:
        logger.setLevel(logging.INFO if verb else logging.WARNING)
    else:
        logger.setLevel(verb)  # TODO: hacky shit
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verb else logging.WARNING)
    logstring = '\033[1;32m[%(asctime)s]\033[1;0m \033[1m%(name)s\033[1;0m - %(levelname)s - %(message)s'
    formatter = logging.Formatter(logstring, '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(ch)
    logger.propagate = False
    return logger
