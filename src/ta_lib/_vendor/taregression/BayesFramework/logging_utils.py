"""logging_utils module returns a logger object with console and file log handlers."""

import logging
import os
import time

dt = time.strftime("%b-%d-%Y_%H%M", time.localtime())


LOG_LEVEL = logging.DEBUG

LOG_FORMAT = logging.Formatter(
    "%(asctime)s | "
    "%(filename)s | "
    "%(funcName)s | "
    "%(levelname)s | "
    "line %(lineno)d :: "
    "%(message)s"
)


def add_console_handler(logger):
    """
    Function to include a console handler(logs printing in console).

    Parameters
    ----------
    logger: Predefined logger object

    Returns
    -------
    logging.Logger object with console handler
    """
    ch = logging.StreamHandler()
    ch.setFormatter(LOG_FORMAT)
    logger.addHandler(ch)
    return logger


def add_file_handler(logger, log_file, log_level):
    """
    Function to include a file handler(logs saving in log file).

    Parameters
    ----------
    logger:
        Predefined logger object
    log_file: str
        Path to the log file for logs to be stored
    log_level:
        One of `[logging.DEBUG, logging.WARNING, logging.ERROR]`

    Returns
    -------
    logging.Logger object with console handler
    """
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    fh.setFormatter(LOG_FORMAT)
    logger.addHandler(fh)
    return logger


def get_logger(name):
    """Function to setup configurations of logger."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    logger = add_console_handler(logger)  # console log

    return logger
