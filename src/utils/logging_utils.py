import logging
import os
import sys


def setup_logger(log_fpath: str):
    log_dir = os.path.dirname(log_fpath)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Remove existing log file if it exists
    if os.path.exists(log_fpath):
        os.remove(log_fpath)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    # formatter = logging.Formatter("%(message)s")

    # Create console handler and set level to debug
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler and set level to debug
    fh = logging.FileHandler(log_fpath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def close_logger(logger):
    """Close the logger with removing the handlers."""
    handlers = logger.handlers[:]

    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
