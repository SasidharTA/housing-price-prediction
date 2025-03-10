import os

from .utils import set_logger

# setting the logger
log_dir = os.environ.get("logger_path", None)
_LOGGER = set_logger(__name__, verbose=2, log_dir=log_dir)
_LOGGER.propagate = False
