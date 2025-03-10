import logging
import os
from tigerml.core.utils import set_logger

from .model_drift import ModelDrift
from .multiple_models import MultipleModelDrift
from .segmented import SegmentedModelDrift

# Configure logger for the module
log_dir = os.environ.get("logger_path", None)
_LOGGER = set_logger(__name__, verbose=2, log_dir=log_dir)
_LOGGER.propagate = False
