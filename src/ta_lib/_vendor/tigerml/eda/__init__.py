"""Utility for exploratory data analysis."""
import os
from tigerml.core.utils import set_logger

# from .Visualiser import DataExplorer, Visualizer
# from .dashboard import Dashboard
from . import plotters
from .Analyser import Analyser, TSAnalyser
from .base import EDAReport
from .segmented import SegmentedEDAReport
from .time_series import SegmentedTSReport, TSReport

# Configure logger for the module
log_dir = os.environ.get("logger_path", None)
_LOGGER = set_logger(__name__, verbose=2, log_dir=log_dir)
_LOGGER.propagate = False

# from .. import get_from_config
#
#
# if get_from_config("TRACKING"):
#     try:
#         import rollbar as __rollbar
#         import sys as __sys
#         _data = {
#             "language": "Python",
#             "language_version": __sys.version,
#             "package_version": get_from_config("__version__"),
#             "os": __sys.platform,
#             "accelerator": "pde"
#         }
#
#         __rollbar.report_message('Imported Python Package', 'info', extra_data=_data)
#     except Exception as e:
#         pass
