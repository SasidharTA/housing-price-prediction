"""Model evaluation and explanation utility.

All plots in this utility are built using holoviews library. Interactive plots will be rendered in jupyter and others refer `holoviews <https://hvplot.holoviz.org/user_guide/Viewing.html>`_
"""

import os
from tigerml.core.utils import set_logger

from .base import (
    ClassificationEvaluation,
    ClassificationReport,
    ModelInterpretation,
    RegressionEvaluation,
    RegressionReport,
)
from .comparison import ClassificationComparison, RegressionComparison
from .multi_model import (
    MultiModelComparisonClassification,
    MultiModelComparisonRegression,
)
from .plotters import *
from .segmented import ClassificationSegmentedReport, RegressionSegmentedReport

# Configure logger for the module
log_dir = os.environ.get("logger_path", None)
_LOGGER = set_logger(__name__, verbose=2, log_dir=log_dir)
_LOGGER.propagate = False
