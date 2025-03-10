import logging
import pandas as pd

from . import config as CONFIG

if "LSTMSeqToSeqMultivariate" in CONFIG.algorithms:
    from .ts_algos.LSTMSeqToSeqMultivariate import (  # avoiding costly import if not needed
        LSTMSeqToSeqMultivariate,
    )
if "SARIMAX" in CONFIG.algorithms:
    from .ts_algos.SARIMAX import SARIMAX
if "SimpleExponentialSmoothing" in CONFIG.algorithms:
    from .ts_algos.SimpleExponentialSmoothing import SimpleExponentialSmoothing
if "ExponentialSmoothingHolt" in CONFIG.algorithms:
    from .ts_algos.ExponentialSmoothingHolt import ExponentialSmoothingHolt
if "ExponentialSmoothingHoltWinters" in CONFIG.algorithms:
    from .ts_algos.ExponentialSmoothingHoltWinters import (
        ExponentialSmoothingHoltWinters,
    )

_LOGGER = logging.getLogger(__name__)


class ModelFactory(object):
    """Modelfactory class initializer."""

    @staticmethod
    def createModelObject(modelName):
        """Assigns modelName string to class object."""
        if modelName == "SimpleExponentialSmoothing":
            return SimpleExponentialSmoothing()
        elif modelName == "ExponentialSmoothingHolt":
            return ExponentialSmoothingHolt()
        elif modelName == "ExponentialSmoothingHoltWinters":
            return ExponentialSmoothingHoltWinters()
        elif modelName == "SARIMAX":
            return SARIMAX()
        elif modelName == "LSTMSeqToSeqMultivariate":
            return LSTMSeqToSeqMultivariate()
        else:
            _LOGGER.info(modelName + " - implementation does not exist")


class Forecaster:
    """Forecaster class."""

    def __init__(self):
        """Forecaster class initializer."""
        self.config = CONFIG
        self.results = None

    def set_config(self):
        """Sets the config."""
        return self

    def get_config(self):
        """Returns config."""
        return self.config

    def fit(self):
        """Fits the model."""
        algosLst = self.config.algorithms
        final_summary_list = [pd.DataFrame()]
        lstm_result = pd.DataFrame()
        for algo in algosLst:
            modelObj = ModelFactory().createModelObject(algo)
            resultDF = modelObj.startModelProcess()
            if algo == "LSTMSeqToSeqMultivariate":
                lstm_result = resultDF
            else:
                final_summary_list.append(resultDF)
        temp_df = pd.concat(final_summary_list, axis=0)
        temp_df = pd.concat(
            [temp_df.reset_index(drop=True), lstm_result.reset_index(drop=True)], axis=0
        )
        self.results = temp_df
        return self

    def score(self):
        """Forecaster class initializer."""
        return self

    def get_report(self):
        """Saves report to csv file."""
        self.results.drop(["rsqTrain"], axis=1).to_csv(
            CONFIG.code_output_path + "Time_Series_Model_Summary.csv", index=False
        )
