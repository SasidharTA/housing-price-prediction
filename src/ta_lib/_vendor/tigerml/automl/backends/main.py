# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:23:07 2018.

@author: ranjith.a
"""
import config as CONFIG
import logging
import os
import pandas as pd
import warnings

if "LSTMSeqToSeqMultivariate" in CONFIG.algorithms:
    from LSTMSeqToSeqMultivariate import (  # avoiding costly import if not needed
        LSTMSeqToSeqMultivariate,
    )
if "SARIMAX" in CONFIG.algorithms:
    from SARIMAX import SARIMAX
if "SimpleExponentialSmoothing" in CONFIG.algorithms:
    from SimpleExponentialSmoothing import SimpleExponentialSmoothing
if "ExponentialSmoothingHolt" in CONFIG.algorithms:
    from ExponentialSmoothingHolt import ExponentialSmoothingHolt
if "ExponentialSmoothingHoltWinters" in CONFIG.algorithms:
    from ExponentialSmoothingHoltWinters import ExponentialSmoothingHoltWinters


warnings.filterwarnings("ignore")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ignore special warning
_LOGGER = logging.getLogger(__name__)


class ModelFactory(object):
    """ModelFactory class initializer."""

    @staticmethod
    def createModelObject(modelName):
        """Assigns modelName string to class object - class initializer."""
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


# In[]
if __name__ == "__main__":
    algosLst = CONFIG.algorithms
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

temp_df.drop(["rsqTrain"], axis=1).to_csv(
    CONFIG.code_output_path + "Time_Series_Model_Summary.csv", index=False
)
