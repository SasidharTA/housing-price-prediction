# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 02:10:00 2018.

@author: ranjith.a
"""

import numpy as np
import pandas as pd
import pickle
from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error

from .. import config as CONFIG


class TimeSeriesModels(object):
    """Timeseriesmodels class initializer."""

    __metaclass__ = ABCMeta
    #

    def __init__(self, algo="", hyperParams={}):
        algo = algo
        hyperParams = hyperParams
        inpDF, trainDF, testDF = self.readData()

    #

    def readData(self):
        """Reads data for Timeseriesmodels class."""
        inpDF = pd.read_csv(CONFIG.data_location)
        inpDF["Date"] = pd.to_datetime(inpDF["Date"], infer_datetime_format=True)
        inpDF = inpDF.sort_values("Date", ascending=True)
        splitInd = int(len(inpDF) * 0.80)
        trainDF = inpDF[0:splitInd]
        testDF = inpDF[splitInd:]
        trainDF.index = trainDF.Date
        testDF.index = testDF.Date
        inpDF.index = inpDF.Date
        return inpDF, trainDF, testDF

    #

    @abstractmethod
    def startModelProcess(self):
        """Starts model process for Timeseriesmodels class."""
        pass

    #

    @abstractmethod
    def createSearchSpace(self):
        """Creates search space for Timeseriesmodels class."""
        pass

    #

    def getAccuracyMetricsBase(self, actuals, predictions):
        """Gets accuracy metrics for Timeseriesmodels class."""
        rmse = None
        if "rmse" in CONFIG.perf_metrics:
            rmse = np.round(np.sqrt(mean_squared_error(actuals, predictions)), 2)
        mape = None
        if "mape" in CONFIG.perf_metrics:
            mape = np.round(np.mean(np.abs((actuals - predictions) / actuals)) * 100, 2)
        mae = None
        if "mae" in CONFIG.perf_metrics:
            mae = np.round(np.sum(np.absolute(actuals - predictions)) / len(actuals), 2)
        rsq = None
        if "rsquared" in CONFIG.perf_metrics:
            ssTot = np.sum(pow(actuals - np.mean(actuals), 2))
            ssRes = np.sum(pow(actuals - predictions, 2))
            rsq = 1 - (ssRes / ssTot)
            # rsq in terms of variance
            # rsq = 1 -(pow(np.std(actuals - predictions),2) / pow(np.std(actuals),2))
        return rmse, mape, mae, rsq

    def save_model(self, model_object):
        """Saves model for Timeseriesmodels class."""
        with open(CONFIG.code_output_path + self.algo + "_fitted_model.pkl", "wb") as f:
            pickle.dump(model_object, f, protocol=2)

    def load_model(self):
        """Loads model for Timeseriesmodels class."""
        with open(CONFIG.code_output_path + self.algo + "_fitted_model.pkl", "rb") as f:
            model_object = pickle.load(f)
            return model_object
