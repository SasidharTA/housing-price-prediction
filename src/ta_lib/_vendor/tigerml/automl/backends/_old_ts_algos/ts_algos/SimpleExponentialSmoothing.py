# coding: utf-8
"""
Created on Mon Sep 17 15:23:07 2018.

@author: ranjith.a
"""

import logging
import pandas as pd
import time
from multiprocessing.dummy import Pool as ThreadPool
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from .. import config as CONFIG
from .base import TimeSeriesModels

_LOGGER = logging.getLogger(__name__)


class SimpleExponentialSmoothing(TimeSeriesModels):
    """Simpleexponentialsmoothing class."""

    def __init__(self):
        TimeSeriesModels.__init__(self)
        self.algo = "SimpleExponentialSmoothing"
        self.hyperParams = self.createSearchSpace()

    #

    def createSearchSpace(self):
        """Creates search space for Simpleexponentialsmoothing class."""
        levelLst = CONFIG.hyperparams_simpleexponentialsmoothing["smoothing_level"]
        if len(levelLst) == 0:
            # [{'smoothing_level':val1} for val1 in [np.round(val,1) for val in np.linspace(0.1,1,10).tolist()]]
            levelLst = [None]
        paramLst = []
        paramLst = [{"smoothing_level": val} for val in levelLst]
        return paramLst

    #

    def startModelProcess(self):
        """Starts model process for Simpleexponentialsmoothing class."""
        #
        _LOGGER.info("starting model : " + self.algo)
        st = time.time()
        inpDataDF, trainDataDF, testDataDF = self.readData()
        #

        def runModel(paramDict):
            """Runs model for Simpleexponentialsmoothing class."""
            alpha = paramDict["smoothing_level"]
            model = SimpleExpSmoothing(trainDataDF[CONFIG.dv_variable_name])
            fit = model.fit(smoothing_level=alpha, optimized=True)
            trainDataDFPredicted, testDataDFPredicted = getPredictions(
                fit, trainDataDF, testDataDF
            )
            rmse_train, mape_train, mae_train, rsq_dummy = self.getAccuracyMetricsBase(
                trainDataDFPredicted[CONFIG.dv_variable_name],
                trainDataDFPredicted["predictions"],
            )
            rmse_test, mape_test, mae_test, rsq_dummy = self.getAccuracyMetricsBase(
                testDataDFPredicted[CONFIG.dv_variable_name],
                testDataDFPredicted["predictions"],
            )
            return [
                self.algo,
                {"smoothing_level": fit.params["smoothing_level"]},
                rmse_train,
                mape_train,
                mae_train,
                rsq_dummy,
                rmse_test,
                mape_test,
                mae_test,
            ]

        #

        def getPredictions(fit, trainDataDF, testDataDF):
            """Gets predictions for Simpleexponentialsmoothing class."""
            #
            trainDataDFPredicted = trainDataDF.copy()
            testDataDFPredicted = testDataDF.copy()
            #
            self.save_model(fit)
            #
            trainDataDFPredicted = trainDataDFPredicted.assign(predictions=fit.level)
            testDataDFPredicted = testDataDFPredicted.assign(
                predictions=fit.forecast(len(testDataDFPredicted)).tolist()
            )
            #
            return trainDataDFPredicted, testDataDFPredicted

        """
        # provide alpha value directionally if needed, initialize self.hyperParams to None, remove multi-threading.
        resultLst = []
        for val in self.hyperParams['smoothing_level']:
            resultLst.append(runModel(val))
        """
        pool = ThreadPool(2)
        resultLst = pool.map(runModel, self.hyperParams)
        pool.close()
        pool.join()
        _LOGGER.info("time taken :  %f minutes" % (((time.time() - st) / 60.0)))
        #
        return pd.DataFrame(
            resultLst,
            columns=[
                "algo",
                "hyperParams",
                "rmse_train",
                "mape_train",
                "mae_train",
                "rsqTrain",
                "rmse_test",
                "mape_test",
                "mae_test",
            ],
        )

    #
