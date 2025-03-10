# coding: utf-8
"""
Created on Mon Sep 17 15:23:07 2018.

@author: ranjith.a
"""
import itertools
import logging
import pandas as pd
import time
from multiprocessing.dummy import Pool as ThreadPool
from statsmodels.tsa.holtwinters import Holt

from .. import config as CONFIG
from .base import TimeSeriesModels

_LOGGER = logging.getLogger(__name__)


class ExponentialSmoothingHolt(TimeSeriesModels):
    """Exponentialsmoothingholt class initializer."""

    def __init__(self):
        TimeSeriesModels.__init__(self)
        self.algo = "ExponentialSmoothingHolt"
        self.hyperParams = self.createSearchSpace()

    #

    def createSearchSpace(self):
        """Creates search space for Exponentialsmoothingholt class."""
        levelLst = CONFIG.hyperparams_exponentialsmoothingholt["smoothing_level"]
        trendLst = CONFIG.hyperparams_exponentialsmoothingholt["smoothing_slope"]
        #
        if len(levelLst) == 0:
            levelLst = [
                None
            ]  # [np.round(val,1) for val in np.linspace(0.1,1,10).tolist()]
        if len(trendLst) == 0:
            trendLst = [
                None
            ]  # [np.round(val,1) for val in np.linspace(0.1,1,10).tolist()]
        #
        paramLst = []
        for item in list(itertools.product(levelLst, trendLst)):
            paramLst.append({"smoothing_level": item[0], "smoothing_slope": item[1]})
        return paramLst

    #

    def startModelProcess(self):
        """Starts model process for Exponentialsmoothingholt class."""
        #
        _LOGGER.info("starting model : " + self.algo)
        st = time.time()
        inpDataDF, trainDataDF, testDataDF = self.readData()
        #

        def runModel(paramDict):
            """Runs model for Exponentialsmoothingholt class."""
            alpha = paramDict["smoothing_level"]
            beta = paramDict["smoothing_slope"]
            model = Holt(trainDataDF[CONFIG.dv_variable_name])
            fit = model.fit(smoothing_level=alpha, smoothing_slope=beta, optimized=True)
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
            # _LOGGER.info('alpha : %f, sse : %d, rmse_train : %f, mape_train : %f, rmse_test : %f, mape_test : %f'%(alpha,fit.sse,rmse_train, mape_train, rmse_test, mape_test))
            return [
                self.algo,
                {
                    "smoothing_level": fit.params["smoothing_level"],
                    "smoothing_slope": fit.params["smoothing_slope"],
                },
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
            """Gets predictions for Exponentialsmoothingholt class."""
            #
            trainDataDFPredicted = trainDataDF.copy()
            testDataDFPredicted = testDataDF.copy()
            #
            self.save_model(fit)
            #
            trainDataDFPredicted["predictions"] = fit.fittedvalues
            testDataDFPredicted = testDataDFPredicted.assign(
                predictions=fit.forecast(len(testDataDFPredicted)).tolist()
            )
            #
            return trainDataDFPredicted, testDataDFPredicted

        """
        # Provide alpha value directionally if needed, initialize self.hyperParams to None, remove multi-threading.
        ResultLst = []
        For val in self.hyperParams['smoothing_level']:
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
