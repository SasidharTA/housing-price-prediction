# coding: utf-8
"""
Created on Mon Sep 17 15:23:07 2018.

@author: ranjith.a
"""
import itertools
import logging
import pandas as pd
import time
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .. import config as CONFIG
from .base import TimeSeriesModels

_LOGGER = logging.getLogger(__name__)


class ExponentialSmoothingHoltWinters(TimeSeriesModels):
    """Exponentialsmoothingholtwinters class."""

    def __init__(self):
        TimeSeriesModels.__init__(self)
        self.algo = "ExponentialSmoothingHoltWinters"
        self.hyperParams = self.createSearchSpace()

    #
    def createSearchSpace(self):
        """Creates search space for Exponentialsmoothingholtwinters class."""
        trendType = CONFIG.hyperparams_exponentialsmoothingholtwinters["trend_type"]
        seasonalType = CONFIG.hyperparams_exponentialsmoothingholtwinters[
            "seasonal_type"
        ]
        seasonalPeriods = CONFIG.hyperparams_exponentialsmoothingholtwinters[
            "seasonal_periods"
        ]

        smoothingLevelLst = CONFIG.hyperparams_exponentialsmoothingholtwinters[
            "smoothing_level"
        ]
        smoothingSlopeLst = CONFIG.hyperparams_exponentialsmoothingholtwinters[
            "smoothing_slope"
        ]
        smoothingSeasonalLst = CONFIG.hyperparams_exponentialsmoothingholtwinters[
            "smoothing_seasonal"
        ]
        #
        if len(trendType) == 0:
            trendType = [None]
        if len(seasonalType) == 0:
            seasonalType = [None]
        if len(seasonalPeriods) == 0:
            seasonalPeriods = [None]
        if len(smoothingLevelLst) == 0:
            # [np.round(val,1) for val in np.linspace(0.1,1,10).tolist()]
            smoothingLevelLst = [None]
        if len(smoothingSlopeLst) == 0:
            # [np.round(val,1) for val in np.linspace(0.1,1,10).tolist()]
            smoothingSlopeLst = [None]
        if len(smoothingSeasonalLst) == 0:
            # [np.round(val,1) for val in np.linspace(0.1,1,10).tolist()]
            smoothingSeasonalLst = [None]
        #
        paramLst = []
        for item in list(
            itertools.product(
                trendType,
                seasonalType,
                seasonalPeriods,
                smoothingLevelLst,
                smoothingSlopeLst,
                smoothingSeasonalLst,
            )
        ):
            paramLst.append(
                {
                    "trendType": item[0],
                    "seasonalType": item[1],
                    "seasonalPeriods": item[2],
                    "smoothingLevel": item[3],
                    "smoothingSlope": item[4],
                    "smoothingSeasonal": item[5],
                }
            )
        return paramLst

    #
    def startModelProcess(self):
        """Exponentialsmoothingholtwinters class."""
        #
        _LOGGER.info("starting model : " + self.algo)
        st = time.time()
        inpDataDF, trainDataDF, testDataDF = self.readData()

        #
        def runModel(paramDict):
            """Runs model for Exponentialsmoothingholtwinters class."""
            trendType = paramDict["trendType"]
            seasonalType = paramDict["seasonalType"]
            seasonalPeriods = paramDict["seasonalPeriods"]
            smoothingLevel = paramDict["smoothingLevel"]
            smoothingSlope = paramDict["smoothingSlope"]
            smoothingSeasonal = paramDict["smoothingSeasonal"]
            #
            model = ExponentialSmoothing(
                trainDataDF[CONFIG.dv_variable_name],
                seasonal=seasonalType,
                trend=trendType,
                seasonal_periods=seasonalPeriods,
            )
            fit = model.fit(
                smoothing_level=smoothingLevel,
                smoothing_slope=smoothingSlope,
                smoothing_seasonal=smoothingSeasonal,
                optimized=True,
            )
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
                {
                    "trendType": trendType,
                    "seasonalType": seasonalType,
                    "seasonalPeriods": seasonalPeriods,
                    "smoothing_level": fit.params["smoothing_level"],
                    "smoothing_slope": fit.params["smoothing_slope"],
                    "smoothing_seasonal": fit.params["smoothing_seasonal"],
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
            """Gets predictions for Exponentialsmoothingholtwinters class."""
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

        # provide alpha value directionally if needed, initialize self.hyperParams to None, remove multi-threading.
        resultLst = []
        for val in self.hyperParams:
            resultLst.append(runModel(val))
        """
        pool = ThreadPool(1)
        resultLst = pool.map(runModel, self.hyperParams)
        pool.close()
        pool.join()
        """

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
