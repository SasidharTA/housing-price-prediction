# coding: utf-8
"""
Created on Mon Sep 17 15:23:07 2018.

@author: ranjith.a
"""
import itertools
import logging
import numpy as np
import pandas as pd
import time
from pyramid.arima import auto_arima
from statsmodels.tsa.stattools import acf, adfuller, pacf

from .. import config as CONFIG
from .base import TimeSeriesModels

_LOGGER = logging.getLogger(__name__)


class SARIMAX(TimeSeriesModels):
    """Sarimax class."""

    def __init__(self):
        TimeSeriesModels.__init__(self)
        self.algo = "SARIMAX"
        self.hyperParams = self.createSearchSpace()

    #
    def createSearchSpace(self):
        """Creates search space for Sarimax class."""
        confDict = CONFIG.hyperparams_sarimax
        seasonalOrder = confDict["seasonalOrder"]
        #
        inpDataDF, trainDataDF, testDataDF = self.readData()
        newd = self.get_diff_lag_stationary_series(trainDataDF[CONFIG.dv_variable_name])
        newp, newq = self.searchARIMAparams(trainDataDF[CONFIG.dv_variable_name], newd)
        val_from_auto_arima = [
            0 if pd.notnull(confDict["p"][0]) else 1,
            0 if pd.notnull(confDict["d"][0]) else 1,
            0 if pd.notnull(confDict["q"][0]) else 1,
        ]

        p = confDict["p"] if pd.notnull(confDict["p"]) else [newp]
        d = confDict["d"] if pd.notnull(confDict["d"]) else [newd]
        q = confDict["q"] if pd.notnull(confDict["q"]) else [newq]

        seasonal_P_D_Q_dict = {"seasonal_cycle": seasonalOrder[3]}
        if pd.isnull(seasonal_P_D_Q_dict["seasonal_cycle"]):
            seasonal_P_D_Q_dict["seasonal_cycle"] = 1

        seasonal_P_D_Q_dict["p"] = (
            seasonalOrder[0] if pd.notnull(seasonalOrder[0]) else newp
        )
        seasonal_P_D_Q_dict["d"] = (
            seasonalOrder[1] if pd.notnull(seasonalOrder[1]) else newd
        )
        seasonal_P_D_Q_dict["q"] = (
            seasonalOrder[2] if pd.notnull(seasonalOrder[2]) else newq
        )
        seasonal_P_D_Q_dict["val_from_auto_arima"] = [
            0 if pd.notnull(seasonalOrder[0]) else 1,
            0 if pd.notnull(seasonalOrder[1]) else 1,
            0 if pd.notnull(seasonalOrder[2]) else 1,
        ]
        #
        paramLst = []
        for item in list(itertools.product(p, d, q)):
            paramLst.append(
                {
                    "p": item[0],
                    "d": item[1],
                    "q": item[2],
                    "seasonalOrder": seasonalOrder,
                    "val_from_auto_arima": val_from_auto_arima,
                    "seasonal_P_D_Q_dict": seasonal_P_D_Q_dict,
                }
            )
        return paramLst

    #
    def searchARIMAparams(self, ts_data, diff_lag):
        """Searches arima params for Sarimax class."""
        temp_data = pd.DataFrame(ts_data.copy())
        temp_data["diff"] = temp_data[CONFIG.dv_variable_name] - temp_data[
            CONFIG.dv_variable_name
        ].shift(diff_lag)
        temp_data = temp_data.dropna()
        if len(temp_data) <= 40:
            diff_lag = len(temp_data) - 1
        else:
            diff_lag = 40
        lag_acf = acf(temp_data["diff"], nlags=diff_lag, alpha=0.05)
        lag_acf_vals = lag_acf[0]
        lag_acf_bounds = pd.DataFrame(lag_acf[1], columns=["LB", "UB"])
        lag_acf_bounds["values"] = lag_acf_vals
        lag_pacf = pacf(temp_data["diff"], nlags=diff_lag, alpha=0.05, method="ols")
        lag_pacf_vals = lag_pacf[0]
        lag_pacf_bounds = pd.DataFrame(lag_pacf[1], columns=["LB", "UB"])
        lag_pacf_bounds["values"] = lag_pacf_vals
        lag_pacf_bounds = lag_pacf_bounds[
            (lag_pacf_bounds["values"] < lag_pacf_bounds["LB"])
            | (lag_pacf_bounds["values"] > lag_pacf_bounds["UB"])
        ]
        try:
            p = lag_pacf_bounds.index[-1]
        except Exception:
            p = 3
        lag_acf_bounds["values"] = lag_acf_vals
        lag_acf_bounds = lag_acf_bounds[
            (lag_acf_bounds["values"] < lag_acf_bounds["LB"])
            | (lag_acf_bounds["values"] > lag_acf_bounds["UB"])
        ]
        try:
            q = lag_acf_bounds.index[-1]
        except Exception:
            q = 3
        return p, q

    #
    def is_stationary(self, timeseries):
        """Checks if is stationary for Sarimax class."""
        # Perform Dickey-Fuller test:
        try:
            dftest = adfuller(timeseries, autolag="AIC")
        except Exception:
            dftest = adfuller(
                timeseries, autolag="AIC", maxlag=int(len(timeseries) / 2)
            )
        # dftest_p_val = float(dftest[1:2][0])
        dfoutput = pd.Series(dftest[0:2], index=["Test Statistic", "p-value"])
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
            # print(dfoutput)
        if dfoutput["p-value"] <= 0.05:
            return True
        return False

    #

    def get_diff_lag_stationary_series(self, timeseries):
        """Gets difference in lag for stationary series for Sarimax class."""
        temp_series = pd.DataFrame(timeseries.copy())
        max_iter = int(len(temp_series) / 2)
        for diff_lag in range(1, max_iter + 1):
            temp_series["diff"] = temp_series[CONFIG.dv_variable_name] - temp_series[
                CONFIG.dv_variable_name
            ].shift(diff_lag)
            stationary_flag = self.is_stationary(temp_series["diff"].dropna())
            # print("stationary_flag: {} for lag-{}".format(stationary_flag, diff_lag))
            if stationary_flag is True:
                return diff_lag
        return max_iter

    def get_updated_p_d_q_val(self, paramDict):
        """Gets updated p,d,q values for Sarimax class."""
        temp_dict = paramDict.copy()
        p = temp_dict["p"]
        d = temp_dict["d"]
        q = temp_dict["q"]
        if temp_dict["val_from_auto_arima"][0] == 1:
            p_upper = p
            p = 1
        else:
            p_upper = p
        if temp_dict["val_from_auto_arima"][2] == 1:
            q_upper = q
            q = 1
        else:
            q_upper = q
        if temp_dict["val_from_auto_arima"][1] == 1:
            d_upper = d
            d = None
        else:
            d_upper = d
            d = d
        return p, d, q, p_upper, d_upper, q_upper

    #

    def startModelProcess(self):
        """Starts model process for Sarimax class."""
        #
        _LOGGER.info("starting model : " + self.algo)
        st = time.time()
        inpDataDF, trainDataDF, testDataDF = self.readData()
        #

        def runModel(paramDict):
            """Runs model for Sarimax class."""
            p = paramDict["p"]
            d = paramDict["d"]
            q = paramDict["q"]
            seasonal_P_D_Q_dict = paramDict["seasonal_P_D_Q_dict"]
            seasonal_cycle = seasonal_P_D_Q_dict["seasonal_cycle"]
            p, d, q, p_upper, d_upper, q_upper = self.get_updated_p_d_q_val(paramDict)
            P, D, Q, P_upper, D_upper, Q_upper = self.get_updated_p_d_q_val(
                seasonal_P_D_Q_dict
            )
            # if all 3 (p,d,q) values not given then run complete stepwise auto-arima
            auto_arima_flag = np.sum(paramDict["val_from_auto_arima"])
            seasonal_auto_arima_flag = np.sum(
                seasonal_P_D_Q_dict["val_from_auto_arima"]
            )
            #
            fit = None
            if (auto_arima_flag == 3) and (seasonal_auto_arima_flag == 3):
                stepwise_flag = True
                max_order_val = 10
            else:
                stepwise_flag = False
                max_order_val = None
            # set d to 0 to handle the stationarity issue(if d_upper is too large)
            if d_upper >= 4:
                d = 0
            # print("stepwise_flag (auto-arima flag): ", stepwise_flag)
            # print("seasonal_cycle: ", seasonal_cycle)
            # print("lower-limit: p, d, q, P, D, Q: ", p, d, q, P, D, Q)
            # print("upper-limit: p, d, q, P, D, Q: ", p_upper,d_upper, q_upper, P_upper, D_upper, Q_upper)
            if len(CONFIG.idv_variable_names) > 0:
                fit = auto_arima(
                    np.asarray(trainDataDF[CONFIG.dv_variable_name]).astype(np.float64),
                    exogenous=trainDataDF[CONFIG.idv_variable_names],
                    start_p=p,
                    start_q=q,
                    max_p=p_upper,
                    max_q=q_upper,
                    m=seasonal_cycle,
                    start_P=P,
                    start_Q=Q,
                    max_P=P_upper,
                    max_Q=Q_upper,
                    seasonal=True,
                    stationary=False,
                    d=d,
                    max_d=d_upper,
                    D=D,
                    max_D=D_upper,
                    trace=False,
                    error_action="ignore",  # don't want to know if an order does not work
                    suppress_warnings=False,  # don't want convergence warnings
                    stepwise=stepwise_flag,  # set to stepwise
                    max_order=max_order_val,
                )  # ignoring the sum of p, q values limits
            else:
                fit = auto_arima(
                    np.asarray(trainDataDF[CONFIG.dv_variable_name]).astype(np.float64),
                    start_p=p,
                    start_q=q,
                    max_p=p_upper,
                    max_q=q_upper,
                    m=seasonal_cycle,
                    start_P=P,
                    start_Q=Q,
                    max_P=P_upper,
                    max_Q=Q_upper,
                    seasonal=True,
                    stationary=False,
                    d=d,
                    max_d=d_upper,
                    D=D,
                    max_D=D_upper,
                    trace=False,
                    error_action="ignore",  # don't want to know if an order does not work
                    suppress_warnings=False,  # don't want convergence warnings
                    stepwise=stepwise_flag,  # set to stepwise
                    max_order=max_order_val,
                )  # ignoring the sum of p, q values limits
            # print(trainDataDF[CONFIG.idv_variable_names].head())
            # print(fit.summary())
            arima_params = fit.get_params()["order"]
            seasonal_order = fit.get_params()["seasonal_order"]
            p = arima_params[0]
            d = arima_params[1]
            q = arima_params[2]
            trainDataDFPredicted, testDataDFPredicted = getPredictions(
                fit, trainDataDF, testDataDF
            )
            rmse_train, mape_train, mae_train, rsqTrain = self.getAccuracyMetricsBase(
                trainDataDFPredicted[CONFIG.dv_variable_name],
                trainDataDFPredicted["predictions"],
            )
            rmse_test, mape_test, mae_test, rsqTestDummy = self.getAccuracyMetricsBase(
                testDataDFPredicted[CONFIG.dv_variable_name],
                testDataDFPredicted["predictions"],
            )
            resultsParams = {
                "ar.L" + str(lag): fit.arparams()[lag - 1] for lag in range(1, p + 1)
            }
            if p == 0:
                resultsParams["ar.L"] = "no p-coef"
            resultsParams.update(
                {"ma.L" + str(lag): fit.maparams()[lag - 1] for lag in range(1, q + 1)}
            )
            if q == 0:
                resultsParams["ma.L"] = "no q-coef"
            seasonal_D = seasonal_order[0] + seasonal_order[2]
            if seasonal_D < 1:
                resultsParams["ar_ma.S.L"] = "no seasonal coef"
            else:
                temp_list = fit.params()[
                    (len(CONFIG.idv_variable_names) + p + q + 1) : -1
                ]
                resultsParams.update(
                    {
                        "ar.S.L" + str(lag * seasonal_order[3]): temp_list[lag - 1]
                        for lag in range(1, seasonal_order[0] + 1)
                    }
                )
                temp_list = temp_list[seasonal_order[0] :]
                resultsParams.update(
                    {
                        "ma.S.L" + str(lag * seasonal_order[3]): temp_list[lag - 1]
                        for lag in range(1, seasonal_order[2] + 1)
                    }
                )
            resultsParams["series_differencing"] = d
            resultsParams["seasonal_differencing"] = seasonal_order[1]
            resultsParams["AIC"] = fit.aic()
            resultsParams["BIC"] = fit.bic()
            return [
                self.algo,
                resultsParams,
                rmse_train,
                mape_train,
                mae_train,
                rsqTrain,
                rmse_test,
                mape_test,
                mae_test,
            ]

        #
        def getPredictions(fit, trainDataDF, testDataDF):
            """Gets predictions for Sarimax class."""
            #
            trainDataDFPredicted = trainDataDF.copy()
            testDataDFPredicted = testDataDF.copy()
            #
            self.save_model(fit)
            #
            trainDataDFPredicted["predictions"] = fit.predict_in_sample(
                exogenous=trainDataDFPredicted[CONFIG.idv_variable_names]
            )
            testDataDFPredicted = testDataDFPredicted.assign(
                predictions=fit.predict(
                    n_periods=len(testDataDFPredicted),
                    exogenous=testDataDFPredicted[CONFIG.idv_variable_names],
                )
            )
            #
            return trainDataDFPredicted, testDataDFPredicted

        resultLst = []
        for val in self.hyperParams:
            resultLst.append(runModel(val))
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
