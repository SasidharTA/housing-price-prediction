# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:20:26 2021.

@authors: revanth.battaram, rithesh.sunku
"""

import copy
import logging
import pandas as pd
import time
from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    Holt,
    SimpleExpSmoothing,
)
from tigerml.automl.backends.ts_algos.TimeSeriesSplit import TimeSeriesSplit

from .base import TimeSeriesModels, get_default_args

_LOGGER = logging.getLogger(__name__)


class ExponentialSmoothingRelated(TimeSeriesModels):
    """Class for choosing and running exponential smoothing related algorithms for TimeSeries. Uses statsmodels package.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Dataframe containing the timeseries data, by default pd.DataFrame()
    date_col : str, optional
        Column name for date, by default ""
    endog_col : str, optional
        Column name for endogenous variable, by default ""
    exog_cols : list, optional
        Column names for exogenous variables, by default []
    """

    def __init__(self, **kwargs):
        """The initialisation for ExponentialSmoothingRelated class with with kwargs."""

        super().__init__(**kwargs)

        self.algo_choice_dict = {
            "SimpleExponentialSmoothing": SimpleExpSmoothing,
            "ExponentialSmoothingHolt": Holt,
            "ExponentialSmoothingHoltWinters": ExponentialSmoothing,
        }

        master_kwargs_simple = {
            "fit": get_default_args(SimpleExpSmoothing.fit),
            "__init__": get_default_args(SimpleExpSmoothing.__init__),
        }
        master_kwargs_holt = {
            "fit": get_default_args(Holt.fit),
            "__init__": get_default_args(Holt.__init__),
        }

        master_kwargs_holt_winters = {
            "fit": get_default_args(ExponentialSmoothing.fit),
            "__init__": get_default_args(ExponentialSmoothing.__init__),
        }

        self.kwargs_choice_dict = {
            "SimpleExponentialSmoothing": master_kwargs_simple,
            "ExponentialSmoothingHolt": master_kwargs_holt,
            "ExponentialSmoothingHoltWinters": master_kwargs_holt_winters,
        }

    def runModel(self, model_data: pd.DataFrame, model_param_dict: dict):
        """
        Fits the chosen exponential smoothing model to the input data.

        Parameters
        ----------
        model_data: pd.DataFrame
            DataFrame with the timeseries data to fit the model.

        model_param_dict: dict
            Dictionary of parameters for the chosen model, including "algorithm" key and "fit" and "__init__" keys for parameters to pass to the fit and __init__ methods respectively.

        Returns
        -------
        fitted_model : model object
            Fitted model object.
        """
        algorithm = model_param_dict["algorithm"]
        model_param_dict["__init__"]["endog"] = model_data[self.endog_col]
        model = self.algo_choice_dict[algorithm](**model_param_dict["__init__"])
        fitted_model = model.fit(**model_param_dict["fit"])

        return fitted_model

    def getPredictions(self, model_data: pd.DataFrame, model_object):
        """
        Generate predictions/scoring from the fitted model on the input data.

        Parameters
        ----------
        model_data : pd.DataFrame
            pandas Dataframe on which predictions to be done using the model_object

        model_object: exponential smoothing model object
            model object built from 'runModel' method

        Returns
        -------
        model_data_w_pred : pd.DataFrame
            Original dataframe with "predictions" column added to it.
        """
        model_data_w_pred = model_data.copy()
        model_data_w_pred["predictions"] = model_object.predict(
            start=model_data_w_pred.index.min(), end=model_data_w_pred.index.max()
        )
        return model_data_w_pred

    def GridSearchCV(self, model_related_dict: dict):
        """
        Performs a grid search for the input parameters and models.

        This method tunes for the right parameters and build out the model for the right parameters.

        Parameters
        ----------
        model_related_dict : dict
            Dictionary containing "param_dict" key with parameters for the models to search over and "validation" key with the validation method to use.

        Returns
        -------
        hyperParams_metrics_log : pd.DataFrame
            Dataframe of grid search results with columns for parameters, metrics and algorithm.
        """

        algorithm_ = (
            model_related_dict["param_dict"]["algorithm"]
            if model_related_dict["param_dict"]["algorithm"] is not None
            else "SimpleExponentialSmoothing"
        )

        _LOGGER.info("starting model : " + algorithm_)

        st = time.time()

        master_kwargs = self.kwargs_choice_dict[algorithm_]
        model_param_dict_ = copy.deepcopy(model_related_dict["param_dict"][algorithm_])

        assert (
            type(model_param_dict_) is dict
        ), "Error at model_param_dict not being a dict"

        hyperParams = self._model_param_check(
            master_kwargs=master_kwargs, model_param_dict=model_param_dict_
        )

        hyperParams_w_algo = []

        for iter_ in range(len(hyperParams)):
            hyperParams_w_algo.append(
                {**hyperParams[iter_], **{"algorithm": algorithm_}}
            )

        timeseriessplit_ = TimeSeriesSplit(data=self.data, date_column=self.date_col)
        list_indices_tuple = timeseriessplit_.split(
            copy.deepcopy(model_related_dict["validation"])
        )

        metric_ = model_related_dict["validation"]["metric"]

        hyperParams_metrics_log = pd.DataFrame()
        for iter_ in range(len(hyperParams_w_algo)):
            hyperParams_iter = hyperParams_w_algo[iter_]
            cv_metrics_log = pd.DataFrame()

            for cv_index in range(len(list_indices_tuple)):
                cv_ = list_indices_tuple[cv_index]
                train_df, test_df = self._read_data_helper(cv_)

                model_ = self.runModel(
                    model_data=train_df,
                    model_param_dict=copy.deepcopy(hyperParams_iter),
                )

                train_df_w_pred = self.getPredictions(
                    model_data=train_df.copy(), model_object=model_
                )
                test_df_w_pred = self.getPredictions(
                    model_data=test_df.copy(), model_object=model_
                )

                train_metrics = self.getAccuracyMetricsBase(
                    train_df_w_pred[self.endog_col], train_df_w_pred["predictions"]
                )
                test_metrics = self.getAccuracyMetricsBase(
                    test_df_w_pred[self.endog_col], test_df_w_pred["predictions"]
                )

                metrics = {
                    **{"cv_index": [cv_index]},
                    **{
                        "train_" + key_: [train_metrics[key_]]
                        for key_ in train_metrics.keys()
                    },
                    **{
                        "test_" + key_: [test_metrics[key_]]
                        for key_ in test_metrics.keys()
                    },
                }

                cv_metrics_log = pd.concat(
                    [cv_metrics_log, pd.DataFrame(metrics)], ignore_index=True
                )

            agg_metrics = {
                "iter": [iter_],
                "params": [str(hyperParams_iter)],
                "train_metric": [cv_metrics_log["train_" + metric_].mean()],
                "test_metric": [cv_metrics_log["test_" + metric_].mean()],
            }
            hyperParams_metrics_log = pd.concat(
                [hyperParams_metrics_log, pd.DataFrame(agg_metrics)], ignore_index=True
            )

        return hyperParams_metrics_log
