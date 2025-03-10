# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:20:26 2021.

@authors: rithesh.sunku
"""

import copy

# Importing libraries
import logging
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import ElasticNet
from tigerml.automl.backends.ts_algos.TimeSeriesSplit import TimeSeriesSplit

from .base import TimeSeriesModels, flatten_dict, get_default_args

_LOGGER = logging.getLogger(__name__)


class ElasticNetTS(TimeSeriesModels):
    """Elasticnet related class for Time Series Models. Uses sklearn package.

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
    x_vars_dict : dict, optional
        The dictionary contains directionality of exog variables.
    """

    def __init__(self, x_vars_dict: dict = None, **kwargs):
        """The initialisation for ElasticNetTS class."""

        super().__init__(**kwargs)

        assert len(self.exog_cols) > 0, "No Exogenous Variables are provided"
        if x_vars_dict is None:
            print(
                "x vars expected sign not provied. \
                Assuming for exog cols to be positive if positive = True is set"
            )
            x_vars_dict_1 = {exog_col: "+" for exog_col in self.exog_cols}
            self.missing_x_vars_dict = {}
            self.x_vars_dict = copy.deepcopy(x_vars_dict_1)

        else:
            cols_missing_sign = [
                exog_col
                for exog_col in self.exog_cols
                if exog_col not in x_vars_dict.keys()
            ]
            if len(cols_missing_sign) > 0:
                print(
                    "Exogenous Columns with no expected sign specified. \
                      Assuming positve direction for them if positive = True is set"
                )

            missing_x_vars_dict = {}

            for missing_col in cols_missing_sign:
                print(missing_col)
                missing_x_vars_dict[missing_col] = "+"

            self.missing_x_vars_dict = missing_x_vars_dict
            x_vars_dict_1 = {**x_vars_dict, **missing_x_vars_dict}

            x_vars_dict_2 = {}
            for key_ in x_vars_dict.keys():
                x_vars_dict_2[key_] = x_vars_dict_1[key_]

            self.x_vars_dict = copy.deepcopy(x_vars_dict_2)

        self.master_kwargs = {"__init__": get_default_args(ElasticNet.__init__)}

        # self.ar_featuring_engineering = ar_fe_fn

    def runModel(self, model_data: pd.DataFrame, model_param_dict: dict):
        """
        Fits an Elastic Net Model to the input data.

        Parameters
        ----------
        model_data : pd.DataFrame
            Dataframe with timeseries data on which model to be fitted.
        model_param_dict : dict
            Dictionary of parameters for ElasticNet.

        Returns
        -------
        fitted_model : model object
            Fitted model object.
        """

        model_data_copy = model_data.copy()
        if model_param_dict["__init__"]["positive"]:
            if len(self.exog_cols) > 0:
                invert_sign_cols = [
                    col for col in self.exog_cols if self.x_vars_dict[col] == "-"
                ]
                model_data_copy.loc[:, invert_sign_cols] = model_data_copy.loc[
                    :, invert_sign_cols
                ].mul(-1)

        model = ElasticNet(**model_param_dict["__init__"])

        fitted_model = model.fit(
            X=model_data_copy[self.exog_cols], y=model_data_copy[self.endog_col]
        )
        return fitted_model

    def getPredictions(self, model_data: pd.DataFrame, model_object):
        """
        Generate predictions/scoring from the fitted model on the input data.

        Parameters
        ----------
        model_data : pd.DataFrame
            pandas Dataframe on which predictions to be done using the model_object

        model_object: elasticnet model object
            model object built from 'runModel' method

        Returns
        -------
        model_data_w_pred : pd.DataFrame
            Original dataframe with "predictions" column added to it.
        """
        model_data_copy = model_data.copy()

        if model_object.get_params()["positive"]:
            if len(self.exog_cols) > 0:
                invert_sign_cols = [
                    col for col in self.exog_cols if self.x_vars_dict[col] == "-"
                ]
                model_data_copy.loc[:, invert_sign_cols] = model_data_copy.loc[
                    :, invert_sign_cols
                ].mul(-1)

        model_data_w_pred = model_data.copy()
        model_data_w_pred["predictions"] = model_object.predict(
            model_data_copy[self.exog_cols]
        )

        return model_data_w_pred

    '''def getPredictions_sequential(self, model_data: pd.DataFrame, model_object):
        """
        This method to generate predictions/scoring.

        Parameters
        ----------
        model_data : pd.DataFrame
            pandas Dataframe on which scoring to be done using the model_object

        model_object: holtwinters model object
            model object built from 'runModel' method
        Returns
        ------
        model_data: pd.DataFrame with predictions

        """
        index_forecast = model_data.index.to_list()

        for index_ in range(len(index_forecast)):
            present_index = index_forecast[index_]
            forecast_temp = model_data.loc[[present_index]].copy()
            predictions = self.getPredictions(forecast_temp, model_object)[
                "predictions"
            ].iloc[0]

            model_data.loc[present_index, "predictions"] = predictions

            if (index_ + 1) < len(index_forecast):
                next_index = index_forecast[index_ + 1]
                model_data = self.ar_featuring_engineering(
                    model_data, next_index, present_index, predictions
                )

        return model_data'''

    def GridSearchCV(self, model_related_dict: dict):
        """
        Performs a grid search on hyperparameters for Elastic Net Model.

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

        st = time.time()

        master_kwargs = copy.deepcopy(self.master_kwargs)
        model_param_dict_ = copy.deepcopy(model_related_dict["param_dict"])

        assert (
            type(model_param_dict_) is dict
        ), "Error at model_param_dict not being a dict"

        hyperParams = self._model_param_check(
            master_kwargs=master_kwargs, model_param_dict=model_param_dict_
        )

        hyperParams_w_algo = copy.deepcopy(hyperParams)
        for iter_ in range(len(hyperParams)):
            base_config = copy.deepcopy(hyperParams[iter_])
            eg = flatten_dict(base_config)
            eg_adjusted_None = {
                i: None if eg[i] == "None" else eg[i] for i in eg.keys()
            }
            eg_adjusted_None = {
                i: np.nan if eg_adjusted_None[i] == "nan" else eg_adjusted_None[i]
                for i in eg_adjusted_None.keys()
            }
            for i in eg_adjusted_None.keys():
                temp_ = eg_adjusted_None[i]
                exec("base_config" + "['" + "']['".join(i.split("--")) + "'] = temp_")
            hyperParams_w_algo[iter_] = copy.deepcopy(base_config)

        timeseriessplit_ = TimeSeriesSplit(data=self.data, date_column=self.date_col)
        list_indices_tuple = timeseriessplit_.split(
            copy.deepcopy(model_related_dict["validation"])
        )

        metric_ = model_related_dict["validation"]["metric"]

        hyperParams_metrics_log = pd.DataFrame()
        for iter_ in range(len(hyperParams_w_algo)):
            hyperParams_iter = copy.deepcopy(hyperParams_w_algo[iter_])
            cv_metrics_log = pd.DataFrame()
            # print(iter_)

            for cv_index in range(len(list_indices_tuple)):
                cv_ = list_indices_tuple[cv_index]
                train_df, test_df = self._read_data_helper(cv_)

                model_ = self.runModel(
                    model_data=train_df.copy(),
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
