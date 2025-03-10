# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:20:26 2021.

@authors: rithesh.sunku
"""

import copy
import logging
import pandas as pd
import time
from tigerml.automl.backends.ts_algos.TimeSeriesSplit import TimeSeriesSplit
from xgboost import XGBModel

from .base import flatten_dict, get_default_args
from .ElasticNetRelated import ElasticNetTS

_LOGGER = logging.getLogger(__name__)


class XGBoostTS(ElasticNetTS):
    """A class for XGBoost time series models.

    This class extends the ElasticNetTS class and uses the XGBoost library to fit and make predictions for time series data.

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

    def __init__(self, **kwargs):
        """The initialisation for XGBoostTS class."""

        super().__init__(**kwargs)

        self.master_kwargs = {"__init__": get_default_args(XGBModel.__init__)}

        x_vars_dict_modified = {}
        for key_ in self.x_vars_dict.keys():
            if key_ in self.missing_x_vars_dict.keys():
                x_vars_dict_modified[key_] = 0
            else:
                if self.x_vars_dict[key_] == "+":
                    x_vars_dict_modified[key_] = 1
                elif self.x_vars_dict[key_] == "-":
                    x_vars_dict_modified[key_] = -1
        self.x_vars_dict = copy.deepcopy(x_vars_dict_modified)

    def runModel(self, model_data: pd.DataFrame, model_param_dict: dict):
        """
        Fits an XGBoost model to the input data.

        Parameters
        ----------
        model_data : pd.DataFrame
            Dataframe with timeseries data on which model to be fitted.
        model_param_dict : dict
            Dictionary of parameters for XGBoost.

        Returns
        -------
        fitted_model : model object
            Fitted model object.
        """
        model_data_copy = model_data.copy()
        model = XGBModel(**model_param_dict["__init__"])
        fitted_model = model.fit(
            X=model_data_copy[self.exog_cols], y=model_data_copy[self.endog_col]
        )
        return fitted_model

    def getPredictions(self, model_data: pd.DataFrame, model_object):
        """
        Generate predictions/scoring from the fitted XGBoost model on the input data.

        Parameters
        ----------
        model_data : pd.DataFrame
            pandas Dataframe on which scoring to be done using the model_object

        model_object: XGBoost model object
            model object built from 'runModel' method

        Returns
        -------
        model_data_w_pred : pd.DataFrame
            Original dataframe with "predictions" column added to it.
        """
        model_data_w_pred = model_data.copy()
        model_data_w_pred["predictions"] = model_object.predict(
            model_data_w_pred[self.exog_cols]
        )

        return model_data_w_pred
