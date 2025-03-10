# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:20:26 2021.

@authors: rithesh.sunku
"""
import copy
import json
import logging
import os
import pandas as pd
import time
import warnings
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from prophet.utilities import regressor_coefficients

# from fbprophet import Prophet
# from fbprophet.serialize import model_from_json, model_to_json
# from fbprophet.utilities import regressor_coefficients
from tigerml.automl.backends.ts_algos.TimeSeriesSplit import TimeSeriesSplit

from .base import TimeSeriesModels, flatten_dict, get_default_args

warnings.filterwarnings("ignore")

_LOGGER = logging.getLogger(__name__)

logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


class FBProphet(TimeSeriesModels):
    """A time series prediction model based on Facebook's Prophet library. Uses prophet package.

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
    holidays_data : pd.DataFrame, optional
        Dataframe containing the timeseries holiday data, by default is None.
    """

    def __init__(self, holidays_data: pd.DataFrame = None, **kwargs):
        """The initialisation for FBProphet class."""

        super().__init__(**kwargs)

        self.holidays_data = holidays_data
        self.master_kwargs = {"__init__": get_default_args(Prophet.__init__)}

    def runModel(self, model_data: pd.DataFrame, model_param_dict: dict):
        """
        Fit the Prophet model to the input data.

        Parameters
        ----------
        model_data : pd.DataFrame
            Dataframe with timeseries data on which model to be fitted.
        model_param_dict : dict
            Dictionary of parameters for Prophet model.

        Returns
        -------
        fitted_model : model object
            Fitted model object.
        """
        model = Prophet(**model_param_dict["__init__"])
        if len(self.exog_cols) > 0:
            for col in self.exog_cols:
                model.add_regressor(col)
        model_data_copy = model_data.copy()
        model_data_copy.rename(
            columns={self.date_col: "ds", self.endog_col: "y"}, inplace=True
        )
        fitted_model = model.fit(df=model_data_copy)

        return fitted_model

    def getPredictions(self, model_data: pd.DataFrame, model_object):
        """
        Generate predictions/scoring from the fitted prophet model on the input data.

        Parameters
        ----------
        model_data : pd.DataFrame
            pandas Dataframe on which predictions to be done using the model_object

        model_object: prophet model object
            model object built from 'runModel' method

        Returns
        -------
        model_data_w_pred : pd.DataFrame
            Original dataframe with "predictions" column added to it.
        """
        model_data_w_pred = model_data.copy()
        model_data_w_pred.rename(columns={self.date_col: "ds"}, inplace=True)
        predictions = model_object.predict(model_data_w_pred[["ds"] + self.exog_cols])[
            ["ds", "yhat"]
        ]
        predictions["ds"] = predictions["ds"].dt.strftime("%Y-%m-%d")

        assert (
            predictions.shape[0] == model_data_w_pred.shape[0]
        ), "Lengths are not equal"

        model_data_w_pred["ds"] = model_data_w_pred["ds"].astype(str)
        model_data_w_pred = model_data_w_pred.merge(predictions, on=["ds"], how="left")

        model_data_w_pred.rename(
            columns={"ds": self.date_col, "yhat": "predictions"}, inplace=True
        )

        return model_data_w_pred

    '''def getPredictionComponents(self, model_data: pd.DataFrame, model_object):
        """
        This method to generate predictions/scoring.

        Parameters
        ----------
        model_data : pd.DataFrame
            pandas Dataframe on which scoring to be done using the model_object

        model_object: holtwinters model object
            model object built from 'runModel' method
        Return
        ------
        model_data: pd.DataFrame with predictions and components

        """
        model_data_w_pred = model_data.copy()
        model_data_w_pred.rename(columns={self.date_col: "ds"}, inplace=True)

        return model_object.predict(model_data_w_pred[["ds"] + self.exog_cols])'''

    def getCoefficients(self, model_object):
        """
        Get the coefficients of the fitted prophet model object.

        Parameters
        ----------
        model_object: Prophet model object
            The fitted Prophet model object.

        Returns
        -------
        coefficients: pd.DataFrame
            A dataframe containing the coefficients of the fitted prophet model.

        """

        return regressor_coefficients(model_object)

    def GridSearchCV(self, model_related_dict: dict):
        """
        Performs a grid search on hyperparameters for the Prophet Model.

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
            hyperParams_iter["__init__"]["holidays"] = (
                self.holidays_data
                if hyperParams_iter["__init__"]["holidays"] is not None
                else hyperParams_iter["__init__"]["holidays"]
            )
            cv_metrics_log = pd.DataFrame()

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

            hyperParams_iter["__init__"]["holidays"] = (
                "used_holidays_data"
                if hyperParams_iter["__init__"]["holidays"] is not None
                else hyperParams_iter["__init__"]["holidays"]
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

    def save_model(
        self,
        model_object,
        save_path: str = os.getcwd(),
        model_nm: str = "_fitted_model",
    ):
        """
        Save the fitted model object to a specified filepath.

        Parameters
        ----------
        model_object: model object
            prophet model object generated after fitting the model
        save_path: str, optional
            path at which model object to be saved, by default current working directory.
        model_nm : str, optional
            name by which to save in specified path, by default "_fitted_model".
        """
        output_path = os.path.join(save_path, "prophet")
        isExist = os.path.exists(output_path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(output_path)
            print("The new directory is created!")

        with open(os.path.join(output_path, model_nm + ".json"), "w") as fout:
            json.dump(model_to_json(model_object), fout)

    def load_model(self, load_path: str = os.getcwd(), model_nm: str = "_fitted_model"):
        """Load a saved model object from a specified filepath.

        Parameters
        ----------
        load_path: str
            The filepath of the saved model object, by default current working directory.
        model_nm : str, optional
            name by which model is saved, by default takes as "_fitted_model".

        Returns
        -------
        model_object: object
            The loaded Prophet model object.
        """
        with open(os.path.join(load_path, "prophet", model_nm + ".json"), "r") as fin:
            model_object = model_from_json(json.load(fin))
        return model_object
