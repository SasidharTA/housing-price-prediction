"""
Created on Wed Sep 19 02:10:00 2018.

Latest script for the base code file contains model parameter check
and read_data_helper function to split the dataframe based on indices
and also deepstate related function

@authors: revanth.battaram, rithesh.sunku
"""

# Importing library

import inspect
import itertools
import numpy as np
import os
import pandas as pd
import pickle
from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_default_args(func):
    """
    Get default args of a function.

    Parameters
    ----------
    func : function
        The function for which to retrieve default arguments.

    Returns
    -------
    dict
        A dictionary of the function's parameters, where the keys are the parameter names and the values are lists containing the default value of the parameter, if one exists. If a parameter does not have a default value, the value in the dictionary is None. The parameter "self" is excluded from the dictionary.
    """
    signature = inspect.signature(func)
    return {
        k: [v.default] if v.default is not inspect.Parameter.empty else [None]
        for k, v in signature.parameters.items()
        if k != "self"
    }


def flatten_dict(dd, separator="--", prefix=""):
    """
    Flatten a nested dictionary.

    Parameters
    ----------
    dd : dict
        The nested dictionary to be flattened.
    separator : str, optional
        The separator used to separate the keys in the flattened dictionary, by default "--".
    prefix : str, optional
        The prefix for the keys in the flattened dictionary, by default "".

    Returns
    -------
    dict
        A flattened version of the input dictionary where all nested keys are concatenated with separator and prefix.
    """
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


class TimeSeriesModels(object):
    """
    This class serves as an abstract base class for time series modeling.

    It contains helper functions for defining the grid of hyperparameters and evaluating
    the accuracy of the models using metrics like RMSE, MAPE, MAE, and R2 score..

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

    __metaclass__ = ABCMeta

    def __init__(
        self,
        data: pd.DataFrame = pd.DataFrame(),
        date_col: str = "",
        endog_col: str = "",
        exog_cols: list = [],
    ):
        self.data = data
        self.date_col = date_col
        self.endog_col = endog_col
        self.exog_cols = exog_cols

    @abstractmethod
    def get_data_frequency(self):
        """Get data frequency."""
        frequencies = ["Daily", "Weekly", "Monthly", "Yearly"]
        data_ = self.data[[self.date_col]].copy()
        data_.drop_duplicates(subset=self.date_col, inplace=True)
        data_.sort_values(by=self.date_col, inplace=True)
        days = (data_[self.date_col].iloc[1] - data_[self.date_col].iloc[0]).days
        date_map = {
            1: "Daily",
            7: "Weekly",
            30: "Monthly",
            31: "Monthly",
            28: "Monthly",
            29: "Monthly",
            365: "Yearly",
            366: "Yearly",
        }
        try:
            frequency = date_map[days]
        except:
            frequency = "Daily"
        return days, frequencies[frequencies.index(frequency) :]

    @abstractmethod
    def GridSearchCV(self, model_related_dict: dict = None):
        """An abstract method that is meant to be overridden by the subclass to implement time series modeling."""
        pass

    def _model_param_check(self, master_kwargs: dict, model_param_dict: dict):
        """
        Model Param check.

        This method verifies the model params by comparing against.
        the kwargs of the respective method. It asserts that user supplied correct
        arguments by comparing against the all possible arguments
        This method also creates a grid of possible parameters
        Future work : creating a grid based on continuous distributions

        Parameters
        ----------
        master_kwargs : dict
            This is the default dict defined in the method '__init__' of the current class

        model_param_dict : dict
            This is the user supplied model parameters it is a dictionary of dictionaries which has parameter values for all the methods of the simple exponential smoothing

        Returns
        -------
        verified_model_param_dict : dict
        """
        assert set(model_param_dict.keys()).issubset(
            set(master_kwargs.keys())
        ), "keys supplied by the user for the model_param_dict must be valid"

        verified_model_param_dict = {}
        for m_key in master_kwargs.keys():
            if m_key in model_param_dict.keys():
                assert set(model_param_dict[m_key].keys()).issubset(
                    (set(master_kwargs[m_key].keys()))
                ), f"keys supplied by the user for the method {m_key} does not exist"

                for s_key in master_kwargs[m_key].keys():
                    if s_key not in model_param_dict[m_key].keys():
                        model_param_dict[m_key][s_key] = master_kwargs[m_key][s_key]
                verified_model_param_dict[m_key] = self.createSearchSpace(
                    model_param_dict[m_key]
                )

        return self.createSearchSpace(verified_model_param_dict)

    def createSearchSpace(self, params_dict: dict):
        """
        This method creates a parameter grid based on the values provided as a list in the dict.

        Parameters
        ----------
        params_dict : dict
            This is a dictionary with keys being the key word arguments and values being a list

        Returns
        -------
        possible_params_grid : list
            This is the list with each element being a dictionary comprising the kwargs for a method

        """
        possible_params_grid = [
            dict(zip(params_dict.keys(), v))
            for v in itertools.product(*params_dict.values())
        ]

        return possible_params_grid

    def _rmse_(self, actuals, predictions):
        """
        Compute Root Mean Squared Error.

        Parameters
        ----------
        self : object
            The object of the class
        actuals : array-like
            The actual values
        predictions : array-like
            The predicted values

        Returns
        -------
        float
            The Root Mean Squared Error rounded to 2 decimal points
        """
        return np.round(np.sqrt(mean_squared_error(actuals, predictions)), 2)

    def _mape_(self, actuals, predictions):
        """
        Compute Mean Absolute Percentage Error.

        Parameters
        ----------
        self : object
            The object of the class
        actuals : array-like
            The actual values
        predictions : array-like
            The predicted values

        Returns
        -------
        float
            The Mean Absolute Percentage Error rounded to 2 decimal points
        """

        return np.round(
            np.mean(np.abs((actuals - predictions) / (actuals + 1e-15))) * 100, 2
        )

    def _mae_(self, actuals, predictions):
        """
        Compute Mean Absolute Error.

        Parameters
        ----------
        self : object
            The object of the class
        actuals : array-like
            The actual values
        predictions : array-like
            The predicted values

        Returns
        -------
        float
            The Mean Absolute Error rounded to 2 decimal points
        """
        return np.round(mean_absolute_error(actuals, predictions), 2)

    def _r2_(self, actuals, predictions):
        """
        Compute R-squared score.

        Parameters
        ----------
        self : object
            The object of the class
        actuals : array-like
            The actual values
        predictions : array-like
            The predicted values

        Returns
        -------
        float
            The R-squared score rounded to 2 decimal points
        """
        return np.round(r2_score(actuals, predictions), 2)

    def getAccuracyMetricsBase(self, actuals, predictions):
        """
        Compute accuracy metrics for Timeseriesmodels class.

        Parameters
        ----------
        self : object
            The object of the class
        actuals : array-like
            The actual values
        predictions : array-like
            The predicted values

        Returns
        -------
        dict
            A dictionary containing the accuracy metrics (rmse, mape, mae, r2) and their values rounded to 2 decimal points.
        """

        name_func_dict = {
            "rmse": self._rmse_,
            "mape": self._mape_,
            "mae": self._mae_,
            "r2": self._r2_,
        }

        metrics_values = {
            key: name_func_dict[key](actuals, predictions)
            for key in name_func_dict.keys()
        }
        return metrics_values

    def _read_data_helper(self, split_indices_tuple: tuple):
        """
        This method accepts a dataframe which comprises all the indices for train and test split and returns the train and test dataframe.

        Parameters
        ----------
        split_indices_tuple : tuple
            It is the tuple with index 0 corresponding to train indices, and index 1, test indices

        Returns
        -------
        train_df : pd.DataFrame
        test_df : pd.DataFrame

        """
        train_df = self.data.iloc[split_indices_tuple[0]]
        test_df = self.data.iloc[split_indices_tuple[1]]

        return train_df, test_df

    def save_model(
        self,
        model_object,
        save_path: str = os.getcwd(),
        model_pkl_nm: str = "_fitted_model",
    ):
        """
        Saves model for Timeseriesmodels class.

        Parameters
        ----------
        model_object: model object generated after fitting the model
        save_path: str
            path at which model object to be saved
        model_pkl_nm: str
            name of model pickle object

        """
        algorithm = "Custom" if self.algo == "" else self.algo
        output_path = os.path.join(save_path, algorithm)
        isExist = os.path.exists(output_path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(output_path)
            print("The new directory is created!")
        with open(os.path.join(output_path, model_pkl_nm + ".pkl"), "wb") as f:
            pickle.dump(model_object, f, protocol=2)

    def load_model(
        self, load_path: str = os.getcwd(), model_pkl_nm: str = "_fitted_model.pkl"
    ):
        """
        Load a pre-trained model for Timeseriesmodels class.

        Parameters
        ----------
        self : object
            The object of the class
        load_path : str, optional
            The directory path where the model is located, by default os.getcwd()
        model_pkl_nm : str, optional
            The name of the model file, by default "_fitted_model.pkl"

        Returns
        -------
        object
            The model object that is loaded
        """
        algorithm = "Custom" if self.algo == "" else self.algo
        with open(os.path.join(load_path, algorithm, model_pkl_nm), "rb") as f:
            model_object = pickle.load(f)
            return model_object
