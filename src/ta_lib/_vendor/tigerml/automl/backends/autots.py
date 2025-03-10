import ast
import copy
import logging
import pandas as pd
from numpy import nan

from . import config as CONFIG
from .ts_algos.base import TimeSeriesModels  # noqa: F401
from .ts_algos.ElasticNetRelated import ElasticNetTS
from .ts_algos.ExponentialSmoothingRelated import ExponentialSmoothingRelated
from .ts_algos.ProphetRelated import FBProphet
from .ts_algos.SARIMAX import ARIMAXRelated
from .ts_algos.TimeSeriesSplit import TimeSeriesSplit  # noqa: F401
from .ts_algos.XGBRelated import XGBoostTS

_LOGGER = logging.getLogger(__name__)


class ModelFactory:
    """Modelfactory class initializer."""

    @staticmethod
    def createModelObject(modelName, data, date_col, endog_col, exog_cols):
        """
        Assigns `modelName` string to a class object.

        Parameters
        ----------
        modelName : str
            name of the algorithm for which model to be created
        data : pandas dataframe
            the data on which the model is to be fit
        date_col : str
            the name of the date column in the dataframe
        endog_col : str
            the name of the target variable column
        exog_cols : list
            list of column names of the explanatory variables

        Returns
        -------
        object
            An instance of the class corresponding to the `modelName` passed.

        """

        model_map = {
            "SimpleExponentialSmoothing": ExponentialSmoothingRelated,
            "ExponentialSmoothingHolt": ExponentialSmoothingRelated,
            "ExponentialSmoothingHoltWinters": ExponentialSmoothingRelated,
            "SARIMAX": ARIMAXRelated,
            "elasticnet": ElasticNetTS,
            "prophet": FBProphet,
            "xgboost": XGBoostTS,
        }

        model_class = model_map.get(modelName)
        if model_class is None:
            _LOGGER.info(modelName + " - implementation does not exist")
            return
        return model_class(
            data=data, date_col=date_col, endog_col=endog_col, exog_cols=exog_cols
        )


class Forecaster:
    """Forecaster class for automatic time-series modeling.

    This class is used to fit the best time-series model and make predictions.

    Examples
    --------
    Download the sample data ("his_tvr.csv") from `here <https://drive.google.com/file/d/1ZJPEtmSqEizW2ARin1EmNO7FXqXsDjRR/view?usp=sharing>`__.

    >>> import pandas as pd
    >>> from tigerml.automl.backends.autots import Forecaster

    >>> # Load the timeseries data
    >>> df = pd.read_csv('his_tvr.csv')
    >>> df = df.drop(columns=['Rch´000 {Av(Wg)}', 'user_rating'])

    >>> # Split the data into train and test datasets
    >>> from tigerml.automl.backends.autots import TimeSeriesSplit
    >>> data_split = TimeSeriesSplit(
    ...    data=df, date_column="Date"
    ... )
    >>> split_index = data_split._split_with_test_perc(test_perc=0.2)
    >>> train_df = df.iloc[split_index[0][0]]
    >>> test_df = df.iloc[split_index[0][1]]

    >>> # Initialize Forecaster() class object
    >>> forecaster = Forecaster()

    >>> print(forecaster.get_config()) # prints the default configs

    >>> # fit the best timeseries model
    >>> final_model = forecaster.fit(
    ...     data=train_df, date_col= "Date",
    ...     endog_col= "TVR", exog_cols= ['Channel']
    ... )

    >>> # Generate the predictions
    >>> test_df_w_pred = forecaster.score(
    ...    model_data=test_df
    ... )
    >>> test_df_w_pred.head()
    """

    def __init__(self):
        """Forecaster class initializer."""
        self.data = None
        self.date_col = None
        self.endog_col = None
        self.exog_cols = None
        self.config = CONFIG.dafault_config
        self.results = None
        self.final_modelObj = None
        self.final_model = None
        self.model_factory = ModelFactory()

    def _update_dict(self, original_dict, new_dict):
        """
        Helper function to update the original dictionary with new dictionary.

        Parameters
        ----------
        original_dict : dict
            original dictionary to be updated
        new_dict : dict
            dictionary containing the updated values

        Returns
        -------
        dict
            the updated original dictionary
        """
        for key, value in new_dict.items():
            if isinstance(value, dict):
                original_dict[key] = self._update_dict(
                    original_dict.get(key, {}), value
                )
            else:
                original_dict[key] = value
        return original_dict

    def set_config(self, config_params=None):
        """
        Sets the config.

        Parameters
        ----------
        config_params : dict, optional
            The new config parameters to update the existing config, by default None

        Returns
        -------
        dict
            Returns the updated config

        Examples
        --------
        >>> from tigerml.automl.backends.autots import Forecaster
        >>> # Defining config_dict for running specific algorithms or updating any parameters
        >>> config_dict = {
        ...     "algorithms": ["SimpleExponentialSmoothing","ExponentialSmoothingHolt"],
        ...     "validation_hyperparams": {
        ...         "metric": "rmse"
        ...    },
        ...    "model_hyperparams": {
        ...        "SimpleExponentialSmoothing": {
        ...            "fit": {
        ...                "smoothing_level":[0.4]
        ...            }
        ...        }
        ...    }
        ... }
        >>> forecaster = Forecaster()
        >>> # To update the default config params
        >>> updated_configs = forecaster.set_config(config_params = config_dict)
        >>> # Printing the updated_configs to check
        >>> print(updated_configs)

        """
        if config_params is not None:
            self.config = self._update_dict(self.config, config_params)
            print("updated config params")
        return self.config

    def get_config(self):
        """
        Returns config.

        Returns
        -------
        dict
            config parameters

        """

        return self.config

    def prep_data(self):
        """
        Data processing and splitting.

        Returns
        -------
        self
        """
        return self

    def accuracyMetrics(self):
        """Generates accuracy metrics like rmse, mape, mae, mse."""
        return self

    def fit(self, data, date_col, endog_col, exog_cols=[]):
        """
        Fits the best model.

        Parameters
        ----------
        data : pd.DataFrame
            Data on which the model is to be fit.
        date_col : str
            The name of the date column in the dataframe
        endog_col : str
            The name of the target variable column
        exog_cols : list, optional
            List of column names of the explanatory variables, by default []

        Returns
        -------
        model_object
            final fitted model or object.

        """
        self.data = data
        self.date_col = date_col
        self.endog_col = endog_col
        self.exog_cols = exog_cols
        algosLst = self.config["algorithms"]
        final_params_list = []
        for algo in algosLst:
            modelObj = self.model_factory.createModelObject(
                algo, self.data, self.date_col, self.endog_col, self.exog_cols
            )

            model_params = {
                "algorithm": algo,
                algo: self.config["model_hyperparams"][algo],
            }

            if algo in ["elasticnet", "prophet", "xgboost"]:
                model_params = self.config["model_hyperparams"][algo]

            validation_params = self.config["validation_hyperparams"]

            hyperParams_df = modelObj.GridSearchCV(
                model_related_dict={
                    "validation": validation_params,
                    "param_dict": model_params,
                }
            )
            hyperParams_df["algorithm"] = algo
            final_params_list.append(hyperParams_df)

        final_params = pd.concat(final_params_list, ignore_index=False)

        best_index = final_params["test_metric"].idxmin()
        final_algo = final_params["algorithm"].iloc[best_index]

        if final_algo == "xgboost":
            final_model_params = eval(final_params["params"].iloc[best_index])
        else:
            final_model_params = ast.literal_eval(
                final_params["params"].iloc[best_index]
            )

        self.final_modelObj = self.model_factory.createModelObject(
            final_algo,
            self.data,
            self.date_col,
            self.endog_col,
            self.exog_cols,
        )

        self.final_model = self.final_modelObj.runModel(
            model_data=self.data, model_param_dict=final_model_params
        )
        print("Final Fitted model: " + final_algo)
        return self.final_model

    def score(self, model_data: pd.DataFrame):
        """
        Generates predictions/scoring.

        Parameters
        ----------
        model_data : pd.DataFrame
            Data on which predictions are to be generated.

        Returns
        -------
        pd.DataFrame
            Dataframe containing original data along with the generated predictions.

        """
        model_object = self.final_model
        model_class = self.final_modelObj
        model_data_w_pred = model_class.getPredictions(model_data, model_object)

        return model_data_w_pred
