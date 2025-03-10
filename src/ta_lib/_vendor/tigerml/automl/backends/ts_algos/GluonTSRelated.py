# coding: utf-8
"""
Created on Mon Sep 17 15:23:07 2018.

@author: Mohammed Naseef
"""

# TODO : GluonTSRelated module is currently having some dependencies issues and no yet tested. Need to test it throughly and update the dependencies before using it.

import copy
import itertools
import json
import logging
import numpy as np
import pandas as pd
import time
from gluonts.dataset import common
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model import deepar, deepstate, prophet
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.mx.trainer import Trainer
from tigerml.automl.backends.ts_algos.TimeSeriesSplit import TimeSeriesSplit

from .base import TimeSeriesModels, flatten_dict, get_default_args

_LOGGER = logging.getLogger(__name__)


class GluonTSRelated(TimeSeriesModels):
    """GluonTSRelated Class for deepAR, deepState Models."""

    def __init__(
        self,
        ts_identifier_cols: list = [],
        prediction_length: int = 1,
        json_specs: dict = {"real": [], "cat": []},
        **kwargs
    ):
        """Initialization of GluonTSRelated Class.

        Parameters
        ----------
        ts_identifier_cols : list
            List of columns thet represent the indetifier
        prediction_length : int, optional
            Maximum prediction length
        json_specs : dict, optional
            Dictionary contains categorical and continuous variable information
        """

        super().__init__(**kwargs)
        self.algo_choice_dict = {
            "DeepAR": DeepAREstimator,
            "deepstate": DeepStateEstimator,
        }

        self.ts_identifier_cols = ts_identifier_cols
        self.prediction_length = prediction_length
        self.json_specs = json_specs
        self.data_freq = self.get_data_frequency()

        master_kwargs_deepAR = {
            "Estimator": get_default_args(DeepAREstimator),
            "Trainer": get_default_args(Trainer),
        }
        master_kwargs_deepstate = {
            "Estimator": get_default_args(DeepStateEstimator),
            "Trainer": get_default_args(Trainer),
        }

        self.kwargs_choice_dict = {
            "DeepAR": master_kwargs_deepAR,
            "deepstate": master_kwargs_deepstate,
        }

    def get_gluonts_data(self, data):
        """
        This function prepares the data for the model training using gluonts.dataset.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataframe containing the timeseries data, by default pd.DataFrame()

        Returns
        -------
        data : pd.DataFrame
            Dataframe to be consumed by GluonTS model

        """
        data.set_index(self.date_col, inplace=True)
        data_gluon = PandasDataset.from_long_dataframe(
            data,
            item_id="concat_ts_identifier",
            target=self.endog_col,
            feat_dynamic_real=self.json_specs["real"],
            feat_static_cat=self.json_specs["cat"],
        )
        return data_gluon

    def train_test_split(self):
        """This function splits the data into train and test data."""

        data = self.data.groupby(self.ts_identifier_cols).filter(
            lambda x: len(x) > self.prediction_length
        )
        data_sorted = self.data.sort_values(by=self.date_col)
        train_data = (
            data_sorted.groupby(self.ts_identifier_cols)
            .apply(lambda x: x.iloc[: -self.prediction_length])
            .reset_index(drop=True)
        )
        train_data_gluon = self.get_gluonts_data(train_data)
        test_data_gluon = self.get_gluonts_data(data_sorted)
        return train_data_gluon, test_data_gluon

    def prepare_gluonts_data(self, df: pd.DataFrame = pd.DataFrame()):
        """Function to prepare Gluonts dataset.

        Parameters
        ----------
        data : pd.DataFrame, optional
             Dataframe containing the timeseries data, by default pd.DataFrame()

        Returns
        -------
         data : dictionary
             data in dictionary format to be consumed by GluonTS model
        """

        if ("ts_identifier" not in df.columns) & (len(self.ts_identifier_cols) != 0):
            df["ts_identifier"] = (
                df[self.ts_identifier_cols].astype(str).apply("_".join, axis=1)
            )

        # df["ts_identifier"] = df["ts_identifier"].astype("category").cat.codes.values

        df.sort_values(["ts_identifier", self.date_col], inplace=True)

        # Preparing Metadata
        dynamic_real = []
        # dynamic_cat = []
        static_real = []
        static_cat = []

        prediction_length = self.prediction_length
        time_period = (
            df[self.date_col].sort_values().unique()[-self.prediction_length :]
        )
        time_period = pd.to_datetime(time_period)
        input_dict = {}
        input_dict["Date variable"] = self.date_col
        input_dict["Target variable"] = self.endog_col
        input_dict["Data granularity"] = "ts_identifier"
        input_dict["Forecast window"] = self.prediction_length
        input_dict["Data frequency"] = self.data_freq
        input_dict["Forecast start date"] = min(time_period).strftime("%Y-%m-%d")
        input_dict["Forecast end date"] = max(time_period).strftime("%Y-%m-%d")

        real = self.json_specs["real"]
        cat = self.json_specs["cat"]

        feat_static_cat = []
        # feat_dynamic_cat = []
        feat_static_real = []
        feat_dynamic_real = []

        # Feat Static and Dynamic Cat
        for col in cat:
            if not sum(df.groupby("ts_identifier")[col].nunique() > 1):
                feat_static_cat.append(col)
            else:
                feat_dynamic_real.append(col)

        # Feat Static and Dynamic Real
        for col in real:
            if not sum(df.groupby("ts_identifier")[col].nunique() > 1):
                feat_static_real.append(col)
            else:
                feat_dynamic_real.append(col)

        for col in feat_static_real:
            static_real.append({"name": col})

        for col in feat_dynamic_real:
            dynamic_real.append({"name": col})

        for col in feat_static_cat:
            static_cat.append({"name": col, "cardinality": df[col].nunique()})

        feat_dynamic_real = sorted(feat_dynamic_real)
        idv_cols = [i for i in feat_dynamic_real]
        list_cols = [self.date_col, self.endog_col] + idv_cols
        df_idv = (
            df.groupby(["ts_identifier"] + self.ts_identifier_cols)
            .agg({k: list if i else min for i, k in enumerate(list_cols)})
            .reset_index()
        )
        df_idv.columns = (
            ["item_id"] + self.ts_identifier_cols + ["Start_Date", "Actual"] + idv_cols
        )

        metadata = {}
        metadata["freq"] = self.data_freq
        metadata["prediction_length"] = prediction_length
        metadata["feat_dynamic_real"] = dynamic_real
        metadata["feat_static_cat"] = static_cat
        metadata["feat_static_real"] = static_real

        # Lablencoding categorical columns
        for col in feat_static_cat:
            df[col] = df[col].astype("category").cat.codes.values

        feat = {self.date_col: min, self.endog_col: list}
        feat.update({i: list for i in feat_dynamic_real})

        df = df.groupby("ts_identifier").agg(feat)

        if feat_dynamic_real:
            df["dynamic_real"] = df[feat_dynamic_real].values.tolist()

        # if feat_dynamic_cat:
        #     df['dynamic_cat'] = df[feat_dynamic_cat].values.tolist()

        if feat_static_cat:
            df["static_cat"] = (
                df.groupby("ts_identifier")
                .agg({i: "first" for i in feat_static_cat})
                .values.tolist()
            )

        if feat_static_real:
            df["static_real"] = (
                df.groupby("ts_identifier")
                .agg({i: "first" for i in feat_static_real})
                .values.tolist()
            )

        all_variables = np.array(
            [1, 1, 1]
            + [
                1 if feat_dynamic_real else 0,
                1 if feat_static_cat else 0,
                1 if feat_static_real else 0,
            ]
        )

        all_variable_names = np.array(
            [
                "item_id",
                "start",
                "target",
                "feat_dynamic_real",
                "feat_static_cat",
                "feat_static_real",
            ]
        )

        all_variable = all_variable_names[all_variables == 1]

        # Comb 1
        if len(all_variable) == 3:
            test_ds = [
                json.dumps(
                    {
                        all_variable[0]: str(ID),
                        all_variable[1]: str(start_date),
                        all_variable[2]: target,
                    }
                )
                for ID, (start_date, target, *_) in df.iterrows()
            ]

        # Comb 2
        if len(all_variable) == 4:
            test_ds = [
                json.dumps(
                    {
                        all_variable[0]: str(ID),
                        all_variable[1]: str(start_date),
                        all_variable[2]: target,
                        all_variable[3]: f1,
                    }
                )
                for ID, (start_date, target, *_, f1) in df.iterrows()
            ]

        # Comb 3
        if len(all_variable) == 5:
            test_ds = [
                json.dumps(
                    {
                        all_variable[0]: str(ID),
                        all_variable[1]: str(start_date),
                        all_variable[2]: target,
                        all_variable[3]: f1,
                        all_variable[4]: f2,
                    }
                )
                for ID, (start_date, target, *_, f1, f2) in df.iterrows()
            ]

        # Comb 4
        if len(all_variable) == 6:
            test_ds = [
                json.dumps(
                    {
                        all_variable[0]: str(ID),
                        all_variable[1]: str(start_date),
                        all_variable[2]: target,
                        all_variable[3]: f1,
                        all_variable[4]: f2,
                        all_variable[5]: f3,
                    }
                )
                for ID, (start_date, target, *_, f1, f2, f3) in df.iterrows()
            ]

        input_dict["Feat dynamic real"] = feat_dynamic_real
        # input_dict["experiment name"] = self.experiment_name.value

        # Save the dataset in Gluonts format
        test_ds_json = "\n".join(test_ds)
        test_ds_json_bytes = test_ds_json.encode("utf-8")
        test_ds_json_list = [json.loads(item) for item in test_ds_json.split("\n")]
        freq = str(self.data_freq[0]) + "D"
        test_ds_json_listDataset = ListDataset(test_ds_json_list, freq=freq)

        # Save the metadata
        json_str = json.dumps(metadata, default=str)
        json_bytes = json_str.encode("utf-8")

        return {
            "list": test_ds_json_list,
            "dataset": test_ds_json_listDataset,
            "dataset_info": json_bytes,
        }

    def data_prep(self):
        """This function prepares the data for the model training.

        Returns
        -------
        train_data : pd.DataFrame
        test_data : pd.DataFrame
        """

        data_sorted = self.data.sort_values(by=self.date_col)
        train_data = (
            data_sorted.groupby(self.ts_identifier_cols)
            .apply(lambda x: x.iloc[: -self.prediction_length])
            .reset_index(drop=True)
        )
        train_data_gluon = self.prepare_gluonts_data(train_data)
        test_data_gluon = self.prepare_gluonts_data(data_sorted)
        return train_data_gluon, test_data_gluon

    def runModel(
        self, algorithm: str, model_data: pd.DataFrame, model_param_dict: dict
    ):
        """
        Runs model for gluonts returns a fitted model.

        Parameters
        ----------
        algorithm : str - name of the algorithm (deepar, deepstate, etc)
        model_data: pd.DataFrame
        Data on which model to be fitted
        model_param_dict_fit: dict
        dict with parameters of the "fit" method
        Returns
        ------
        model
        model: fitted model
        """
        model = self.algo_choice_dict[algorithm](**model_param_dict["Estimator"])
        trained_model = model.train(training_data=model_data)
        return trained_model

    def GridSearchCV_gluon(self, model_related_dict: dict):
        """

        Starts model process for GluonTS Related Algo.

        This method tunes for the right parameters and build out the model for the right parameters.

        Parameters
        ----------
        model_related_dict : dict
            model_related_dict['validation']
            model_related_dict['param_dict']

        Returns
        -------
        hyperParams_metrics_log : pd.DataFrame
        """

        st = time.time()

        algorithm_ = (
            model_related_dict["param_dict"]["algorithm"]
            if model_related_dict["param_dict"]["algorithm"] is not None
            else "SimpleExponentialSmoothing"
        )
        master_kwargs = copy.deepcopy(self.kwargs_choice_dict[algorithm_])
        model_param_dict_ = copy.deepcopy(model_related_dict["param_dict"][algorithm_])

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
        ##########################################

        ####################################
        # Adding trainer parmeters to the estimator params
        # TODO: improve this loop
        hyperParams_df_list = []
        for i in range(len(hyperParams_w_algo)):
            hyperParams_df_est = {}
            Trainer_iter = hyperParams_w_algo[i]["Trainer"]
            trainer = Trainer(**Trainer_iter)
            hyperParams_w_algo[i]["Estimator"]["trainer"] = trainer
            hyperParams_df_est["Estimator"] = hyperParams_w_algo[i]["Estimator"]
            hyperParams_df_list.append(hyperParams_df_est)
        hyperParams_w_algo = hyperParams_df_list
        ##################################

        train, test = self.data_prep()

        ds_data_train = train["dataset"]
        ds_data_test = test["dataset"]

        metric_ = model_related_dict["validation"]["metric"]

        hyperParams_metrics_log = pd.DataFrame()
        params_list = []
        for iter_ in range(len(hyperParams_w_algo)):
            hyperParams_iter = copy.deepcopy(hyperParams_w_algo[iter_])
            print(hyperParams_iter)
            cv_metrics_log = pd.DataFrame()
            model_ = self.runModel(
                algorithm_,
                model_data=ds_data_train,
                model_param_dict=copy.deepcopy(hyperParams_iter),
            )
            forecast_iter, ts_iter = make_evaluation_predictions(
                dataset=ds_data_test,  # test dataset
                predictor=model_,  # predictor
                num_samples=100,  # number of sample paths we want for evaluation
            )
            evaluator = Evaluator()
            agg_metrics, _ = evaluator(iter(ts_iter), iter(forecast_iter))
            hyperParams_metrics_log = pd.concat(
                [hyperParams_metrics_log, pd.DataFrame(agg_metrics, index=[0])],
                ignore_index=True,
            )
            params_list.append(hyperParams_iter)

        hyperParams_metrics_log.rename(columns={metric_: "test_metric"}, inplace=True)
        hyperParams_metrics_log["params"] = params_list

        return hyperParams_metrics_log

    def getPredictions(self, model_data: pd.DataFrame, test_data, model_object):
        """
        This method to generate predictions/scoring.

        Parameters
        ----------
        model_data : pd.DataFrame
            pandas Dataframe on which scoring to be done using the model_object

        model_object: GluonTS model object
            model object built from 'runModel' method

        Return
        ------
        model_data: pd.DataFrame with predictions

        """
        forecast_it, _ = make_evaluation_predictions(
            dataset=test_data, predictor=model_object
        )
        forecasts = list(forecast_it)
        reslts_df_list = []
        freq = str(self.data_freq[0]) + "D"
        for ts_forecasts in forecasts:
            df_temp = pd.DataFrame(
                {
                    self.date_col: pd.date_range(
                        ts_forecasts.start_date.to_timestamp(),
                        periods=self.prediction_length,
                        freq=freq,
                    ),
                    "predictions": ts_forecasts.samples.mean(axis=0),
                    "ts_identifier": ts_forecasts.item_id,
                },
            )
            reslts_df_list.append(df_temp)
        df_predictions = pd.concat(reslts_df_list)
        model_data["ts_identifier"] = (
            model_data[self.ts_identifier_cols].astype(str).apply("_".join, axis=1)
        )
        test_df_w_pred = pd.merge(
            model_data, df_predictions, on=["ts_identifier"] + [self.date_col]
        )

        return test_df_w_pred
