import logging
import numpy as np
import pandas as pd
import pdb
import yaml
from collections import defaultdict
from pickle import TRUE
from tigerml.core.reports import create_report
from tigerml.core.utils.segmented import (
    calculate_all_segments,
    get_segment_filter,
)
from tigerml.model_monitoring.base_drift import BaseDrift
from tigerml.model_monitoring.config.drift_options import DRIFT_OPTIONS
from tigerml.model_monitoring.config.glossary import GLOSSARY_DATAFRAME
from tigerml.model_monitoring.config.summary_options import SUMMARY_OPTIONS
from tigerml.model_monitoring.config.threshold_options import THRESHOLD_OPTIONS
from tigerml.model_monitoring.model_drift import ModelDrift
from tigerml.model_monitoring.plotters.plot import get_barchart
from tigerml.model_monitoring.segmented import SegmentedModelDrift
from tigerml.model_monitoring.utils.data_utils import (
    concat_dfs,
    flatten_dict,
    setanalyse_by_features,
)
from tigerml.model_monitoring.utils.highlighting import (
    apply_table_formatter_to_dict,
    table_formatter,
)
from typing import Dict, List, Optional, Union

_LOGGER = logging.getLogger(__name__)


class MultipleModelDrift:
    """
    Base Class for creation of multi-model target, feature and concept Drift.

    Calculate target, feature and concept drift and statistical for data for multiple models

    Parameters
    ----------
    config: dict
        The config dictionary containing list of ModelDrift/ SegmentedModelDrift Params
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from tigerml.model_monitoring import MultipleModelDrift
    >>> data = load_breast_cancer(as_frame=True)
    >>> X, y = data["data"], data["target"]
    >>> X_base, X_curr, y_base, y_curr = train_test_split(
    ...     X, y, test_size=0.33, random_state=42
    ... )
    >>> model = LogisticRegression()
    >>> model.fit(X_base, y_base)
    >>> yhat_base = model.predict(X_base)
    >>> yhat_curr = model.predict(X_curr)
    >>> base_df = pd.concat([X_base, y_base], axis=1)
    >>> base_df.loc[:, "predicted_target1"] = yhat_base
    >>> curr_df = pd.concat([X_curr, y_curr], axis=1)
    >>> curr_df.loc[:, "predicted_target1"] = yhat_curr
    >>> X_base2 = X_base.drop(["worst concavity", "fractal dimension error"], axis=1)
    >>> X_curr2 = X_curr.drop(["worst concavity", "fractal dimension error"], axis=1)
    >>> model2 = LogisticRegression()
    >>> model2.fit(X_base2, y_base)
    >>> yhat_base2 = model2.predict(X_base2)
    >>> yhat_curr2 = model2.predict(X_curr2)
    >>> base_df.loc[:, "predicted_target2"] = yhat_base2
    >>> curr_df.loc[:, "predicted_target2"] = yhat_curr2
    >>> config_dict = {
    "Model1::Version1": {
        "base_df":base_df,
        "current_df":curr_df,
        "yhat": "predicted_target1",
        "y": "target",
    },
    "Model2::Version2": {
        "base_df":base_df,
        "current_df":curr_df,
        "yhat": "predicted_target2",
        "y": "target",
    },
    }
    ... multiple_model_drift_base = MultipleModelDrift(config=config_dict)
    >>> multiple_model_drift_base.get_report()
    """

    def __init__(
        self,
        config: dict,
    ):

        self.config = config
        (
            self.single_model_list,
            self.segmented_model_list,
        ) = self._instantiate_models_from_config()
        _LOGGER.info("Initiated the MultipleModelDrift Class")

    def _instantiate_models_from_config(self):
        single_model_list = []
        segmented_model_list = []
        for model_name, modeldrift_params in self.config.items():
            if (
                "segment_by" in modeldrift_params
                and modeldrift_params["segment_by"] is not None
            ):
                segmented_model_class = SegmentedModelDrift(**modeldrift_params)
                segmented_model_list.append({model_name: segmented_model_class})
            else:
                modeldrift_params.pop(
                    "segment_by", None
                )  # Make sure we don't pass segment_by to ModelDrift class
                single_model_class = ModelDrift(**modeldrift_params)
                single_model_list.append({model_name: single_model_class})
        return single_model_list, segmented_model_list

    def _get_plots_dict(self, drift_plots):
        """Get drift plots for the data and model."""
        plots = {}
        for plot in drift_plots:
            plots[plot] = drift_plots[plot]

        return plots

    def get_all_plots(self, summary_options=None):
        """
        Plotting target, model and concept drifts.

        Parameters
        ----------
        summary_options: tigerml.core.utils._lib.DictObject, default=None
            If None, summary_options=MultipleModelDrift().summary_options

        Returns
        -------
        drift_plots: dict
            Dictionary containing plots
        """

        drift_plots = defaultdict(lambda: defaultdict(dict))
        if summary_options is None:
            summary_options = self.summary_options

        for key in self.multi_model_drift.keys():
            num_feature_drift = self.multi_model_drift[key].num_feature_drift
            num_concept_drift = self.multi_model_drift[key].num_concept_drift

            cat_feature_drift = self.multi_model_drift[key].cat_feature_drift
            cat_concept_drift = self.multi_model_drift[key].cat_concept_drift

            if isinstance(num_feature_drift, pd.DataFrame):
                drift_plots[key]["NumericalFeatures"] = [
                    get_barchart(
                        num_feature_drift,
                        value_on=summary_options["feature_drift_numerical"][
                            "threshold_on"
                        ],
                        sort=True,
                    ),
                    get_barchart(
                        num_concept_drift,
                        value_on=summary_options["concept_drift_numerical"][
                            "threshold_on"
                        ],
                        sort=True,
                    ),
                ]
            else:
                drift_plots[key]["NumericalFeatures"] = "No numerical feature "

            if isinstance(cat_feature_drift, pd.DataFrame):
                drift_plots[key]["CategoricalFeatures"] = [
                    get_barchart(
                        cat_feature_drift,
                        value_on=summary_options["feature_drift_categorical"][
                            "threshold_on"
                        ],
                        sort=True,
                    ),
                    get_barchart(
                        cat_concept_drift,
                        value_on=summary_options["concept_drift_categorical"][
                            "threshold_on"
                        ],
                        sort=True,
                    ),
                ]
            else:
                drift_plots[key]["CategoricalFeatures"] = "No categorical feature "

        _LOGGER.info("All Plots Generation Completed")

        return drift_plots

    def get_glossary_definitions_df(self):
        """
        Returns the Glossary df containing metric name and its definition.

        Returns
        -------
        glossary_definitions_df: pd.DataFrame
        """
        glossary_definitions_df = GLOSSARY_DATAFRAME[
            ["Name of the Metrics", "Description"]
        ]
        return glossary_definitions_df

    def get_single_model_details_dict(self, light=True):
        """
        Returns a consolidated dictionary containing ALL single model's  "Details".

        i.e All details apart from Drift Summary.

        Parameters
        ----------
        light: bool, default=True
            if True, it avoids returning a dict of dfs for each features.

        Returns
        -------
        consolidated_single_model_details_dict: dict
        """
        consolidated_single_model_details_dict = {}
        for single_model_dict in self.single_model_list:
            for model_name, single_model_class in single_model_dict.items():
                details_dict = {
                    model_name: {
                        "Data Summary": single_model_class.get_data_summary_dict(),
                        "Descriptive Stats": {
                            "Target Desc Stats": single_model_class.get_target_desc_stats_dict(),
                            "Feature Desc Stats": single_model_class.get_feature_desc_stats_dict(),
                        },
                        "Model Drift": single_model_class.get_model_monitoring_model_drift_dict(
                            light
                        ),
                        "Data Drift": single_model_class.get_model_monitoring_data_drift_dict(
                            light
                        ),
                        "Thresholds applied": single_model_class.get_glossary_thresholds_df(),
                    }
                }
                consolidated_single_model_details_dict.update(details_dict)
        return consolidated_single_model_details_dict

    def get_segmented_model_details_dict(self, light=True):
        """
        Returns a consolidated dictionary containing ALL segmented model's  "Details".

        i.e All details apart from Drift Summary.

        Parameters
        ----------
        light: bool, default=True
            if True, it avoids returning a dict of dfs for each features.

        Returns
        -------
        consolidated_segmented_model_details_dict: dict
        """
        consolidated_segmented_model_details_dict = {}
        for segmented_model_dict in self.segmented_model_list:
            for model_name, segmented_model_class in segmented_model_dict.items():
                details_dict = {model_name: {}}
                consolidated_segmented_model_details_dict.update(details_dict)
        return consolidated_segmented_model_details_dict

    def get_consolidated_model_details_dict(self, light=True):
        """
        Returns a consolidated dictionary containing ALL single & segmented model's  "Details".

        i.e All details apart from Drift Summary.

        Parameters
        ----------
        light: bool, default=True
            if True, it avoids returning a dict of dfs for each features.

        Returns
        -------
        consolidated_multi_model_monitoring_details_dict: dict
        """
        consolidated_multi_model_monitoring_details_dict = {}
        single_model_details_dict = self.get_single_model_details_dict(light)
        segmented_model_details_dict = self.get_segmented_model_details_dict(light)
        consolidated_multi_model_monitoring_details_dict = dict(
            single_model_details_dict, **segmented_model_details_dict
        )
        return consolidated_multi_model_monitoring_details_dict

    def get_consolidated_model_drift_summary_df(self):
        """
        Returns a consolidated dictionary containing ALL single & segmented model's Drift Summary.

        Returns
        -------
        consolidated_model_monitoring_drift_summary_df: pd.DataFrame
        """
        consolidated_model_monitoring_drift_summary_dict = {}
        for single_model_dict in self.single_model_list:
            for model_name, single_model_class in single_model_dict.items():
                drift_summary_df = single_model_class.get_drift_summary_df()
                consolidated_model_monitoring_drift_summary_dict[
                    model_name
                ] = drift_summary_df
        for segmented_model_dict in self.segmented_model_list:
            for model_name, segmented_model_class in segmented_model_dict:
                drift_summary_df = segmented_model_class._get_drift_summary()["overall"]
                consolidated_model_monitoring_drift_summary_dict[
                    model_name
                ] = drift_summary_df

        consolidated_model_monitoring_drift_summary_df = concat_dfs(
            df_dict=consolidated_model_monitoring_drift_summary_dict, names=["Model"]
        ).set_index(keys="Model")
        consolidated_model_monitoring_drift_summary_df.index.name = None
        return consolidated_model_monitoring_drift_summary_df

    def get_model_monitoring_dict(self, light=True):
        """
        Returns the entire multi-model monitoring dictionary.

        Parameters
        ----------
        light: bool, default=True

        Returns
        -------
        multi_model_monitoring_report_dict: dict
        """
        multi_model_monitoring_report_dict = {
            "Summary": self.get_consolidated_model_drift_summary_df(),
            "Details": self.get_consolidated_model_details_dict(light),
            "Glossary": self.get_glossary_definitions_df(),
        }
        return multi_model_monitoring_report_dict

    def create_report_from_dict(
        self, report_dict=None, name="", path="", format=".html", columns=1
    ):
        """
        Create report in specified format given a dictionary whose leaves are a df/plot.

        Parameters
        ----------
        report_dict: dict
            dictionary whose leaves are a df/plot
        name: str, default=""
            Name of the file you want to store report.
        path: str, default=""
            Path of the file where you want to store report.
        format: str, default=".html"
            Format of the report which you want it could be .xlxs or .html.
        light: bool
            Default=True. Avoids binning calculations
        columns: int, default=1
            If set to 1 leads to only one table per row in row otherwise 2 .
        """
        formatted_report_dict = apply_table_formatter_to_dict(report_dict=report_dict)
        create_report(
            formatted_report_dict, name=name, path=path, format=format, columns=columns
        )
        _LOGGER.info("Final report generated and saved")

    def get_report(self, name="", path="", format=".html", light=True, columns=1):
        """
        Get complete model monitoring report in specified format.

        Parameters
        ----------
        name: str, default=""
            Name of the file you want to store report.
        path: str, default=""
            Path of the file where you want to store report.
        format: str, default=".html"
            Format of the report which you want it could be .xlxs or .html.
        light: bool
            Default=True. Avoids binning calculations
        columns: int, default=1
            If set to 1 leads to only one table per row in row otherwise 2 .
        """
        multi_model_monitoring_report_dict = self.get_model_monitoring_dict(light)
        formatted_report_dict = apply_table_formatter_to_dict(
            report_dict=multi_model_monitoring_report_dict
        )
        create_report(
            formatted_report_dict, name=name, path=path, format=format, columns=columns
        )
        _LOGGER.info("Final report generated and saved")
