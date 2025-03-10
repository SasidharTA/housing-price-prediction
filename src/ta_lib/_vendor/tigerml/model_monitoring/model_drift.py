import itertools
import logging
import numpy as np
import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String
from tigerml.core.dataframe.dataframe import measure_time
from tigerml.core.reports import create_report
from tigerml.model_monitoring.base_drift import BaseDrift
from tigerml.model_monitoring.performance import Performance
from tigerml.model_monitoring.utils.dao import db_connection, metadata
from tigerml.model_monitoring.utils.data_utils import (
    compare_bool_stats,
    compare_cat_stats,
    compare_num_stats,
    concat_dfs,
    get_all_segment_dfs,
    setanalyse_by_features,
)
from tigerml.model_monitoring.utils.highlighting import (
    apply_table_formatter_to_dict,
    table_formatter,
)
from typing import Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class ModelDrift(BaseDrift):
    """
    Base Class for creation of target, feature and concept Drift.

    Calculate target, feature and concept drift and statistical for data without
    any segment.

    Parameters
    ----------
    base_df: pd.DataFrame
        Base data / reference data
    current_df: pd.DataFrame
        Current data for which you want to calculate shift
    yhat: str
        Predicted target column name for data
    y:  str, default=None
        Actual target column name for data
    features: List[str], default=None
        List of features for which you want to calculate drift
    options: dict, default={}
        these options can control identification of categorical data.
    thresholds: dict, default=None
        user defined thresholds for drift metrics.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from tigerml.model_monitoring import ModelDrift
    >>> data = load_breast_cancer(as_frame=True)
    >>> X, y = data["data"], data["target"]
    >>> X_base, X_curr, y_base, y_curr = train_test_split(
    ...    X, y, test_size=0.5, random_state=42
    ... )
    >>> model = LogisticRegression()
    >>> model.fit(X_base, y_base)
    >>> yhat_base = model.predict(X_base)
    >>> yhat_curr = model.predict(X_curr)
    >>> base_df = pd.concat([X_base, y_base], axis=1)
    >>> base_df.loc[:, "predicted_target"] = yhat_base
    >>> curr_df = pd.concat([X_curr, y_curr], axis=1)
    >>> curr_df.loc[:, "predicted_target"] = yhat_curr
    >>> model_drift_base = ModelDrift(
    ...        base_df=base_df,
    ...        current_df=curr_df,
    ...        yhat="predicted_target",
    ...        y="target",
    ...    )
    >>> model_drift_base.get_report()
    """

    def __init__(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        yhat: str,
        y: Optional[str] = None,
        features: Optional[List[str]] = None,
        options: Optional[Dict] = {},
        thresholds: Optional[Dict] = None,
    ):

        super().__init__(
            base_df=base_df,
            current_df=current_df,
            yhat=yhat,
            y=y,
            features=features,
            options=options,
            thresholds=thresholds,
        )
        _LOGGER.info("Initiated the BaseModelDrift Class")

    def _get_drift_report_legacy(self, combine_features, light):

        data_summary = {
            key: table_formatter(value, format_bg_color=False)
            for key, value in self.data_summary.items()
        }
        drift_report = {
            "summary": {
                "drift_summary": self.drift_summary,
                "data_summary": data_summary,
            },
            "model_drift": {
                "target_desc_stats": {
                    "predicted": table_formatter(
                        self.descriptive_stats["target"], format_bg_color=False
                    ),
                    "actual": table_formatter(
                        self.descriptive_stats["actual_target"], format_bg_color=False
                    ),
                },
                "target_drift": table_formatter(self.target_drift, self.thresholds),
                "concept_drift": {
                    "numerical_features": table_formatter(
                        self.num_concept_drift, self.thresholds
                    ),
                    "cat_features": table_formatter(
                        self.cat_concept_drift, self.thresholds
                    ),
                },
                "performance_drift": self.get_performance_drift_df(),
                # "population_stability_index": population_stability_index,
                # "dependency_stability_index": dependency_stability_index,
            },
            "data_drift": {
                "feature_desc_stats": {
                    "numerical_features": table_formatter(
                        self.descriptive_stats["num_features"], format_bg_color=False
                    ),
                    "categorical_features": table_formatter(
                        self.descriptive_stats["cat_features"], format_bg_color=False
                    ),
                    "categorical_features_set_analysis": self.descriptive_stats[
                        "cat_features_set_analysis"
                    ],
                    "boolean_features": self.descriptive_stats["bool_features"],
                },
                "feature_drift": {
                    "num_features": table_formatter(
                        self.num_feature_drift, self.thresholds
                    ),
                    "categorical_features": table_formatter(
                        self.cat_feature_drift, self.thresholds
                    ),
                },
                # "feature_stability_index": feature_stability_index,
            },
            "glossary": self._get_glossary(),
        }

        if not light:

            if combine_features:
                population_stability_index = table_formatter(
                    concat_dfs(self.population_stability_index, names=["feature"]),
                    format_bg_color=False,
                )
                dependency_stability_index = table_formatter(
                    concat_dfs(self.dependency_stability_index, names=["feature"]),
                    format_bg_color=False,
                )
                feature_stability_index = table_formatter(
                    concat_dfs(self.feature_stability_index, names=["feature"]),
                    format_bg_color=False,
                )
            else:
                population_stability_index = {
                    key: table_formatter(value, format_bg_color=False)
                    for key, value in self.population_stability_index.items()
                }
                dependency_stability_index = {
                    key: table_formatter(value, format_bg_color=False)
                    for key, value in self.dependency_stability_index.items()
                }
                feature_stability_index = {
                    key: table_formatter(value, format_bg_color=False)
                    for key, value in self.feature_stability_index.items()
                }
            drift_report["model_drift"][
                "population_stability_index"
            ] = population_stability_index
            drift_report["model_drift"][
                "dependency_stability_index"
            ] = dependency_stability_index
            drift_report["data_drift"][
                "feature_stability_index"
            ] = feature_stability_index

        _LOGGER.info("drift report generated")
        return drift_report

    def _get_target_drift_legacy(self):
        """
        Target Drift.

        Get Target Drift summary and population stability index at bin level for
        target.

        Returns
        -------
        report: dict
            Dictionary with target drift summary and population stability index
        """
        target_drift = self._compute_target_drift()
        self.target_drift = target_drift["var_level"]
        self.population_stability_index = target_drift["bin_level"]["psi"]
        _LOGGER.info("calculated all drift")
        return target_drift

    def _get_concept_drift_legacy(self, combine_features=False):
        """
        Concept Drift summary and dsi.

        Get Concept Drift summary and dependency stability index at bin level
        for each feature.

        Parameters
        ----------
        combine_features: bool, default=False
            Once can set it to true for excel based output and get dependency
            stability index all in one sheet for all features.

        Returns
        -------
        report: dict
            Dictionary with concept drift summary for both numerical
            and categorical features and dependency stability index.
        """
        num_concept_drift, cat_concept_drift = self._compute_concept_drift()
        if num_concept_drift is not None:
            self.num_concept_drift = num_concept_drift["var_level"]
            self.dependency_stability_index.update(
                num_concept_drift["bin_level"]["dsi"]
            )
            _LOGGER.info(
                "Numerical feature concept drift set and dependency stability index for numerical fetaures updated"
            )
        else:
            # self.num_concept_drift = "No numerical feature "
            _LOGGER.info("No numerical features found in data")

        if len(cat_concept_drift):
            self.cat_concept_drift = cat_concept_drift["var_level"]
            self.dependency_stability_index.update(
                cat_concept_drift["bin_level"]["dsi"]
            )
            _LOGGER.info(
                "Categorical feature concept drift set and feature dependency index for categorical fetaures updated"
            )
        else:
            # self.cat_concept_drift = "No categorical feature"
            _LOGGER.info("No categorical features found in data")

        if len(self.dependency_stability_index):
            if combine_features:
                dsi = concat_dfs(self.dependency_stability_index, names=["feature"])
            else:
                dsi = {k: [v] for k, v in self.dependency_stability_index.items()}

        else:
            # self.dependency_stability_index = (
            # "Neither categorical nor numerical feature "
            # )
            _LOGGER.info("No change in dsi through numerical or categorical data")

        report = {
            "concept_drift_summary": {
                "num_features": self.num_concept_drift,
                "cat_features": self.cat_concept_drift,
            },
            "bin_level": {"dependency_stability_index": dsi},
        }

        _LOGGER.info("Concept drift report generated")
        return report

    def _get_feature_drift_legacy(self, combine_features=False):
        """
        Get Feature Drift Report.

        Feature Drift summary and feature stability index at bin level for
        each feature.

        Parameters
        ----------
        combine_features: bool, default=False
            Once can set it to true for excel based output and get dependency
            stability index all in one sheet for all features.

        Returns
        -------
        report: dict
            Dictionary with feature drift summary for both numerical and
            categorical features and feature stability index .
        """
        num_feature_drift, cat_feature_drift = self._compute_feature_drift()
        _LOGGER.info("Numerical and Categorical feature drift calculated")

        if num_feature_drift is not None:
            self.num_feature_drift = num_feature_drift["var_level"]
            self.feature_stability_index.update(num_feature_drift["bin_level"]["psi"])
            _LOGGER.info(
                "Numerical feature drift set and feature stability index for numerical fetaures updated"
            )
        # else:
        #     self.num_feature_drift = "No numerical feature"
        #     _LOGGER.info("No numerical features found in data")

        if len(cat_feature_drift):
            self.cat_feature_drift = cat_feature_drift["var_level"]
            self.feature_stability_index.update(cat_feature_drift["bin_level"]["psi"])
            _LOGGER.info(
                "Categorical feature drift set and feature stability index for categorical fetaures updated"
            )
        else:
            #     self.cat_feature_drift = "No categorical feature "
            _LOGGER.info("No categorical features found in data")

        if not len(self.feature_stability_index):
            # self.feature_stability_index = "Neither categorical nor numerical feature"
            _LOGGER.info("No change in psi through numerical or categorical data")

        if combine_features:
            fsi = concat_dfs(self.feature_stability_index, names=["feature"])
        else:
            fsi = {k: [v] for k, v in self.feature_stability_index.items()}
        report = {
            "feature_drift_summary": {
                "num_features": self.num_feature_drift,
                "cat_features": self.cat_feature_drift,
            },
            "bin_level": {"feature_stability_index": fsi},
        }

        _LOGGER.info("Feature drift report generated")
        return report

    def _get_drift_summary_legacy(self):
        drift_summary_dict = self._compute_drift_summary()
        _LOGGER.info("Drift summary calculated")

        first_index = np.array([])
        second_index = np.array([])
        summary_data = []
        for key, value in drift_summary_dict.items():
            if isinstance(value, dict):
                first_index = np.concatenate(
                    (first_index, np.array([key] * len(value)))
                )
                for sub_key, sub_value in value.items():
                    second_index = np.append(second_index, sub_key)
                    summary_data.append(sub_value)
            else:
                first_index = np.append(first_index, key)
                second_index = np.append(second_index, "")
                summary_data.append(value)
        summary_columns = [first_index, second_index]
        summary_data = np.array([summary_data])
        # Use "summary_columns" and "summary_data" to generate MultiIndex DataFrame
        self.drift_summary = pd.DataFrame(summary_data, columns=summary_columns)

        _LOGGER.info("Self drift_summary object set")
        return self.drift_summary

    def _get_data_summary_legacy(self):
        """
        Get Overall Data Summary for numerical and boolean features.

        For categorical features still need to be supported.

        Returns
        -------
            Dictionary with two keys one for numerical and other for overall.
        """
        data_summary = self._compute_data_summary()
        _LOGGER.info("Concept data summary calculated")
        # if data_summary["numerical_features"] is None:
        #     data_summary["numerical_features"] = "No Numerical Feature "
        #     _LOGGER.info("No numerical features found in data")
        # if data_summary["boolean_features"] is None:
        #     data_summary["boolean_features"] = "No boolean Feature "
        #     _LOGGER.info("No boolean features found in data")
        if data_summary["categorical_features"] is None:
            data_summary["categorical_features"] = pd.DataFrame(
                columns=["variable", "mode_base", "mode_curr"]
            )
            _LOGGER.info("No categorical features found in data")
        self.data_summary = data_summary
        _LOGGER.info("Self data summary object updated")
        return self.data_summary

    def _get_all_drift_legacy(
        self, combine_features=False, summary_options=None, light=True
    ):
        """
        Get complete drift reports (all 3 types).

        Return
        ------
        dict()
        """

        drift_report = dict()
        drift_report["target_drift"] = self._get_target_drift_legacy()
        drift_report["feature_drift"] = self._get_feature_drift_legacy(
            combine_features=combine_features
        )
        drift_report["concept_drift"] = self._get_concept_drift_legacy(
            combine_features=combine_features
        )

        # self.drift_summary = self._compute_drift_summary(threshold_options)
        _LOGGER.info("All drift report stored in drift report object")
        return drift_report

    def _get_summary_report_legacy(self, name="", path="", format=".html", columns=1):
        """
        Get Summary Report consisting of data_summary and drift_summary in specified format.

        Parameters
        ----------
        name: str, default=""
            Name of the file you want to store report.
        path: str, default=""
            Path of the file where you want to store report.
        format: str, default=".html"
            Format of the report which you want it could be ".xlsx" or ".html"
        columns: int, default=1
            If set to 1 leads to only one table per row in row otherwise 2 .
        """
        drift_summary = self._get_drift_summary_legacy()
        data_summary = self._get_data_summary_legacy()

        summary = {"data_summary": data_summary, "drift_summary": drift_summary}
        _LOGGER.info("Data and drift summary object created")
        create_report(summary, name=name, path=path, format=format, columns=columns)
        _LOGGER.info("Report created")

    def _get_model_drift_report_legacy(
        self, name="", path="", format=".html", columns=1
    ):
        """
        Get Model Drift Report.

        Parameters
        ----------
        name: str
            Name of the file you want to store report.
        path: str
            Path of the file where you want to store report.
        format: str
            Format of the report which you want it could be excel or html.

        """
        target_desc_stats = self.get_descriptive_stats()
        target_desc_stats_actual = self.get_actual_target_desc_stats_df()
        target_drift = self._get_target_drift_legacy()
        concept_drift = self._get_concept_drift_legacy()
        performance_drift = self.get_performance_drift_df()

        model_drift = {
            "target_desc_stats": target_desc_stats,
            "target_desc_stats_actual": target_desc_stats_actual,
            "target_drift": target_drift,
            "concept_drift": concept_drift,
            "performance_drift": performance_drift,
        }
        _LOGGER.info("Model drift report object created")
        create_report(model_drift, name=name, path=path, format=format, columns=columns)
        _LOGGER.info("Report created")

    def _get_data_drift_report_legacy(
        self, name="", path="", format=".html", columns=1
    ):
        """
        Get Data Drift Report.

        Parameters
        ----------
        name: str
            Name of the file you want to store report.
        path: str
            Path of the file where you want to store report.
        format: str
            Format of the report which you want it could be excel or html.

        """
        num_features = self.get_num_features_desc_stats_df()
        cat_features = self.get_cat_features_desc_stats_df()
        cat_set_analysis = self.get_cat_features_setanalysis_df()
        bool_features = self.get_bool_features_desc_stats_df()

        feature_drift = self._get_feature_drift_legacy()

        data_drift = {
            "feature_desc_stats": {
                "numerical_features": num_features,
                "categorical_features": cat_features,
                "categorical_feature_set_analysis": cat_set_analysis,
                "boolean_features": bool_features,
            },
            "feature_drift": feature_drift,
        }
        _LOGGER.info("detailed data drift report object created")
        create_report(data_drift, name=name, path=path, format=format, columns=columns)
        _LOGGER.info("Report created")

    def _get_report_legacy(
        self, name="", path="", format=".html", light=True, columns=1
    ):
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

        if format == ".xlsx":
            combine_features = True
        else:
            combine_features = False

        descriptive_stats = self.get_descriptive_stats()
        drift_detailed_report = self._get_all_drift_legacy(
            combine_features=combine_features, light=light
        )
        drift_summary = self._get_drift_summary_legacy()
        data_summary = self._get_data_summary_legacy()

        drift_report = self._get_drift_report_legacy(
            combine_features=combine_features, light=light
        )
        create_report(
            drift_report, name=name, path=path, format=format, columns=columns
        )
        _LOGGER.info("Final report generated and saved")

    def get_performance_drift_df(self):
        """
        Get all the performance metrics for the model.

        Returns
        -------
        pd.DataFrame
        """
        if (self.y_base is not None) and (self.y_curr is not None):
            self.performance = Performance(
                self.base_df,
                self.current_df,
                self.yhat_base,
                self.yhat_curr,
                self.y_base,
                self.y_curr,
            )._compute_performance_drift()
            return self.performance
        else:
            self.performance = pd.DataFrame(
                columns=["measures", "base", "current", "index"]
            )
            _LOGGER.info("Actual target not present to calculate drift")
            return self.performance

    def get_predicted_target_desc_stats_df(self):
        """
        Get target descriptive stats.

        Returns
        -------
        target_summary: pandas.DataFrame
            Returns pandas DataFrame object
        """

        if self.target_type == "categorical":
            summary_func = compare_cat_stats
        else:
            summary_func = compare_num_stats
        target_summary = summary_func(
            base_df=self.base_df[[self.yhat_base]],
            curr_df=self.current_df[[self.yhat_curr]],
        )

        _LOGGER.info("Target Descriptive Stats calculated")
        return target_summary

    def get_actual_target_desc_stats_df(self):
        """Get target descriptive stats."""

        if self.y_base is None or self.y_curr is None:
            return "No actual values for target variable given"

        if self.actual_target_type == "categorical":
            summary_func = compare_cat_stats
        else:
            summary_func = compare_num_stats

        actual_target = summary_func(
            base_df=self.base_df[[self.y_base]],
            curr_df=self.current_df[[self.y_curr]],
        )

        _LOGGER.info("Target Actual Descriptive Stats calculated")
        return actual_target

    def get_num_features_desc_stats_df(self):
        """
        Get numerical features descriptive stats.

        Returns
        -------
        num_summary: pandas.DataFrame or str
            Returns pandas DataFrame object if num features are there, else returns str
        """
        if len(self.num_features):
            num_summary = compare_num_stats(
                base_df=self.base_df,
                curr_df=self.current_df,
                features=self.num_features,
            )
            _LOGGER.info("Numerical features descriptive stats calculated")
        else:
            num_summary = pd.DataFrame(
                columns=[
                    "variable",
                    "count_base",
                    "count_curr",
                    "mean_base",
                    "mean_curr",
                    "std_base",
                    "std_curr",
                    "min_base",
                    "min_curr",
                    "median_base",
                    "median_curr",
                    "max_base",
                    "max_curr",
                ]
            )
            _LOGGER.info("No numerical features found in data")

        return num_summary

    def get_cat_features_desc_stats_df(self):
        """
        Get categorical descriptive stats.

        Returns
        -------
        cat_summary: pandas.DataFrame
            Returns pandas DataFrame object for cat features
        """
        if len(self.cat_features):
            cat_summary = compare_cat_stats(
                base_df=self.base_df,
                curr_df=self.current_df,
                features=self.cat_features,
            )
            _LOGGER.info("Categorical features descriptive stats calculated.")
        else:
            cat_summary = pd.DataFrame(
                columns=[
                    "variable",
                    "count_base",
                    "count_curr",
                    "unique_base",
                    "unique_curr",
                    "mode_base",
                    "mode_curr",
                    "mode_freq_base",
                    "mode_freq_curr",
                    "mode_freq_pct_base",
                    "mode_freq_pct_curr",
                ]
            )
            _LOGGER.info("No categorical features found in data")

        return cat_summary

    def get_bool_features_desc_stats_df(self):
        """Get boolean descriptive stats."""
        if len(self.bool_features):
            bool_summary = compare_bool_stats(
                base_df=self.base_df,
                curr_df=self.current_df,
                features=self.bool_features,
            )
            _LOGGER.info("Boolean features descriptive stats calculated.")
        else:
            bool_summary = pd.DataFrame(
                columns=[
                    "variable",
                    "count_0s_base",
                    "count_0s_curr",
                    "count_1s_base",
                    "count_1s_curr",
                    "Perc_0s_base",
                    "Perc_0s_curr",
                    "Perc_1s_base",
                    "Perc_1s_curr",
                ]
            )
            _LOGGER.info("No boolean features found in data")

        return bool_summary

    def get_cat_features_setanalysis_df(self, diff_only=True):
        """
        Get set difference for categorical features.

        Parameters
        ----------
        diff_only: bool, default=True
            If set to true returns only rows where difference is.

        Returns
        -------
        cat_setanalysis: pandas.DataFrame
            Returns pandas DataFrame object of difference found in cat features
        """
        if len(self.cat_features):
            cat_setanalysis = setanalyse_by_features(
                base_df=self.base_df,
                curr_df=self.current_df,
                features=self.cat_features,
                diff_only=diff_only,
            )
            _LOGGER.info(
                "Analysis of categorical feature levels done and difference calculated"
            )
            if cat_setanalysis.empty:
                cat_setanalysis = pd.DataFrame(
                    columns=[
                        "variable",
                        "n_base",
                        "n_curr",
                        "n_common",
                        "n_base-curr",
                        "n_curr-base",
                        "base-curr",
                        "curr-base",
                        "setdiff",
                    ]
                )
                _LOGGER.info("No difference in categorical feature levels.")
        else:
            cat_setanalysis = pd.DataFrame(
                columns=[
                    "variable",
                    "n_base",
                    "n_curr",
                    "n_common",
                    "n_base-curr",
                    "n_curr-base",
                    "base-curr",
                    "curr-base",
                    "setdiff",
                ]
            )
            _LOGGER.info("No categorical features present")
        return cat_setanalysis

    def get_target_drift_df(self):
        """
        Get the target drift between yhat_base(predicted target of base) and yhat_curr(predicted target of curr).

        Returns
        -------
        target_drift_var_df: pd.DataFrame
        """
        target_drift_var_df = self._compute_target_drift_at_var_level()
        return target_drift_var_df

    def get_target_drift_at_bin_level_df(self):
        """
        Get the target drift at bin level between yhat_base(predicted target of base) and yhat_curr(predicted target of curr).

        Returns
        -------
        target_drift_bin_df: pd.DataFrame
        """
        target_drift_bin_df = self._compute_target_drift_at_bin_level()
        return target_drift_bin_df

    def get_num_feature_drift_df(self):
        """
        Get the feature drift of numerical features.

        Returns
        -------
        num_feature_drift_df: pd.DataFrame
        """
        num_feature_drift_df = self._compute_num_feature_drift()
        return num_feature_drift_df

    def get_cat_feature_drift_df(self):
        """
        Get the feature drift of categorical features.

        Returns
        -------
        cat_feature_drift_df: pd.DataFrame
        """
        cat_feature_drift_df = self._compute_cat_feature_drift()
        return cat_feature_drift_df

    def get_num_feature_drift_at_bin_level_df(self):
        """
        Get the feature drift of numerical features at bin level.

        Returns
        -------
        num_feature_drift_at_bin_level_df: pd.DataFrame
        """
        num_feature_drift_at_bin_level_df = (
            self._compute_num_feature_drift_at_bin_level()
        )
        return num_feature_drift_at_bin_level_df

    def get_cat_feature_drift_at_bin_level_df(self):
        """
        Get the feature drift of categorical features at bin level.

        Returns
        -------
        cat_feature_drift_at_bin_level_df: pd.DataFrame
        """
        cat_feature_drift_at_bin_level_df = (
            self._compute_cat_feature_drift_at_bin_level()
        )
        return cat_feature_drift_at_bin_level_df

    def get_num_concept_drift_df(self):
        """
        Get the concept drift of numerical features.

        Returns
        -------
        num_concept_drift: pd.DataFrame
        """
        num_concept_drift = self._compute_num_concept_drift()
        return num_concept_drift

    def get_cat_concept_drift_df(self):
        """
        Get the concept drift of categorical features.

        Returns
        -------
        cat_concept_drift: pd.DataFrame
        """
        cat_concept_drift = self._compute_cat_concept_drift()
        return cat_concept_drift

    def get_num_concept_drift_at_bin_level_df(self):
        """
        Get the concept drift of numerical features at bin level.

        Returns
        -------
        num_concept_drift_at_bin_level_df: pd.DataFrame
        """
        num_concept_drift_at_bin_level_df = (
            self._compute_num_concept_drift_at_bin_level()
        )
        return num_concept_drift_at_bin_level_df

    def get_cat_concept_drift_at_bin_level_df(self):
        """
        Get the concept drift of categorical features at bin level.

        Returns
        -------
        cat_concept_drift_at_bin_level_df: pd.DataFrame
        """
        cat_concept_drift_at_bin_level_df = (
            self._compute_cat_concept_drift_at_bin_level()
        )
        return cat_concept_drift_at_bin_level_df

    def get_drift_summary_df(self):
        """
        Get the summary of target,feature and concept drift.

        Returns
        -------
        self.drift_summary: pd.DataFrame
        """
        drift_summary_dict = self._compute_drift_summary()
        _LOGGER.info("Drift summary calculated")

        first_index = np.array([])
        second_index = np.array([])
        summary_data = []
        for key, value in drift_summary_dict.items():
            if isinstance(value, dict):
                first_index = np.concatenate(
                    (first_index, np.array([key] * len(value)))
                )
                for sub_key, sub_value in value.items():
                    second_index = np.append(second_index, sub_key)
                    summary_data.append(sub_value)
            else:
                first_index = np.append(first_index, key)
                second_index = np.append(second_index, "")
                summary_data.append(value)
        summary_columns = [first_index, second_index]
        summary_data = np.array([summary_data])
        # Use "summary_columns" and "summary_data" to generate MultiIndex DataFrame
        self.drift_summary = pd.DataFrame(summary_data, columns=summary_columns)
        _LOGGER.info("Self drift_summary object set")
        return self.drift_summary

    def get_num_data_summary_df(self):
        """
        Get the summary of numerical data.

        Returns
        -------
        num_data_summary_df: pd.DataFrame
        """
        num_data_summary_df = self._compute_num_data_summary()
        return num_data_summary_df

    def get_cat_data_summary_df(self):
        """
        Get the summary of categorical data.

        Returns
        -------
        cat_data_summary_df: pd.DataFrame
        """
        cat_data_summary_df = self._compute_cat_data_summary()
        return cat_data_summary_df

    def get_bool_data_summary_df(self):
        """
        Get the summary of boolean data.

        Returns
        -------
        bool_data_summary_df: pd.DataFrame
        """
        bool_data_summary_df = self._compute_bool_data_summary()
        return bool_data_summary_df

    def get_glossary_df(self):
        """
        Get the glossary terms.

        Returns
        -------
        glossary_dataframe: pd.DataFrame
        """
        glossary_dataframe = self.glossary_dataframe
        return glossary_dataframe

    def get_glossary_definitions_df(self):
        """
        Returns the Glossary df containing metric name and its definition.

        Returns
        -------
        glossary_definitions_df: pd.DataFrame
        """
        glossary_definitions_df = self.glossary_dataframe[
            ["Name of the Metrics", "Description"]
        ]
        return glossary_definitions_df

    def get_glossary_thresholds_df(self):
        """
        Returns the Glossary df containing metric name and the threshold applied to it.

        Returns
        -------
        glossary_thresholds_df: pd.DataFrame
        """
        glossary_thresholds_df = self.glossary_dataframe[
            ["Name of the Metrics", "Thresholds"]
        ]
        return glossary_thresholds_df

    def get_descriptive_stats(self, diff_only=True):
        """
        Comparison of summary stats between base and current data.

        Parameters
        ----------
        diff_only: bool, default=False
            If set to true returns only rows where difference is.

        Returns
        -------
        self.descriptive_stats: dict
            Dictionary which consists of get_predicted_target_desc_stats(),
            get_actual_target_desc_stats(),get_num_features_desc_stats(), get_cat_features_desc_stats()
            ,get_cat_features_setanalysis() and get_bool_features_desc_stats() Dataframes.
        """
        self.descriptive_stats = {
            "target": self.get_predicted_target_desc_stats_df(),
            "actual_target": self.get_actual_target_desc_stats_df(),
            "num_features": self.get_num_features_desc_stats_df(),
            "cat_features": self.get_cat_features_desc_stats_df(),
            "cat_features_set_analysis": self.get_cat_features_setanalysis_df(
                diff_only=diff_only
            ),
            "bool_features": self.get_bool_features_desc_stats_df(),
        }
        _LOGGER.info("Self data descriptive stats object updated")
        return self.descriptive_stats

    def get_data_summary_dict(self):
        """
        Get the data summary dict.

        This dict contains get_num_data_summary_df(),
        get_bool_data_summary_df() and get_cat_data_summary_df().

        Returns
        -------
        data_summary_dict: dict
        """
        data_summary_dict = {
            "numerical_features": self.get_num_data_summary_df(),
            "boolean_features": self.get_bool_data_summary_df(),
            "categorical_features": self.get_cat_data_summary_df(),
        }
        return data_summary_dict

    def get_target_desc_stats_dict(self):
        """
        Get the target description stats dict.

        This dict contains get_predicted_target_desc_stats_df(),
        get_actual_target_desc_stats_df().

        Returns
        -------
        target_desc_stats_dict: dict
        """
        target_desc_stats_dict = {
            "Predicted Target Desc Stats": self.get_predicted_target_desc_stats_df(),
            "Actual Target Desc Stats": self.get_actual_target_desc_stats_df(),
        }
        return target_desc_stats_dict

    def get_feature_desc_stats_dict(self):
        """
        Get the feature description stats dict.

        This dict contains get_num_features_desc_stats_df(),
        get_cat_features_desc_stats_df(),get_cat_features_setanalysis_df(),
        get_bool_features_desc_stats_df()

        Returns
        -------
        feature_desc_stats_dict: dict
        """
        feature_desc_stats_dict = {
            "Numerical Features Desc Stats": self.get_num_features_desc_stats_df(),
            "Categorical Features Desc Stats": self.get_cat_features_desc_stats_df(),
            "Categorical Features Set Analysis": self.get_cat_features_setanalysis_df(),
            "Boolean Features Desc Stats": self.get_bool_features_desc_stats_df(),
        }
        return feature_desc_stats_dict

    def get_target_drift_dict(self, light=True):
        """
        Get the target drift dict.

        This dict contains get_target_drift_df(),
        get_target_drift_at_bin_level_df()

        Parameters
        ----------
        light:bool, default=True
            If True, it avoids calculating drift at bin level

        Returns
        -------
        target_drift_dict: dict
        """
        if light:
            target_drift_dict = {
                "Target Drift": self.get_target_drift_df(),
            }
        else:
            target_drift_dict = {
                "Target Drift": self.get_target_drift_df(),
                "Target Drift at bin Level": self.get_target_drift_at_bin_level_df(),
            }
        return target_drift_dict

    def get_concept_drift_dict(self, light=True):
        """
        Get the concept drift dict.

        This dict contains get_num_concept_drift_df(),get_cat_concept_drift_df()
        get_num_concept_drift_at_bin_level_df(),get_cat_concept_drift_at_bin_level_df()

        Parameters
        ----------
        light: bool, default=True
            Avoids generating multiple dfs for each binned variable

        Returns
        -------
        concept_drift_dict: dict
        """
        if light:
            concept_drift_dict = {
                "Numerical Concept Drift": self.get_num_concept_drift_df(),
                "Categorical Concept Drift": self.get_cat_concept_drift_df(),
            }
        else:
            concept_drift_dict = {
                "Numerical Concept Drift": self.get_num_concept_drift_df(),
                "Categorical Concept Drift": self.get_cat_concept_drift_df(),
                "Numerical Concept Drift at bin level": get_all_segment_dfs(
                    df=self.get_num_concept_drift_at_bin_level_df(),
                    segment_by="feature",
                ),
                "Categorical Concept Drift at bin level": get_all_segment_dfs(
                    df=self.get_cat_concept_drift_at_bin_level_df(),
                    segment_by="feature",
                ),
            }
        return concept_drift_dict

    def get_feature_drift_dict(self, light=True):
        """
        Get the feature drift dict.

        This dict contains get_num_feature_drift_df(),get_cat_feature_drift_df()
        get_num_feature_drift_at_bin_level_df(),get_cat_feature_drift_at_bin_level_df()

        Parameters
        ----------
        light: bool, default=True
            Avoids generating multiple dfs for each binned variable

        Returns
        -------
        feature_drift_dict: dict
        """
        if light:
            feature_drift_dict = {
                "Numerical Feature Drift": self.get_num_feature_drift_df(),
                "Categorical Feature Drift": self.get_cat_feature_drift_df(),
            }
        else:
            feature_drift_dict = {
                "Numerical Feature Drift": self.get_num_feature_drift_df(),
                "Categorical Feature Drift": self.get_cat_feature_drift_df(),
                "Numerical Feature Drift at bin level": get_all_segment_dfs(
                    df=self.get_num_feature_drift_at_bin_level_df(),
                    segment_by="feature",
                ),
                "Categorical Feature Drift at bin level": get_all_segment_dfs(
                    df=self.get_cat_feature_drift_at_bin_level_df(),
                    segment_by="feature",
                ),
            }
        return feature_drift_dict

    def get_model_monitoring_summary_dict(self):
        """
        Get the Model Monitoring's Summary dict.

        This dict contains get_drift_summary_df(),get_data_summary_dict()
        get_target_desc_stats_dict(),get_feature_desc_stats_dict()

        Returns
        -------
        model_monitoring_summary_dict: dict
        """
        model_monitoring_summary_dict = {
            "Drift Summary": self.get_drift_summary_df(),
            "Data Summary": self.get_data_summary_dict(),
            "Target Descriptive Stats": self.get_target_desc_stats_dict(),
            "Feature Descriptive Stats": self.get_feature_desc_stats_dict(),
        }
        return model_monitoring_summary_dict

    def get_model_monitoring_model_drift_dict(self, light=True):
        """
        Get the Model Monitoring's Model Drift dict.

        This dict contains get_target_drift_dict(),get_concept_drift_dict()
        get_performance_drift_df()

        Parameters
        ----------
        light: bool, default=True

        Returns
        -------
        model_monitoring_model_drift_dict: dict
        """
        model_monitoring_model_drift_dict = {
            "Target Drift": self.get_target_drift_dict(light),
            "Concept Drift": self.get_concept_drift_dict(light),
            "Performance Drift": self.get_performance_drift_df(),
        }
        return model_monitoring_model_drift_dict

    def get_model_monitoring_data_drift_dict(self, light=True):
        """
        Get the Model Monitoring's Data Drift dict.

        This dict contains get_feature_drift_dict()

        Parameters
        ----------
        light: bool, default=True

        Returns
        -------
        model_monitoring_data_drift_dict: dict
        """
        model_monitoring_data_drift_dict = {
            "Feature Drift": self.get_feature_drift_dict(light)
        }
        return model_monitoring_data_drift_dict

    def get_model_monitoring_dict(self, light=True):
        """
        Get the report_dict for entire model monitoring report.

        Parameters
        ----------
        light: bool, default=True
            Avoids separate dataframes for each binned variable
        """
        model_monitoring_dict = {
            "Summary": self.get_model_monitoring_summary_dict(),
            "Model Drift": self.get_model_monitoring_model_drift_dict(light),
            "Data Drift": self.get_model_monitoring_data_drift_dict(light),
            "Glossary": self.get_glossary_df(),
        }
        return model_monitoring_dict

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
        report_dict = self.get_model_monitoring_dict(light=light)
        formatted_report_dict = apply_table_formatter_to_dict(report_dict=report_dict)
        create_report(
            formatted_report_dict, name=name, path=path, format=format, columns=columns
        )
        _LOGGER.info("Final report generated and saved")
