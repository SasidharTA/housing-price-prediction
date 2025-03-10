import logging
import numpy as np
import os
import pandas as pd
import pdb
import warnings
import yaml
from collections import defaultdict
from sqlalchemy.sql import text
from tigerml.core.dataframe.dataframe import measure_time
from tigerml.model_monitoring.config.drift_options import DRIFT_OPTIONS
from tigerml.model_monitoring.config.glossary import (
    GLOSSARY_DATAFRAME,
    update_glossary_df_with_thresholds,
)
from tigerml.model_monitoring.config.summary_options import (
    SUMMARY_OPTIONS,
    update_summary_with_thresholds,
)
from tigerml.model_monitoring.config.threshold_options import (
    THRESHOLD_OPTIONS,
    update_threshold_options,
)
from tigerml.model_monitoring.utils.dao import db_connection, metadata
from tigerml.model_monitoring.utils.data_utils import (
    compare_bool_stats,
    compare_cat_stats,
    compare_num_stats,
    concat_dfs,
    get_data_type,
)
from tigerml.model_monitoring.utils.highlighting import table_formatter
from tigerml.model_monitoring.utils.misc import (
    apply_threshold,
    get_applicable_metrics,
)
from typing import Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class BaseDrift:
    """
    Common Class containing mostly private functions used by base and segmented.

    Parameters
    ----------
    base_df: pd.DataFrame
        Base data / reference data
    current_df: pd.DataFrame
        Current data for which you want to calculate shift
    yhat: str
        Predicted target column name for data
    y:  str
        Actual target column name for data
    features: List[str]
        List of features for which you want to calculate drift
    """

    def __init__(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        yhat: str,
        y: Optional[str] = None,
        features: Optional[List[str]] = None,
        options: Optional[Dict] = {},
        thresholds: Optional[Dict] = {},
    ):
        self.base_df = base_df
        self.current_df = current_df
        self.yhat_base = yhat
        self.yhat_curr = yhat
        self.y_base = y
        self.y_curr = y
        self.features = features
        self.options = options
        self.thresholds = thresholds

        self.features = self._set_features()
        if len(self.options):
            self.options["max_levels"] = options["max_levels"]
        else:
            self.options["max_levels"] = 0.05

        self.validation_flag, self.violations = self._validate_inputs()
        self._get_data_types(self.options["max_levels"])

        self.drift_metrics = DRIFT_OPTIONS.copy()
        self.threshold_options = THRESHOLD_OPTIONS.copy()
        self.summary_options = SUMMARY_OPTIONS.copy()
        self.glossary_dataframe = GLOSSARY_DATAFRAME.copy()

        # Override default options with user provided "thresholds"
        if self.thresholds:
            self.threshold_options = update_threshold_options(
                thresholds=self.thresholds
            )
            self.summary_options = update_summary_with_thresholds(
                threshold_options=self.threshold_options
            )
            self.glossary_dataframe = update_glossary_df_with_thresholds(
                threshold_options=self.threshold_options
            )

        self.target_drift = None
        self.num_feature_drift = None
        self.cat_feature_drift = None
        self.num_concept_drift = None
        self.cat_concept_drift = None
        self.population_stability_index = {}
        self.feature_stability_index = {}
        self.dependency_stability_index = {}
        self.descriptive_stats = {}
        self.drift_summary = None

        _LOGGER.info("Initiated DriftMetrics Class")

    def _validate_inputs(self):
        """Validate all inputs."""
        # Assume data is valid and negate it if violations found
        validation_flag = True
        violations = []
        # Validation: Check types of input parameters
        if not (
            (type(self.base_df) == pd.DataFrame)
            and (type(self.current_df) == pd.DataFrame)
            and (type(self.yhat_base) == str)
            and (type(self.yhat_curr) == str)
        ):
            validation_flag = False
            violations.append("Validate the type of input parameters")
        # Validation: If y_base is provided, y_curr should also be provided OR both should not be provided(XNOR operation)
        if not (
            (self.y_base and self.y_curr) or ((not self.y_base) and (not self.y_curr))
        ):
            validation_flag = False
            violations.append(
                "If y_base is provided, y_curr should also be provided OR both should not be provided"
            )
        # If both are not provided, pass
        elif not (self.y_base):
            pass
        # Validation: If both y_base,y_curr are provided, their type should be string
        elif not (isinstance(self.y_base, str) and isinstance(self.y_curr, str)):
            validation_flag = False
            violations.append(
                "If y_base,y_curr are provided, thier type should be string"
            )

        # Validation: Feature list of base_df and current_df should be same
        if not (len(self.base_df.columns.difference(self.current_df.columns)) == 0):
            validation_flag = False
            violations.append(
                "Feature list/Column names of base_df and current_df are not same"
            )
        if self.features:
            # Validation: If 'features' param is given, it should be a list of strings
            if not all(isinstance(n, str) for n in self.features):
                validation_flag = False
                violations.append(
                    "If features param is given, it should be a list of strings"
                )
            # Validation: If 'features' param is given, it should be a subset of both base_df and current_df columns
            if not (set(self.features).issubset(self.base_df.columns)) and (
                set(self.features).issubset(self.current_df.columns)
            ):
                validation_flag = False
                violations.append(
                    "If features param is given, it should be a subset of both base_df and current_df"
                )
        if self.options:
            # Validation: if options are given, it should be a dictionary
            if not (type(self.options) == dict):
                validation_flag = False
                violations.append("if options are given, it should be a dictionary")
        if self.thresholds:
            # Validation: if thresholds are given, it should be a dictionary
            if not (type(self.thresholds) == dict):
                validation_flag = False
                violations.append("if thresholds are given, it should be a dictionary")
        if not validation_flag:
            _LOGGER.error(f"Validation Flag: {validation_flag}")
            [_LOGGER.error(f"Violation found: {violation}") for violation in violations]
            raise Exception(
                [Exception(f"ValidationError: {violation}") for violation in violations]
            )
        else:
            _LOGGER.info(f"Validation Flag: {validation_flag}")
        return validation_flag, violations

    def _get_data_types(self, max_levels):

        # Validation: yhat_base dtype & yhat_curr dtype should be same
        yhat_curr_type = get_data_type(
            self.current_df[self.yhat_curr], max_levels=max_levels
        )
        if yhat_curr_type == "boolean":
            yhat_curr_type = "categorical"

        yhat_base_type = get_data_type(
            self.base_df[self.yhat_base], max_levels=max_levels
        )
        if yhat_base_type == "boolean":
            yhat_base_type = "categorical"
            _LOGGER.info("Target is categorical")

        if not (yhat_base_type == yhat_curr_type):
            self.validation_flag = False
            self.violations.append("yhat_base dtype & yhat_curr dtype should be same")

        # Assign after validation
        self.target_type = yhat_base_type

        # Validation: If y_base & y_curr are given, thier dtypes should be the same
        if self.y_base:
            y_curr_type = get_data_type(
                self.current_df[self.y_curr], max_levels=max_levels
            )
            if y_curr_type == "boolean":
                y_curr_type = "categorical"
            y_base_type = get_data_type(
                self.base_df[self.y_base], max_levels=max_levels
            )
            if y_base_type == "boolean":
                y_base_type = "categorical"
                _LOGGER.info("Actual Target is categorical")
            if not (y_base_type == y_curr_type):
                self.validation_flag = False
                self.violations.append(
                    "If y_base & y_curr are given, thier dtypes should be the same"
                )

            # Assign after validation
            self.actual_target_type = y_base_type

        base_feature_df = self.base_df[self.features]
        curr_feature_df = self.current_df[self.features]
        # reason for adding result_type param in apply is that .apply() changes the dtype of series to object and is causing issues in recognizing the data_type correctly
        # https://stackoverflow.com/questions/34917404/why-does-apply-change-dtype-in-pandas-dataframe-columns
        base_data_type = base_feature_df.apply(
            func=get_data_type, result_type="expand", axis=0, max_levels=max_levels
        )
        curr_data_type = curr_feature_df.apply(
            func=get_data_type, result_type="expand", axis=0, max_levels=max_levels
        )

        # Validation: The dtypes of base_df features & current_df features should be same
        base_bool_features = base_data_type.index[base_data_type == "boolean"]
        base_cat_features = base_data_type.index[base_data_type == "categorical"]
        base_num_features = base_data_type.index[base_data_type == "numerical"]
        curr_bool_features = curr_data_type.index[curr_data_type == "boolean"]
        curr_cat_features = curr_data_type.index[curr_data_type == "categorical"]
        curr_num_features = curr_data_type.index[curr_data_type == "numerical"]
        bool_diff = base_bool_features.difference(curr_bool_features).tolist()
        cat_diff = base_cat_features.difference(curr_cat_features).tolist()
        num_diff = base_num_features.difference(curr_num_features).tolist()
        tot_diff = bool_diff + cat_diff + num_diff
        if not (len(tot_diff) == 0):
            self.validation_flag = False
            mismatch_dtype_df = pd.DataFrame(
                columns=[
                    "Column Name",
                    "Detected dtype in Base",
                    "Detected dtype in Current",
                    "pd dtype in Base",
                    "pd dtype in Current",
                ]
            )
            for col in tot_diff:
                row = {
                    "Column Name": col,
                    "Detected dtype in Base": base_data_type.loc[col],
                    "Detected dtype in Current": curr_data_type.loc[col],
                    "pd dtype in Base": self.base_df[col].dtype,
                    "pd dtype in Current": self.current_df[col].dtype,
                }
                mismatch_dtype_df = mismatch_dtype_df.append(row, ignore_index=True)
            self.violations.append(
                f"The dtypes of these columns is different in base_df  & current_df. \n {mismatch_dtype_df}"
            )

        # Assign after validation
        self.bool_features = base_bool_features.tolist()
        self.cat_features = base_cat_features.tolist()
        self.num_features = base_num_features.tolist()
        _LOGGER.info("Calculated the types of features")

        # Raise ValidationError if validation_flag=False
        if not self.validation_flag:
            _LOGGER.error(f"Validation Flag: {self.validation_flag}")
            [
                _LOGGER.error(f"Violation found: {violation}")
                for violation in self.violations
            ]
            [
                warnings.warn(f"ValidationError: {violation}")
                for violation in self.violations
            ]
        else:
            _LOGGER.info(f"Validation Flag: {self.validation_flag}")

    def _set_features(self):
        exclude = {self.yhat_base, self.yhat_curr, self.y_base, self.y_curr}

        if self.features is None:
            features = self.base_df.columns.tolist()
        else:
            features = self.features
        features = [x for x in features if x not in exclude]
        _LOGGER.info("Get all the features to be used")
        return features

    def _get_glossary(self, glossary_path=None):
        glossary_dataframe = self.glossary_dataframe
        return glossary_dataframe

    def _compute_drift_summary(self, summary_options=None):
        if summary_options is None:
            summary_options = self.summary_options

        if self.target_drift is None:
            self._get_target_drift_legacy()
            _LOGGER.info("Calculated target drift")

        if self.num_feature_drift is None:
            self._get_feature_drift_legacy()
            _LOGGER.info("Calculated feature drift")

        if self.num_concept_drift is None:
            self._get_concept_drift_legacy()
            _LOGGER.info("Calculated numerical features concept drift")

        target_summary = apply_threshold(
            self.target_drift, "target_drift", summary_options
        )
        _LOGGER.info("Calculated target drift summary")

        if isinstance(self.num_feature_drift, pd.DataFrame):
            num_feature_summary = apply_threshold(
                self.num_feature_drift, "feature_drift_numerical", summary_options
            )
            _LOGGER.info("Calculated numerical feature drift summary")
            print(num_feature_summary)

        else:
            num_feature_summary = {}

        if isinstance(self.cat_feature_drift, pd.DataFrame):
            cat_feature_summary = apply_threshold(
                self.cat_feature_drift, "feature_drift_categorical", summary_options
            )
            _LOGGER.info("Calculated categorical feature drift summary")
        else:
            cat_feature_summary = {}

        if isinstance(self.num_concept_drift, pd.DataFrame):
            num_concept_summary = apply_threshold(
                self.num_concept_drift, "concept_drift_numerical", summary_options
            )
            _LOGGER.info("Calculated numerical concept features drift summary")

        else:
            num_concept_summary = {}

        if isinstance(self.cat_concept_drift, pd.DataFrame):
            cat_concept_summary = apply_threshold(
                self.cat_concept_drift, "concept_drift_categorical", summary_options
            )
            _LOGGER.info("Calculated categorical concept feature drift summary")
        else:
            cat_concept_summary = {}

        drift_summary_dict = {
            "target_drift": target_summary,
            "total_features": len(self.features),
            "total_numeric_features": len(self.num_features),
            "total_boolean_features": len(self.bool_features),
            "total_categorical_features": len(self.cat_features),
            "feature_drift_numerical": num_feature_summary,
            "feature_drift_categorical": cat_feature_summary,
            "concept_drift_numerical": num_concept_summary,
            "concept_drift_categorical": cat_concept_summary,
        }
        drift_summary_dict = self._calculate_total_feature_concept_drift(
            drift_summary_dict
        )
        _LOGGER.info("Drift summary dictionary created")

        return drift_summary_dict

    def _compute_num_data_summary(self):

        num_summary = None

        if len(self.num_features):
            num_stats = compare_num_stats(
                self.base_df, self.current_df, features=self.num_features
            )
            num_stats = num_stats.set_index("variable")
            num_stats_base = num_stats.filter(regex="_base")
            num_stats_curr = num_stats.filter(regex="_curr")

            cols = {x.replace("_base", "") for x in num_stats_base.columns}
            num_stats_base.columns = cols

            cols = {x.replace("_curr", "") for x in num_stats_curr.columns}
            num_stats_curr.columns = cols
            percentage_diff = (
                ((num_stats_curr - num_stats_base) / num_stats_base)
                .reset_index()
                .rename(columns={"index": "variable"})
            )

            num_summary = percentage_diff[["variable", "mean", "std"]].rename(
                columns={"mean": "perc_mean_diff", "std": "perc_std_diff"}
            )
            _LOGGER.info("Numerical features data summary calculated")
        else:
            num_summary = pd.DataFrame(
                columns=["variable", "perc_mean_diff", "perc_std_diff"]
            )
            _LOGGER.info("No numerical features in df")

        return num_summary

    def _compute_bool_data_summary(self):

        bool_summary = None
        if len(self.bool_features):
            bool_stats = compare_bool_stats(
                self.base_df, self.current_df, features=self.bool_features
            )
            bool_stats = bool_stats.set_index("variable")
            bool_stats_base = bool_stats.filter(regex="_base")
            bool_stats_curr = bool_stats.filter(regex="_curr")

            cols = {x.replace("_base", "") for x in bool_stats_base.columns}
            bool_stats_base.columns = cols

            cols = {x.replace("_curr", "") for x in bool_stats_curr.columns}
            bool_stats_curr.columns = cols

            percentage_diff = (bool_stats_curr - bool_stats_base) / bool_stats_base
            bool_summary = percentage_diff.rename(
                columns={
                    "count_0s": "perc_count_diff(0)",
                    "count_1s": "perc_count_diff(1)",
                }
            )
            bool_summary = bool_summary.reset_index().rename(
                columns={"index": "variable"}
            )
            _LOGGER.info("Boolean features data summary calculated")
        else:
            bool_summary = pd.DataFrame(
                columns=["variable", "perc_count_diff(0)", "perc_count_diff(1)"]
            )
            _LOGGER.info("No boolean features in df")

        return bool_summary

    def _compute_cat_data_summary(self):
        if not len(self.descriptive_stats):
            self.descriptive_stats = self.get_descriptive_stats()
            _LOGGER.info("Descriptive Stats Created")
        cat_summary = None
        # FIXME: This if/else block is failing
        if len(self.cat_features):
            cat_summary = self.descriptive_stats["cat_features"][
                ["variable", "mode_base", "mode_curr"]
            ]
            _LOGGER.info("Categorical features data summary calculated")
        else:
            cat_summary = pd.DataFrame(columns=["variable", "mode_base", "mode_curr"])
            _LOGGER.info("No categorical features in df")
        return cat_summary

    def _compute_data_summary(self):
        if not len(self.descriptive_stats):
            self.descriptive_stats = self.get_descriptive_stats()
            _LOGGER.info("Descriptive Stats Created")

        num_summary = self._compute_num_data_summary()

        bool_summary = self._compute_bool_data_summary()

        cat_summary = self._compute_cat_data_summary()

        data_summary = {
            "numerical_features": num_summary,
            "boolean_features": bool_summary,
            "categorical_features": cat_summary,
        }
        _LOGGER.info("Data summary for features calculated")
        return data_summary

    def _compute_drift_metrics(self, metrics_list, feature_list, feature_type):
        var_level = defaultdict(lambda: defaultdict(dict))
        bin_level = defaultdict(lambda: defaultdict(dict))

        # list of metrics which need both feature and target
        need_xy_both = ["DSI"]
        need_data_type = ["PSI", "DSI"]

        for metric in metrics_list:
            metric_details = self.drift_metrics[metric]
            func = metric_details["func"]
            default_params = metric_details.get("default_params", {})
            if metric in need_data_type:
                default_params["feature_data_type"] = feature_type

            for feature in feature_list:
                if metric in need_xy_both:
                    base = self.base_df[[feature, self.yhat_base]]
                    current = self.current_df[[feature, self.yhat_curr]]
                    base.columns = ["Feature", "Target"]
                    current.columns = ["Feature", "Target"]
                    default_params["target_data_type"] = self.target_type
                else:
                    base = self.base_df[feature]
                    current = self.current_df[feature]

                value = func(base, current, **default_params)

                if isinstance(value, dict):
                    if "bin_level" in value.keys():
                        var_level[feature][metric.lower()] = value["aggregated"]
                        bin_level[metric.lower()][feature] = value["bin_level"]
                    else:
                        var_level[feature][metric.lower()] = value
                else:
                    # Specifically put in to handle kldivergence_value ==> kldivergence
                    var_level[feature][metric.lower()] = value

        var_level = {
            feature: pd.json_normalize(value, sep="_")
            for feature, value in var_level.items()
        }

        var_level = concat_dfs(var_level, names=["variable"])
        metric_dict = {"var_level": var_level, "bin_level": bin_level}
        _LOGGER.info("Calculated drift metrics dictionary for data")
        return metric_dict

    def _compute_target_drift(self):
        metrics_applicable = get_applicable_metrics(
            self.drift_metrics, data_type=self.target_type, drift_type="target_drift"
        )

        target_drift = self._compute_drift_metrics(
            metrics_list=metrics_applicable,
            feature_list=[self.yhat_base],
            feature_type=self.target_type,
        )

        return target_drift

    def _compute_target_drift_at_var_level(self):
        metrics_applicable = get_applicable_metrics(
            self.drift_metrics, data_type=self.target_type, drift_type="target_drift"
        )

        target_drift = self._compute_drift_metrics(
            metrics_list=metrics_applicable,
            feature_list=[self.yhat_base],
            feature_type=self.target_type,
        )

        return target_drift["var_level"]

    def _compute_target_drift_at_bin_level(self):
        metrics_applicable = get_applicable_metrics(
            self.drift_metrics, data_type=self.target_type, drift_type="target_drift"
        )

        target_drift = self._compute_drift_metrics(
            metrics_list=metrics_applicable,
            feature_list=[self.yhat_base],
            feature_type=self.target_type,
        )

        return target_drift["bin_level"]["psi"][self.yhat_base]

    def _compute_feature_drift(self):
        if len(self.num_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="numerical", drift_type="feature_drift"
            )
            num_feature_drift = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.num_features,
                feature_type="numerical",
            )
        else:
            num_feature_drift = {}
            _LOGGER.info("No numerical feature drift")

        if len(self.cat_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="categorical", drift_type="feature_drift"
            )
            cat_feature_drift = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.cat_features,
                feature_type="categorical",
            )

        else:
            cat_feature_drift = {}
            _LOGGER.info("No categorical fetaure drift")

        return num_feature_drift, cat_feature_drift

    def _compute_num_feature_drift(self):
        if len(self.num_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="numerical", drift_type="feature_drift"
            )
            num_feature_drift = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.num_features,
                feature_type="numerical",
            )["var_level"]
        else:
            num_feature_drift = pd.DataFrame(
                columns=[
                    "variable",
                    "psi",
                    "kldivergence",
                    "anderson_stats",
                    "anderson_pvalue",
                    "ks_stats",
                    "ks_pvalue",
                ]
            )
            _LOGGER.info("No numerical feature drift")
        return num_feature_drift

    def _compute_num_feature_drift_at_bin_level(self):
        if len(self.num_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="numerical", drift_type="feature_drift"
            )
            num_feature_drift_at_bin_level_dict = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.num_features,
                feature_type="numerical",
            )["bin_level"]["psi"]
            num_feature_drift_at_bin_level_df = concat_dfs(
                num_feature_drift_at_bin_level_dict, names=["feature"]
            )
        else:
            num_feature_drift_at_bin_level_df = pd.DataFrame(
                columns=[
                    "feature",
                    "bins_or_categories",
                    "count_base",
                    "count_current",
                    "perc_base",
                    "perc_current",
                    "psi",
                ]
            )
            _LOGGER.info("No numerical feature drift")
        return num_feature_drift_at_bin_level_df

    def _compute_cat_feature_drift(self):
        if len(self.cat_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="categorical", drift_type="feature_drift"
            )
            cat_feature_drift = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.cat_features,
                feature_type="categorical",
            )["var_level"]

        else:
            cat_feature_drift = pd.DataFrame(
                columns=["variable", "psi", "chisquare_stats", "chisquare_pvalue"]
            )
            _LOGGER.info("No categorical feature drift")
        return cat_feature_drift

    def _compute_cat_feature_drift_at_bin_level(self):
        if len(self.cat_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="categorical", drift_type="feature_drift"
            )
            cat_feature_drift_at_bin_level_dict = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.cat_features,
                feature_type="categorical",
            )["bin_level"]["psi"]
            cat_feature_drift_at_bin_level_df = concat_dfs(
                cat_feature_drift_at_bin_level_dict, names=["feature"]
            )
        else:
            cat_feature_drift_at_bin_level_df = pd.DataFrame(
                columns=[
                    "feature",
                    "bins_or_categories",
                    "count_base",
                    "count_current",
                    "perc_base",
                    "perc_current",
                    "psi",
                ]
            )
            _LOGGER.info("No numerical feature drift")
        return cat_feature_drift_at_bin_level_df

    def _compute_concept_drift(self):
        if len(self.num_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="numerical", drift_type="concept_drift"
            )
            num_concept_drift = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.num_features,
                feature_type="numerical",
            )

        else:
            num_concept_drift = {}
            _LOGGER.info("No numerical features concept drift")

        if len(self.cat_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="categorical", drift_type="concept_drift"
            )
            cat_concept_drift = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.cat_features,
                feature_type="categorical",
            )

        else:
            cat_concept_drift = {}
            _LOGGER.info("No categorical features concept drift")

        return num_concept_drift, cat_concept_drift

    def _compute_num_concept_drift(self):
        if len(self.num_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="numerical", drift_type="concept_drift"
            )
            num_concept_drift = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.num_features,
                feature_type="numerical",
            )["var_level"]

        else:
            num_concept_drift = pd.DataFrame(columns=["variable", "dsi"])
            _LOGGER.info("No numerical features concept drift")
        return num_concept_drift

    def _compute_num_concept_drift_at_bin_level(self):
        if len(self.num_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="numerical", drift_type="concept_drift"
            )
            num_concept_drift_at_bin_level_dict = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.num_features,
                feature_type="numerical",
            )["bin_level"]["dsi"]
            num_concept_drift_at_bin_level_df = concat_dfs(
                num_concept_drift_at_bin_level_dict, names=["feature"]
            )
        else:
            num_concept_drift_at_bin_level_df = pd.DataFrame(
                columns=[
                    "feature",
                    "feature_bin",
                    "target_bin",
                    "count_base",
                    "count_current",
                    "perc_base",
                    "perc_current",
                    "dsi",
                ]
            )
            _LOGGER.info("No numerical features concept drift")
        return num_concept_drift_at_bin_level_df

    def _compute_cat_concept_drift(self):
        if len(self.cat_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="categorical", drift_type="concept_drift"
            )
            cat_concept_drift = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.cat_features,
                feature_type="categorical",
            )["var_level"]
        else:
            cat_concept_drift = pd.DataFrame(columns=["variable", "dsi"])
            _LOGGER.info("No categorical features concept drift")
        return cat_concept_drift

    def _compute_cat_concept_drift_at_bin_level(self):
        if len(self.cat_features):
            metrics_applicable = get_applicable_metrics(
                self.drift_metrics, data_type="categorical", drift_type="concept_drift"
            )
            cat_concept_drift_at_bin_level_dict = self._compute_drift_metrics(
                metrics_list=metrics_applicable,
                feature_list=self.cat_features,
                feature_type="categorical",
            )["bin_level"]["dsi"]
            cat_concept_drift_at_bin_level_df = concat_dfs(
                cat_concept_drift_at_bin_level_dict, names=["feature"]
            )
        else:
            cat_concept_drift_at_bin_level_df = pd.DataFrame(
                columns=[
                    "feature",
                    "feature_bin",
                    "target_bin",
                    "count_base",
                    "count_current",
                    "perc_base",
                    "perc_current",
                    "dsi",
                ]
            )
            _LOGGER.info("No categorical features concept drift")
        return cat_concept_drift_at_bin_level_df

    def _rename_drift_indicator_keys(self, drift_summary_dict):
        for key, value in drift_summary_dict.items():
            if isinstance(value, dict):
                self._rename_drift_indicator_keys(value)
            if key in ("High", "Moderate"):
                drift_summary_dict[key + " Drift"] = drift_summary_dict.pop(key)
            if key == "Low":
                drift_summary_dict["No Drift"] = drift_summary_dict.pop(key)
        return drift_summary_dict

    def _calculate_total_feature_concept_drift(self, drift_summary_dict):
        drift_summary_dict = self._rename_drift_indicator_keys(drift_summary_dict)

        # Handle None
        if drift_summary_dict["feature_drift_numerical"] is None:
            drift_summary_dict["feature_drift_numerical"] = 0
            drift_summary_dict["concept_drift_numerical"] = 0
            _LOGGER.info("No numerical features found in data")
        if drift_summary_dict["feature_drift_categorical"] is None:
            drift_summary_dict["feature_drift_categorical"] = 0
            drift_summary_dict["concept_drift_categorical"] = 0
            _LOGGER.info("No categorical features found in data")

        # Create "feature_drift"
        if isinstance(
            drift_summary_dict["feature_drift_numerical"], dict
        ) and isinstance(drift_summary_dict["feature_drift_categorical"], dict):
            for key in drift_summary_dict["feature_drift_numerical"]:
                if key in drift_summary_dict["feature_drift_categorical"]:
                    drift_summary_dict["feature_drift_numerical"][key] = (
                        drift_summary_dict["feature_drift_numerical"][key]
                        + drift_summary_dict["feature_drift_categorical"][key]
                    )
            drift_summary_dict["feature_drift"] = drift_summary_dict[
                "feature_drift_numerical"
            ]
            del drift_summary_dict["feature_drift_numerical"]
            del drift_summary_dict["feature_drift_categorical"]
        elif isinstance(
            drift_summary_dict["feature_drift_numerical"], np.int64
        ) and isinstance(drift_summary_dict["feature_drift_categorical"], np.int64):
            drift_summary_dict["feature_drift"] = (
                drift_summary_dict["feature_drift_numerical"]
                + drift_summary_dict["feature_drift_categorical"]
            )
            del drift_summary_dict["feature_drift_numerical"]
            del drift_summary_dict["feature_drift_categorical"]

        # Create "concept_drift"
        if isinstance(
            drift_summary_dict["concept_drift_numerical"], dict
        ) and isinstance(drift_summary_dict["concept_drift_categorical"], dict):
            for key in drift_summary_dict["concept_drift_numerical"]:
                if key in drift_summary_dict["concept_drift_categorical"]:
                    drift_summary_dict["concept_drift_numerical"][key] = (
                        drift_summary_dict["concept_drift_numerical"][key]
                        + drift_summary_dict["concept_drift_categorical"][key]
                    )
            drift_summary_dict["concept_drift"] = drift_summary_dict[
                "concept_drift_numerical"
            ]
            del drift_summary_dict["concept_drift_numerical"]
            del drift_summary_dict["concept_drift_categorical"]
        elif isinstance(
            drift_summary_dict["concept_drift_numerical"], np.int64
        ) and isinstance(drift_summary_dict["concept_drift_categorical"], np.int64):
            drift_summary_dict["concept_drift"] = (
                drift_summary_dict["concept_drift_numerical"]
                + drift_summary_dict["concept_drift_categorical"]
            )
            del drift_summary_dict["concept_drift_numerical"]
            del drift_summary_dict["concept_drift_categorical"]

        # Convert any key that has a numerical value(except total_features) to {"Drift": , "No Drift":}
        # By end of this block, all values in drift_summary_dict are converted to dict
        for key, value in drift_summary_dict.items():
            if (
                key
                not in (
                    "total_features",
                    "total_numeric_features",
                    "total_categorical_features",
                    "total_boolean_features",
                )
            ) and (not isinstance(value, dict)):
                if key in ("feature_drift_numerical", "concept_drift_numerical"):
                    drift_summary_dict[key] = {
                        "Drift": value,
                        "No Drift": drift_summary_dict["total_numeric_features"]
                        - value,
                    }
                elif key in ("feature_drift_categorical", "concept_drift_categorical"):
                    drift_summary_dict[key] = {
                        "Drift": value,
                        "No Drift": drift_summary_dict["total_categorical_features"]
                        - value,
                    }
                else:
                    drift_summary_dict[key] = {
                        "Drift": value,
                        "No Drift": drift_summary_dict["total_features"] - value,
                    }

        # convert target_drift to a string instead of dict
        for k, v in drift_summary_dict["target_drift"].items():
            if v != 0:
                drift_summary_dict["target_drift"] = k

        return drift_summary_dict
