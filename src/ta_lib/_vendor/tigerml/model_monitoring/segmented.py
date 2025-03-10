import logging
import numpy as np
import pandas as pd
import pdb
import time
from collections import defaultdict
from tigerml.core.reports import create_report
from tigerml.core.utils import measure_time
from tigerml.core.utils.segmented import (
    calculate_all_segments,
    get_segment_filter,
)
from tigerml.model_monitoring.base_drift import BaseDrift
from tigerml.model_monitoring.model_drift import ModelDrift
from tigerml.model_monitoring.plotters.plot import get_heatmap
from tigerml.model_monitoring.utils.data_utils import concat_dfs
from tigerml.model_monitoring.utils.highlighting import table_formatter
from typing import Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class SegmentedModelDrift(BaseDrift):
    """
    Segmented Class for creation of target, feature and concept Drift.

    Calculate target, feature and concept drift and statistical for data with
    any segment that could be a combination of multiple columns.

    Parameters
    ----------
    base_df: pd.DataFarme
        Base data / reference data
    current_df: pd.DataFrame
        Current data for which you want to calculate shift
    yhat: str
        Predicted target column name for data
    segment_by: List[str]
        List of columns used for segmenting
    y: str
        Actual target column name for data
    features: List[str]
        List of features for which you want to calculate drift

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from tigerml.model_monitoring import SegmentedModelDrift
    >>> data = load_diabetes(as_frame=True)
    >>> X, y = data["data"], data["target"]
    >>> X_base, X_curr, y_base, y_curr = train_test_split(
    ...    X, y, test_size=0.33, random_state=42
    ... )
    >>> model = LogisticRegression()
    >>> model.fit(X_base, y_base)
    >>> yhat_base = model.predict(X_base)
    >>> yhat_curr = model.predict(X_curr)
    >>> base_df = pd.concat([X_base, y_base], axis=1)
    >>> base_df.loc[:, "predicted_target"] = yhat_base
    >>> curr_df = pd.concat([X_curr, y_curr], axis=1)
    >>> curr_df.loc[:, "predicted_target"] = yhat_curr
    >>> base_df["sex"] = np.where(base_df["sex"] >= 0, "Male", "Female")
    >>> curr_df["sex"] = np.where(curr_df["sex"] >= 0, "Male", "Female")
    >>> segmented_mm = SegmentedModelDrift(
    ... base_df=base_df,
    ... current_df=curr_df,
    ... yhat="predicted_target",
    ... y="target",
    ... segment_by=["sex"],
    >>> segmented_mm.get_report()
    """

    def __init__(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        yhat: str,
        y: Optional[str] = None,
        features: Optional[List[str]] = None,
        segment_by: Optional[List[str]] = None,
        options: Optional[Dict] = {},
    ):
        self.segment_by = segment_by
        super().__init__(
            base_df=base_df,
            current_df=current_df,
            yhat=yhat,
            y=y,
            features=features,
            options=options,
        )
        self.each_segment_drift = {}
        self._initialise_segment_drift()
        _LOGGER.info("Initiated the SegmentedModelDrift Class")

    def _validate_inputs(self):
        self.validation_flag, self.violations = super()._validate_inputs()
        if self.segment_by:
            # SegmentedValidation: segment_by should be a list of str or int
            if not (
                all(isinstance(n, str) for n in self.segment_by)
                or all(isinstance(n, int) for n in self.segment_by)
            ):
                self.validation_flag = False
                self.violations.append("segment_by should be a list of str or int")
        if not self.validation_flag:
            _LOGGER.error(f"Validation Flag: {self.validation_flag}")
            [
                _LOGGER.error(f"Violation found: {violation}")
                for violation in self.violations
            ]
            raise Exception(
                [
                    Exception(f"ValidationError: {violation}")
                    for violation in self.violations
                ]
            )
        else:
            _LOGGER.info(f"Validation Flag: {self.validation_flag}")
        return self.validation_flag, self.violations

    def _get_drift_report(self, psi_heatmap, fsi_heatmap, dsi_heatmap):
        try:
            data_summary = {
                key: table_formatter(value, format_bg_color=False)
                for key, value in self.data_summary.items()
            }
            drift_report = {
                "summary": {
                    "data_summary": data_summary,
                    "drift_summary": {
                        "overall": self.drift_summary["overall"],
                        "segmented": self.drift_summary["segmented"],
                    },
                },
                "model_drift": {
                    "target_desc_stats": table_formatter(
                        self.descriptive_stats["target"], format_bg_color=False
                    ),
                    "target_drift": table_formatter(self.target_drift, self.thresholds),
                    "concept_drift": {
                        "numerical_features": table_formatter(
                            self.num_concept_drift, format_bg_color=False
                        ),
                        "categorical_features": table_formatter(
                            self.cat_concept_drift, format_bg_color=False
                        ),
                    },
                    "performance_drift": "Not applicable",
                    # "psi_heatmap": psi_heatmap,
                    # "dsi_heatmap": dsi_heatmap,
                },
                "data_drift": {
                    "feature_desc_stats": {
                        "numerical_features": table_formatter(
                            self.descriptive_stats["num_features"],
                            format_bg_color=False,
                        ),
                        "categorical_features": table_formatter(
                            self.descriptive_stats["cat_features"],
                            format_bg_color=False,
                        ),
                        "cat_features_set_analysis": self.descriptive_stats[
                            "cat_features_set_analysis"
                        ],
                    },
                    "feature_drift": {
                        "numerical_features": table_formatter(
                            self.num_feature_drift, self.thresholds
                        ),
                        "categorical_features": table_formatter(
                            self.cat_feature_drift, self.thresholds
                        ),
                    },
                    # "fsi_heatmap": fsi_heatmap,
                },
            }
            _LOGGER.info("drift report generated")
        except Exception as e:
            _LOGGER.error(e, "object not found")
        return drift_report

    def _initialise_segment_drift(self):
        self.all_segments = calculate_all_segments(self.base_df, self.segment_by)
        self.all_segments = [tuple(segment) for segment in self.all_segments]
        for each_segment in self.all_segments:
            base_df_sub = self.base_df[
                get_segment_filter(
                    self.base_df, segment_by=self.segment_by, segment=each_segment
                )
            ]
            current_df_sub = self.current_df[
                get_segment_filter(
                    self.current_df, segment_by=self.segment_by, segment=each_segment
                )
            ]

            self.each_segment_drift[each_segment] = ModelDrift(
                base_df=base_df_sub,
                current_df=current_df_sub,
                yhat=self.yhat_base,
                y=self.y_curr,
                features=self.features,
                options=self.options,
            )
        _LOGGER.info(
            "Initiated the BaseModelDrift Class for each of the \
        segment and one for entire data"
        )

    def _get_drift_summary(self, summary_options=None):
        """
        Get Overall and Segmented drift summary.

        By using standard threshold, configurable from summary_options
        file it calculates whether their is a drift or not for all the three
        drift types.

        Parameteres
        -----------
        summary_options: dict
            dictionary containing thresholds for each of the drift.

        Returns
        -------
        Dictionary containing two keys one for overall and other segmented
        containing drift at overall and segmented level.

        """

        obj = ModelDrift(
            base_df=self.base_df,
            current_df=self.current_df,
            yhat=self.yhat_base,
            y=self.yhat_curr,
            features=self.features,
            options=self.options,
        )

        drift_overall_summary = obj._get_drift_summary_legacy()

        # if drift_overall_summary["total_numeric_features"] is None:
        #     drift_overall_summary_dict[
        #         "feature_drift_numerical"
        #     ] = "No Numerical Feature "
        #     drift_overall_summary["concept_drift_numerical"] = "No Numerical Feature "
        # if drift_overall_summary["total_categorical_features"] is None:
        #     drift_overall_summary[
        #         "feature_drift_categorical"
        #     ] = "No Categorical Feature "
        #     drift_overall_summary[
        #         "concept_drift_categorical"
        #     ] = "No Categorical Feature "

        # already a pandas dataframe
        # drift_overall_summary = pd.DataFrame(drift_overall_summary_dict, index=[0])

        dict = {}
        for each_segment in self.all_segments:
            dict[str(each_segment)] = self.each_segment_drift[
                each_segment
            ]._get_drift_summary_legacy()

        drift_segmented_summary = concat_dfs(dict, names=["segment"])

        drift_segmented_summary = pd.DataFrame(
            drift_segmented_summary.astype(bool).sum(axis=0)
        ).T

        if len(self.num_features) == 0:
            drift_segmented_summary["feature_drift_numerical"] = "No Numerical Feature "
            drift_segmented_summary["concept_drift_numerical"] = "No Numerical Feature "
        if len(self.cat_features) == 0:
            drift_segmented_summary[
                "feature_drift_categorical"
            ] = "No Categorical Feature "
            drift_segmented_summary[
                "concept_drift_categorical"
            ] = "No Categorical Feature "

        drift_segmented_summary.columns = [
            "{}(#NoOfSegments)".format(column)
            for column in drift_segmented_summary.columns
        ]

        self.drift_summary = {
            "overall": drift_overall_summary,
            "segmented": drift_segmented_summary,
        }

        _LOGGER.info("Drift Summary Generated")

        return self.drift_summary

    def get_data_summary(self):
        """
        Get Overall Data Summary for numerical and boolean features.

        For categorical features still need to be supported.

        Returns
        -------
            Dictionary with two keys one for numerical and other for overall.
        """

        # data_summary = self._compute_data_summary()
        data_summary = ModelDrift(
            base_df=self.base_df,
            current_df=self.current_df,
            yhat=self.yhat_base,
            y=self.y_base,
            features=self.features,
        )._compute_data_summary()

        if data_summary["numerical_features"] is None:
            data_summary["numerical_features"] = "No Numerical Feature "
        if data_summary["boolean_features"] is None:
            data_summary["boolean_features"] = "No boolean Feature "
        if data_summary["categorical_features"] is None:
            data_summary["categorical_features"] = "No categorical features "
        self.data_summary = data_summary
        _LOGGER.info("Data Summary Generated")
        return self.data_summary

    def get_target_desc_stats(self):
        """
        Get target descriptive stats.

        Returns
        -------
        target_summary_df: pd.DataFrame
            Returns pd.DataFrame containing target_summary of each segment.
        """
        dict = {}
        for each_segment in self.all_segments:
            dict[str(each_segment)] = self.each_segment_drift[
                each_segment
            ].get_predicted_target_desc_stats_df()
        target_summary_df = concat_dfs(dict, names=["segment"])
        _LOGGER.info("Target Descriptive Scripts Generated")
        return target_summary_df

    def get_num_desc_stats(self):
        """
        Get numerical descriptive stats.

        Returns
        -------
        num_summary_df: pd.DataFrame
            Returns pd.DataFrame containing num_summary of each segment.
        """
        if len(self.num_features):
            dict = {}
            for each_segment in self.all_segments:
                dict[str(each_segment)] = self.each_segment_drift[
                    each_segment
                ].get_num_features_desc_stats_df()
            num_summary_df = concat_dfs(dict, names=["segment"])
        else:
            num_summary_df = "No numerical feature "
        _LOGGER.info("Numerical Descriptive Scripts Generated")
        return num_summary_df

    def get_cat_desc_stats(self):
        """
        Get categorical descriptive statistics.

        Returns
        -------
        cat_summary_df: pd.DataFrame
            Returns pd.DataFrame containing cat_summary of each segment.
        """
        if len(self.cat_features):
            dict = {}
            for each_segment in self.all_segments:
                dict[str(each_segment)] = self.each_segment_drift[
                    each_segment
                ].get_cat_features_desc_stats_df()
            cat_summary_df = concat_dfs(dict, names=["segment"])
        else:
            cat_summary_df = "No categorical feature "
        _LOGGER.info("Categorical Descriptive Scripts Generated")
        return cat_summary_df

    def get_setanalysis_by_features(self, diff_only=False):
        """
        Get set difference for categorical features.

        Parameters
        ----------
        diff_only: bool, default=False
            If set to true returns only rows where difference is.

        Returns
        -------
        set_analysis_df: pd.DataFrame or str
            Returns DataFrame if cat feature present else returns str
        """
        if len(self.cat_features):
            dict = {}
            for each_segment in self.all_segments:
                dict[str(each_segment)] = self.each_segment_drift[
                    each_segment
                ].get_cat_features_setanalysis_df(diff_only=diff_only)
            set_analysis_df = concat_dfs(dict, names=["segment"])
        else:
            set_analysis_df = "No categorical feature "
        _LOGGER.info("Categorical Set Analysis Generated")
        return set_analysis_df

    def get_segmentanalysis_by_feature(self):
        """To be defined."""
        return 0

    def get_descriptive_stats(self):
        """
        Comparison of summary stats between base and current data.

        Returns
        -------
        self.descriptive_stats: dict
            Dictionary with descriptive statistics for target , numerical features ,
            categorical features and Categorical set difference.
        """
        self.descriptive_stats = {
            "target": self.get_target_desc_stats(),
            "num_features": self.get_num_desc_stats(),
            "cat_features": self.get_cat_desc_stats(),
            "cat_features_set_analysis": self.get_setanalysis_by_features(
                diff_only=False
            ),
        }
        return self.descriptive_stats

    def get_target_drift(self):
        """
        Get Target Drift summary.

        Returns
        -------
        report: dict
            Dictionary with target drift summary.
        """
        dict = {}
        for each_segment in self.all_segments:
            target_drift = self.each_segment_drift[each_segment]._compute_target_drift()
            dict[str(each_segment)] = target_drift["var_level"]

        self.target_drift = concat_dfs(dict, names=["segment"])
        report = {"target_drift_summary": self.target_drift}
        _LOGGER.info("Target Drift Generated")
        return report

    def get_feature_drift(self, combine_features=False):
        """
        Get Feature Drift summary.

        Parameters
        ----------
        combine_features: bool, default=False
            Once can set it to true for excel based output and get dependency
            stability index all in one sheet for all features.

        Returns
        -------
        report: dict
            Dictionary with concept drift summary differently for numerical
            features and categorical features.
        """
        dict_num = {}
        dict_cat = {}
        for each_segment in self.all_segments:
            num_feature_drift, cat_feature_drift = self.each_segment_drift[
                each_segment
            ]._compute_feature_drift()
            if num_feature_drift is not None:
                dict_num[str(each_segment)] = num_feature_drift["var_level"]
            if cat_feature_drift is not None:
                dict_cat[str(each_segment)] = cat_feature_drift["var_level"]

        if len(self.num_features):
            self.num_feature_drift = concat_dfs(dict_num, names=["segment"])
        else:
            self.num_feature_drift = "No numerical feature "

        if len(self.cat_features):
            self.cat_feature_drift = concat_dfs(dict_cat, names=["segment"])
        else:
            self.cat_feature_drift = "No categorical feature "

        report = {
            "feature_drift_summary": {
                "num_features": self.num_feature_drift,
                "cat_features": self.cat_feature_drift,
            }
        }
        _LOGGER.info("Feature Drift Generated")
        return report

    def get_concept_drift(self, combine_features=False):
        """
        Get Concept Drift summary.

        Parameters
        ----------
        combine_features: bool, default=False
            Once can set it to true for excel based output and get dependency
            stability index all in one sheet for all features.

        Returns
        -------
        report: dict
            Dictionary with concept drift summary differently for numerical
            features and categorical features.
        """
        dict_num = {}
        dict_cat = {}
        for each_segment in self.all_segments:
            num_concept_drift, cat_concept_drift = self.each_segment_drift[
                each_segment
            ]._compute_concept_drift()
            if num_concept_drift is not None:
                dict_num[str(each_segment)] = num_concept_drift["var_level"]
            if cat_concept_drift is not None:
                dict_cat[str(each_segment)] = cat_concept_drift["var_level"]

        if len(self.num_features):
            self.num_concept_drift = concat_dfs(dict_num, names=["segment"])
        else:
            self.num_concept_drift = "No numerical feature "

        if len(self.cat_features):
            self.cat_concept_drift = concat_dfs(dict_cat, names=["segment"])
        else:
            self.cat_concept_drift = "No categorical feature "

        report = {
            "concept_drift_summary": {
                "num_features": self.num_concept_drift,
                "cat_features": self.cat_concept_drift,
            }
        }
        _LOGGER.info("Concept Drift Generated")
        return report

    def filter_each_var(self, df, dummy):
        """Filter through each variable."""

        df_subset = df.loc[df["variable"] == dummy]
        df_subset_1 = pd.DataFrame(
            df_subset["segment"].apply(lambda x: x[2:-2].split(",")).tolist(),
            columns=self.segment_by,
        )
        df_subset = pd.concat(
            [
                df_subset.drop(columns="segment").reset_index(drop=True),
                df_subset_1.reset_index(drop=True),
            ],
            axis=1,
        )
        return df_subset

    def get_heatmap_plots(self, summary_options=None):
        """
        Get Heatmap Plots.

        Parameters
        ----------
        summary_options: dict, default=None
            dictionary containing thresholds for each of the drift.

        Returns
        -------
        psi_heatmap: hvplot
        fsi_heatmap: hvplot
        dsi_heatmap: hvplot

        """
        if summary_options is None:
            summary_options = self.summary_options

        fsi_heatmap = defaultdict(lambda: defaultdict(dict))
        dsi_heatmap = defaultdict(lambda: defaultdict(dict))
        psi_heatmap = get_heatmap(
            self.target_drift,
            x_axis="segment",
            y_axis="variable",
            heatmap_value=summary_options["target_drift"]["threshold_on"],
            heatmap_title="Target HeatMap using PSI_value",
        )
        if len(self.num_features):
            for each_variable in self.num_feature_drift["variable"].unique():

                self.num_feature_drift_each_var = self.filter_each_var(
                    self.num_feature_drift, each_variable
                )
                self.num_concept_drift_each_var = self.filter_each_var(
                    self.num_concept_drift, each_variable
                )

                fsi_heatmap["num_features"][each_variable] = get_heatmap(
                    self.num_feature_drift_each_var,
                    x_axis=self.segment_by[0],
                    y_axis=self.segment_by[1],
                    heatmap_value=summary_options["feature_drift_numerical"][
                        "threshold_on"
                    ],
                    heatmap_title="Numerical Feature HeatMap using PSI_value",
                )

                dsi_heatmap["num_features"][each_variable] = get_heatmap(
                    self.num_concept_drift_each_var,
                    x_axis=self.segment_by[0],
                    y_axis=self.segment_by[1],
                    heatmap_value=summary_options["concept_drift_numerical"][
                        "threshold_on"
                    ],
                    heatmap_title="Numerical Feature HeatMap using DSI_value",
                )
        else:
            fsi_heatmap["num_features"] = "No numerical feature "
            dsi_heatmap["num_features"] = "No numerical feature "

        if len(self.cat_features):
            for each_variable in self.cat_feature_drift["variable"].unique():

                self.cat_feature_drift_each_var = self.filter_each_var(
                    self.cat_feature_drift, each_variable
                )
                self.cat_concept_drift_each_var = self.filter_each_var(
                    self.cat_concept_drift, each_variable
                )
            fsi_heatmap["cat_features"] = get_heatmap(
                self.cat_feature_drift_each_var,
                x_axis=self.segment_by[0],
                y_axis=self.segment_by[1],
                heatmap_value=summary_options["feature_drift_categorical"][
                    "threshold_on"
                ],
                heatmap_title="Categorical Feature HeatMap using PSI_value",
            )

            dsi_heatmap["cat_features"] = get_heatmap(
                self.cat_concept_drift_each_var,
                x_axis=self.segment_by[0],
                y_axis=self.segment_by[1],
                heatmap_value=summary_options["concept_drift_categorical"][
                    "threshold_on"
                ],
                heatmap_title="Categorical Feature HeatMap using DSI_value",
            )
        else:
            fsi_heatmap["cat_features"] = "No categorical feature "
            dsi_heatmap["cat_features"] = "No categorical feature "

        _LOGGER.info("Heatmap Plots Generated")

        return psi_heatmap, fsi_heatmap, dsi_heatmap

    def _get_all_drift(self, combine_features=False, summary_options=None):
        """
        Calling all drifts  here and putting in a dictionary.

        Parameters
        ----------
        combine_features: boolean
            Once can set it to true for excel based output and get dependency
            stability index all in one sheet for all features.

        Returns
        -------
        Dictionary containing all three drifts.
        """
        drift_report = dict()
        drift_report["target_drift"] = self.get_target_drift()
        drift_report["feature_drift"] = self.get_feature_drift(
            combine_features=combine_features
        )
        drift_report["concept_drift"] = self.get_concept_drift(
            combine_features=combine_features
        )

        # self.drift_summary = self._compute_drift_summary(summary_options)
        return drift_report

    def get_report(
        self, name="", path="", format=".html", summary_options=None, columns=1
    ):
        """
        Get Report.

        Parameters
        ----------
        name: str, default=""
            Name of the file you want to store report.
        path: str, default=""
            Path of the file where you want to store report.
        format: str, default=".html"
           Format of the report which you want it could be ".xlsx" or ".html."
        summary_options: dict
            dictionary containing thresholds for each of the drift.
        columns: int, default=1
            If set to 1 leads to only one table per row in row otherwise 2 .
        """
        descriptive_stats = self.get_descriptive_stats()
        drift_detailed_report = self._get_all_drift()
        data_summary = self.get_data_summary()
        drift_summary = self._get_drift_summary()
        # psi_heatmap, fsi_heatmap, dsi_heatmap = self.get_heatmap_plots()
        psi_heatmap, fsi_heatmap, dsi_heatmap = {}, {}, {}
        drift_report = self._get_drift_report(psi_heatmap, fsi_heatmap, dsi_heatmap)

        create_report(
            drift_report, name=name, path=path, format=format, columns=columns
        )
        _LOGGER.info("Final Segmented report generated and saved")
