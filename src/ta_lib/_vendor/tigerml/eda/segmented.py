import gc
import logging
import numpy as np
import pandas as pd
import tigerml.core.dataframe as td
from tigerml.core.dataframe.helpers import detigerify
from tigerml.core.plots import hvPlot
from tigerml.core.preprocessing import DataProcessor
from tigerml.core.reports import create_report, format_tables_in_report
from tigerml.core.utils import (
    compute_if_dask,
    get_bool_cols,
    get_non_num_cols,
    get_num_cols,
    measure_time,
    time_now_readable,
)
from tigerml.eda.helpers import is_missing

from .plotters import (
    FeatureAnalysisMixin,
    FeatureInteractionsMixin,
    HealthMixin,
    KeyDriversMixin,
)

_LOGGER = logging.getLogger(__name__)


class SegmentedEDAReport(
    DataProcessor,
    HealthMixin,
    FeatureAnalysisMixin,
    FeatureInteractionsMixin,
    KeyDriversMixin,
):
    """EDA toolkit for classification and regression models on Segmented data.

    To evaluate and generate reports to summarize, data health ,univariate & bivariate analyis, interactions and keydrivers.

    Parameters
    ----------
    data : pd.DataFrame, dataframe to be analyzed

    y : string, default=None
        Name of the target column

    y_continuous : bool, default=None
        Set to False, for classificaiton target

    segment_by : string
        Name of the column using which the data is grouped


    Examples
    --------
    >>> from tigerml.eda import SegmentedEDAReport
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> an = SegmentedEDAReport(df, segment_by= 'Gender', y = 'Survived', y_continuous = False)
    >>> an.get_report(quick = True)
    """

    def __init__(self, data, segment_by, y=None, y_continuous=None):
        if not data.__module__.startswith("tigerml"):
            data = td.DataFrame(data)
            data = data.convert_datetimes()
        super().__init__(data, segment_by=segment_by, y=y, y_continuous=y_continuous)
        self.data = self._segment_cleaner()

    def _segment_cleaner(self, data=None):
        df = data if data is not None else self.data
        for col in self.segment_by:
            df[col][is_missing(df[col], self.NA_VALUES)] = None
        return df.dropna(subset=self.segment_by).reset_index(drop=True)

    def missing_per_segment(self, group_segments_by=None):
        """Returns the number of missing values for each segment.

        This function returns the number of missing values for each variable
        in each segment of the data set. If there are no missing values in any
        of the segment, empty data frame will be returned

        Parameters
        ----------
        group_segments_by : list, default=None
            list of columns by which data frame is to be segmented.
            By default it takes segment_by values.

        Returns
        -------
        df: pandas.DataFrame
        """
        # col_list = [s for s in self.data.columns if s not in group_segments_by]
        group_segments_by = group_segments_by or self.segment_by
        missing_df = self.data.groupby(group_segments_by).apply(
            lambda s: compute_if_dask(is_missing(s, self.NA_VALUES).sum())
        )
        col_sums = missing_df.sum()
        missing_df = missing_df[col_sums[col_sums > 0].index.values]
        row_sums = missing_df.sum(axis=1)
        missing_df = missing_df.loc[row_sums[row_sums > 0].index.values]
        return missing_df.reset_index()

    def rows_per_segment(self, group_segments_by=None):
        """Returns the number of observations for each segment.

        Parameters
        ----------
        group_segments_by : list, default=None
            list of columns by which dataframe is to be segmented.
            By default it takes segment_by values.

        Returns
        -------
        df: pandas.DataFrame
        """
        group_segments_by = group_segments_by or self.segment_by
        if not isinstance(group_segments_by, list):
            group_segments_by = [group_segments_by]
        return (
            self.data[group_segments_by]
            .groupby(group_segments_by)
            .size()
            .to_frame(name="count")
            .reset_index()
        )

    def outliers_per_segment(self, group_segments_by=None):
        """Returns the outlier analysis table which shows outliers per each segment.

        This function returns the outlier analysis table which shows outliers
        for each variable in each segment. If there are no outlier's, empty
        data frame will be returned.

        Parameters
        ----------
        group_segments_by : list, default=None
        list of columns by which data frame is to be segmented.
        By default it takes segment_by values.

        Returns
        -------
        df: pandas.DataFrame
        """

        group_segments_by = group_segments_by or self.segment_by
        func = super().get_outliers_df_for_data

        # Leaving out the segments with No outliers data
        unique = []
        self.data["Concat"] = ""
        for i in group_segments_by:
            self.data["Concat"] = self.data["Concat"] + self.data[i]
        for i in self.data["Concat"].values:
            if i in unique:
                continue
            else:
                unique.append(i)
        f = 0
        for i in unique:
            idf = self.data[self.data["Concat"] == i]
            idf = idf.drop(["Concat"], axis=1)
            idf = idf.groupby(group_segments_by).apply(lambda sdf: func(sdf))
            if type(idf._data.values[0]) == str:
                self.data = self.data[self.data["Concat"] != i]
                f = f + 1
        self.data = self.data.drop(["Concat"], axis=1)
        if len(unique) == f:
            return pd.DataFrame()

        outliers_df = self.data.groupby(group_segments_by).apply(lambda sdf: func(sdf))
        # swapping first and last level, bringing feature to left most
        outliers_df.index = outliers_df.index.swaplevel(0, -1)
        outliers_df.sort_index(inplace=True)
        outliers_df.reset_index(inplace=True)
        outliers_sum = outliers_df.sum(axis=1)
        outliers_df = outliers_df[outliers_sum > 0]
        return outliers_df

    def segments_summary(self):
        # if not isinstance(group_segments_by, list):
        # 	group_segments_by = [group_segments_by]
        # operations_dict = {}
        # for col in metric_columns:
        # 	operations_dict[col] = [agg_func for agg_func in metric_aggregations]
        # return self.data.groupby(group_segments_by).agg(operations_dict)
        """Returns summary text."""
        summary_text = ""
        for seg_col in self.segment_by:
            summary_text += (
                f"There are {self.data[seg_col].nunique()} unique {seg_col}s. "
            )
        summary_text += f'There are {len(self.all_segments)} unique combinations of {", ".join(self.segment_by)}.'
        return summary_text

    def numeric_summary(self, feature=None, group_segments_by=None, quick=True):
        """Gets the numeric summary."""
        # return a dataframe like numeric_summary in EDA of summary stats
        # where each row is each segment for given feature
        summary_dict = {}
        group_segments_by = group_segments_by or self.segment_by
        parent_func = super().numeric_summary

        def func(x, **kwargs):
            """Returns result."""
            result = detigerify(parent_func(data=x, **kwargs))
            if isinstance(result, str):
                return result
            else:
                return result.T[0]

        if feature:
            summary_dict["Complete Data"] = super().numeric_summary(cols=[feature])
            if not quick:
                for groupby_col in group_segments_by:
                    segmented_summary = {}
                    summary_dict[f"{groupby_col}_wise"] = segmented_summary
                    num_cols = [
                        i
                        for i in get_num_cols(self.data)
                        if i not in get_bool_cols(self.data)
                    ]
                    summary_df = self.data.groupby(groupby_col).apply(
                        lambda x: func(x[[feature]], cols=num_cols)
                    )
                    summary_df["count"] = self.data.groupby(groupby_col).count()[
                        feature
                    ]
                    segmented_summary[feature] = [summary_df]
        else:
            summary_dict["Complete Data"] = super().numeric_summary()
            if not quick:
                if type(summary_dict["Complete Data"]) != str:
                    for groupby_col in group_segments_by:
                        segmented_summary = {}
                        summary_dict[f"{groupby_col}_wise"] = segmented_summary
                        cols = [
                            col
                            for col in summary_dict["Complete Data"][
                                "Variable Name"
                            ].unique()
                            if col not in group_segments_by
                        ]
                        num_cols = [
                            i
                            for i in get_num_cols(self.data)
                            if i not in get_bool_cols(self.data)
                        ]
                        # summary_df_seg_mine = self.data.groupby(groupby_col).apply(lambda x: func(x, num_cols=num_cols))
                        for col in cols:
                            summary_df_seg = self.data.groupby(groupby_col).apply(
                                lambda x: func(x[[col]], cols=num_cols)
                            )
                            summary_df_seg["Count"] = (
                                self.data[[col, groupby_col]]
                                .groupby(groupby_col)
                                .count()
                            )
                            summary_df_seg.drop(
                                ["Variable Name", "Datatype", "Samples"],
                                axis=1,
                                inplace=True,
                            )
                            segmented_summary[col] = [summary_df_seg]
        return summary_dict

    def non_numeric_summary(self, feature=None, group_segments_by=None, quick=True):
        """Returns summary dictinary."""
        # return a dataframe like non_numeric_summary in EDA of summary stats
        # where each row is for each segment for given feature
        summary_dict = {}
        group_segments_by = group_segments_by or self.segment_by
        parent_func = super().non_numeric_summary

        def func(x, **kwargs):
            result = detigerify(parent_func(data=x, **kwargs))
            if isinstance(result, str):
                return result
            else:
                return result.T[0]

        if feature:
            summary_dict["Complete Data"] = super().non_numeric_summary(cols=[feature])
            if not quick:
                non_num_cols = [
                    i for i in get_non_num_cols(self.data) if i not in group_segments_by
                ]
                if type(summary_dict["Complete Data"]) != str and len(non_num_cols) > 0:
                    for groupby_col in group_segments_by:
                        segmented_summary = {}
                        summary_dict[f"{groupby_col}_wise"] = segmented_summary
                        summary_df = self.data.groupby(groupby_col).apply(
                            lambda x: func(x[[feature]], segmented=True)
                        )
                        # FIXME: Remove segmented flag here and change non_numeric_summary func
                        summary_df["count"] = self.data.groupby(groupby_col).count()[
                            feature
                        ]
                        segmented_summary[feature] = [summary_df]
        else:
            summary_dict["Complete Data"] = super().non_numeric_summary()
            if not quick:
                non_num_cols = [
                    i for i in get_non_num_cols(self.data) if i not in group_segments_by
                ]
                if type(summary_dict["Complete Data"]) != str and len(non_num_cols) > 0:
                    for groupby_col in group_segments_by:
                        segmented_summary = {}
                        summary_dict[f"{groupby_col}_wise"] = segmented_summary
                        cols = [
                            col
                            for col in summary_dict["Complete Data"][
                                "Variable Name"
                            ].unique()
                            if col not in group_segments_by
                        ]
                        for col in cols:
                            summary_df_seg = self.data.groupby(groupby_col).apply(
                                lambda x: func(x[[col]], segmented=True)
                            )
                            # FIXME: Remove segmented flag here and change non_numeric_summary func
                            summary_df_seg["Count"] = (
                                self.data[[col, groupby_col]]
                                .groupby(groupby_col)
                                .count()
                            )
                            summary_df_seg.drop(
                                ["Variable Name", "Datatype", "Samples"],
                                axis=1,
                                inplace=True,
                            )
                            segmented_summary[col] = [summary_df_seg]
        return summary_dict

    def correlation_table(
        self, x_vars=None, y_vars=None, group_segments_by=None, data=None
    ):
        """Returns correlation table."""
        # between every feature combination, compute correlations after grouping by each segment,
        # and return the summary stats of correlations for each feature as row
        group_segments_by = group_segments_by or self.segment_by
        df = data if data is not None else self.data
        # col = self.data.columns
        func = super().correlation_table_for_data
        num_cols = get_num_cols(df)
        if len(num_cols) > 1:
            corr_table_df = df.groupby(group_segments_by)
            corr_table_df = corr_table_df.apply(
                lambda x: func(data=x, x_vars=num_cols, y_vars=num_cols)
            )
            corr_table_df = corr_table_df.set_index(["Variable 1", "Variable 2"])[
                "Corr Coef"
            ]
            if corr_table_df.empty:
                corr_table_stats = "No correlation found between the numeric columns"
            else:
                corr_table_df = corr_table_df.reset_index()
                corr_table_stats = corr_table_df.groupby(
                    ["Variable 1", "Variable 2"]
                ).describe()
                corr_table_stats.columns = [col[1] for col in corr_table_stats.columns]
                corr_table_stats.reset_index(inplace=True)
                corr_table_stats.drop(["std"], axis=1, inplace=True)
                corr_table_stats["count"] = corr_table_stats["count"].astype(int)
                corr_table_stats.rename(
                    columns={"count": "observed_segments"}, inplace=True
                )
                corr_table_stats["total_segments"] = len(self.all_segments)
        elif len(num_cols) == 1:
            corr_table_stats = "Just one numeric column in the data"
        else:
            corr_table_stats = "No numeric column in the data"
        self.corrs = corr_table_stats
        return corr_table_stats

    def correlation_heatmap(
        self, x_vars=None, y_vars=None, group_segments_by=None, *args, **kwargs
    ):
        """Returns correlation heatmap."""
        if not hasattr(self, "corrs"):
            self.correlation_table()
        if type(self.corrs) == str:
            return self.corrs
        else:
            corr_table_df = self.corrs[["Variable 1", "Variable 2", "mean"]]
            # corr_table_df = corr_table_df.reset_index()
            corr_table_df = td.concat(
                [
                    corr_table_df,
                    corr_table_df.rename(
                        columns={"Variable 1": "Variable 2", "Variable 2": "Variable 1"}
                    ),
                ]
            )
            corr_matrix_df = pd.pivot_table(
                corr_table_df,
                values="mean",
                index=["Variable 1"],
                columns=["Variable 2"],
                aggfunc="sum",
                fill_value=1,
            )
            corr_matrix_df.reset_index(inplace=True)
            corr_matrix_melt = corr_matrix_df.melt(id_vars="Variable 1")
            hv_heatmap = hvPlot(corr_matrix_melt).heatmap(
                x="Variable 1",
                y="Variable 2",
                C="value",
                rot=45,
                height=corr_table_df["Variable 1"].nunique() * 30,
            )
            return hv_heatmap

    def correlation_with_y(self, data=None, group_segments_by=None):
        """
        Returns plot for correlation of numeric columns with target variable.

        Parameters
        ----------
        data : dataframe, default = None
            dataframe used to find correlations
        group_segments_by : list, default=None
            list of columns by which dataframe is to be segmented.
            By default it takes segment_by values.

        Returns
        -------
        Plot with correlation between numeric columns and target variable.

        """
        func = super().correlation_table_for_data
        group_segments_by = group_segments_by or self.segment_by
        df = data if data is not None else self.data
        num_cols = [i for i in get_num_cols(df) if i not in [self._current_y]]
        if len(num_cols) > 0:
            corr_table_df = df.groupby(group_segments_by)
            corr_table_df = corr_table_df.apply(
                lambda x: func(y_vars=[self._current_y], data=x, x_vars=num_cols)
            )
            corr_table_df = corr_table_df.set_index(["Variable 1", "Variable 2"])[
                "Corr Coef"
            ]
            corr_table_df = corr_table_df.reset_index().drop("Variable 2", axis=1)
            corr_table_df = corr_table_df[
                corr_table_df["Variable 1"] != self._current_y
            ]
            corr_table_df.rename(columns={"Variable 1": "Variable Name"}, inplace=True)
            hv_plot = hvPlot(corr_table_df).box(
                y="Corr Coef",
                by="Variable Name",
                invert=True,
                title="Correlation with Target variable",
            )
            hv_plot.opts(tools=["hover"])
            return hv_plot
        else:
            return "No numeric column in the data"

    def _get_segment_contribution(
        self, metric_column, metric_aggregation, group_segments_by, bin_limits=None
    ):
        """Returns segment contributions plot."""
        cum_name = f"{metric_aggregation.__name__} of {metric_column}"
        contributions = (
            self.data[[metric_column] + group_segments_by]
            .groupby(group_segments_by)
            .agg(metric_aggregation)
            .rename(columns={metric_column: cum_name})
        )
        contributions.sort_values(by=cum_name, ascending=False, inplace=True)
        total = contributions.sum()
        perc_contributions = contributions * 100 / total
        cum_perc_name = f"Cumulative % of {cum_name}"
        if len(group_segments_by) > 1 or len(contributions) > 30 or bin_limits:
            contributions[cum_perc_name] = perc_contributions
            if not bin_limits:
                bin_limits = [
                    x for x in range(10, 110, 10)
                ]  # Default bins are 0 to 100 in steps of 10
            # bin_limits = [x for x in range(bin_limits, 100+bin_limits, bin_limits)]
            bins = []
            for ind, limit in enumerate(bin_limits):
                no_of_occurences = int(limit * len(contributions) / 100) - len(bins)
                if no_of_occurences > 0:
                    new_bins = [limit] * no_of_occurences
                    bins += new_bins
            x_name = f'% Number of {" x ".join(group_segments_by)}'
            contributions[x_name] = bins
            contributions = contributions.groupby(x_name).agg(metric_aggregation)
            contributions[cum_perc_name] = contributions[cum_perc_name].cumsum()
            contributions.reset_index(inplace=True)
            bin_values = contributions[x_name].values
            contributions[x_name] = [
                f"{0 if ind==0 else bin_values[ind-1]} - {bin}"
                for ind, bin in enumerate(bin_values)
            ]
        else:
            contributions[cum_perc_name] = perc_contributions.cumsum()
            contributions.reset_index(inplace=True)
            x_name = group_segments_by[0]
        contributions[cum_perc_name].iloc[
            -1
        ] = 100  # Make sure last row sums up to 100,
        # sometimes it comes out as 99.999 etc
        # import holoviews as hv
        bar_plot = hvPlot(contributions).bar(x=x_name, y=cum_name)
        bar_plot = bar_plot.opts(width=600, tools=["hover"], alpha=0.6)
        line_plot = hvPlot(contributions).line(x=x_name, y=cum_perc_name)
        from tigerml.core.plots.bokeh import (
            add_to_secondary,
            finalize_axes_right_keep_0,
        )

        line_plot = line_plot.options(
            hooks=[add_to_secondary, finalize_axes_right_keep_0]
        )
        plot = bar_plot * line_plot
        plot.opts(xrotation=45, width=800, height=500)
        return plot

    def pareto_chart(
        self,
        metric_column=None,
        metric_aggregation=sum,
        group_segments_by=None,
        bin_limits=None,
        return_dict=False,
    ):
        """Returns pareto chart.

        Parameters
        ----------
        metric_column : str, default=None.
             metric column on which pareto chart will be generated.
        metric_aggregation : aggregation function to be applied on
             metric_column, default=sum
        group_segments_by : list, default=None
            list of columns by which dataframe is to be segmented.
            By default it takes segment_by values.
        bin_limits :  list, default=None
            user can give custom bin limits.
        return_dict : bool, default=False
             if True, returns a dictinary pareto_dict.

        Returns
        -------
        dictionary if return_dict is True else holoview plot.
        """
        # Finds the segments/groups of segments that contribute most for a particular metric
        group_segments_by = group_segments_by or self.segment_by
        if not isinstance(group_segments_by, list):
            group_segments_by = [group_segments_by]
        metric_column = metric_column or self._current_y
        if return_dict:
            pareto_dict = {}
            for segment in group_segments_by:
                pareto_dict[segment] = self._get_segment_contribution(
                    metric_column, metric_aggregation, [segment], bin_limits
                )
            if len(group_segments_by) > 1:
                pareto_dict["All"] = self._get_segment_contribution(
                    metric_column, metric_aggregation, group_segments_by, bin_limits
                )
            return pareto_dict
        else:
            if len(group_segments_by) > 1:

                def get_pareto_chart(group_by):
                    if group_by == "All":
                        group_by = group_segments_by
                    else:
                        group_by = [group_by]
                    return self._get_segment_contribution(
                        metric_column, metric_aggregation, group_by, bin_limits
                    )

                import holoviews as hv

                pareto_chart = hv.DynamicMap(
                    get_pareto_chart, kdims=["Group"]
                ).redim.values(Group=group_segments_by + ["All"])
            else:
                pareto_chart = self._get_segment_contribution(
                    metric_column, metric_aggregation, group_segments_by, bin_limits
                )
            return pareto_chart

    def get_segments_analysis(self):
        # segment_by = segment_by or self.segment_by
        # if not metric_column:
        #     metric_column = self._current_y
        """Returns a dictionary consisting of segments_summary, number of rows, missing values, outliers per segment."""
        segmented_analysis = {
            "summary": self.segments_summary(),
            "number_of_rows": self.rows_per_segment(),
            "no_of_missing_values": self.missing_per_segment(),
            "no_of_outliers": self.outliers_per_segment(),
        }
        return segmented_analysis

    def create_report(self, y=None, quick=True, corr_threshold=None):
        """
        Returns a complete analysis report.

        Parameters
        ----------
        y : str, default=None
            Target variable
        quick : boolean, default=True
            If true,calculate SHAP values and create bivariate plots
        corr_threshold : float, default=None
            To specify correlation threshold

        """
        if y:
            self._set_y_cols(y)
        complete_analysis = {}
        complete_analysis["data_preview"] = {
            "head": [self.data.head(5)],
            "tail": [self.data.tail(5)],
        }
        complete_analysis["health_analysis"] = self.health_analysis()
        complete_analysis["segments_analysis"] = self.get_segments_analysis()
        complete_analysis["data_preview"]["pre_processing"] = self._prepare_data(
            corr_threshold
        )
        complete_analysis["feature_analysis"] = {
            "numeric_summary": self.numeric_summary(quick=quick),
            "non_numeric_summary": self.non_numeric_summary(quick=quick),
        }
        complete_analysis["feature_interactions"] = {
            "correlation_table": [self.correlation_table()],
            "correlation_heatmap": [self.correlation_heatmap()],
        }
        if self.y_cols:
            key_drivers = {}
            for col in self.y_cols:
                self._set_current_y(col)
                key_drivers[col] = {
                    "feature_scores - correlation": [self.correlation_with_y()],
                    "pareto_analysis": self.pareto_chart(return_dict=True),
                }
            complete_analysis["key_drivers"] = key_drivers
        else:
            _LOGGER.info(
                "Could not generate key drivers report as dependent variable is not defined"
            )
        self.report = complete_analysis
        return self.report

    @measure_time(_LOGGER)
    def save_report(
        self, format=".html", name="", save_path="", tiger_template=False, **kwargs
    ):
        """
        Saves the report.

        Parameters
        ----------
        format : str, default='.html'
            format of report to be generated. possible values '.xlsx', '.html'
        name : str, default=None
            Name of the report. By default name is auto generated from system timestamp.
        save_path : str, default=''
            location with filename where report to be saved. By default is auto generated from system timestamp and saved in working directory.
        tiger_template : bool, default = False

        """
        _LOGGER.info("Started saving the report")
        if not name:
            name = "data_exploration_report_at_{}".format(time_now_readable())
        compute_if_dask(self.report)
        create_report(
            self.report,
            name=name,
            path=save_path,
            format=format,
            split_sheets=True,
            tiger_template=tiger_template,
            **kwargs,
        )
        del self.report
        gc.collect()
        _LOGGER.info("Saved the report successfully.")
        return

    @measure_time(_LOGGER)
    def get_report(
        self,
        format=".html",
        name="",
        y=None,
        corr_threshold=None,
        quick=True,
        save_path="",
        tiger_template=False,
        light_format=True,
        **kwargs,
    ):
        """Create consolidated report on data preview,feature analysis,feature interaction and health analysis.

        The consolidated report also includes key driver report if y(target dataframe) is passed while
        calling create_report.

        Parameters
        ----------
        y : str, default = None
        format : str, default='.html'
            format of report to be generated. possible values '.xlsx', '.html'
        name : str, default=None
            Name of the report. By default name is auto generated from system timestamp.
        save_path : str, default=''
            location with filename where report to be saved. By default is auto generated from system timestamp and saved in working directory.
        quick : boolean, default=True
            If true,calculate SHAP values and create bivariate plots
        corr_threshold : float, default=None
            To specify correlation threshold
        excel_params : dict
            Dictionary containing the following keys if the format is ".xlsx".
            If a key is not provided, it will take the default values.

                - have_plot : boolean; default False.
                    If True, keep the plots in image format in excel report.
                - n_rows : int; default 100.
                    Number of sample rows to keep for plot types containing all the records in data (for example, density plot, scatter plot etc.)
        """
        self.create_report(y=y, quick=quick, corr_threshold=corr_threshold)
        if light_format:
            self.report = format_tables_in_report(self.report)

        if format == ".xlsx":
            keys_to_combine = [
                ("data_preview", "pre_processing", "encoded_mappings"),  # noqa
                ("feature_analysis", "distributions", "numeric_variables"),  # noqa
                ("feature_analysis", "distributions", "non_numeric_variables"),  # noqa
                (
                    "feature_interactions",
                    "bivariate_plots (Top 50 Correlations)",
                ),  # noqa
                ("key_drivers", self.y_cols[0], "bivariate_plots"),
            ]  # noqa

            from tigerml.core.utils import convert_to_tuples

            convert_to_tuples(keys_to_combine, self.report)

        return self.save_report(
            format=format,
            name=name,
            save_path=save_path,
            tiger_template=tiger_template,
            **kwargs,
        )
