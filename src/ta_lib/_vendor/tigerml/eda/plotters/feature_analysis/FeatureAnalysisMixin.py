import gc
import logging
import numpy as np
import pandas as pd
import tigerml.core.dataframe as td
from tigerml.core.dataframe.helpers import is_dask, tigerify
from tigerml.core.plots import save_plot
from tigerml.core.reports import create_report
from tigerml.core.utils import (
    append_file_to_path,
    compute_if_dask,
    fail_gracefully,
    get_bool_cols,
    measure_time,
    time_now_readable,
)
from tigerml.core.utils.constants import (
    MIN_CUTOFF_FOR_KEY_HEURISTIC,
    SUMMARY_KEY_MAP,
)
from tigerml.core.utils.plots import get_plot_dict_for_dynamicmap

from ...helpers import is_missing

_LOGGER = logging.getLogger(__name__)


def _get_non_null_samples(series, NA_VALUES):
    if "datetime" not in str(series.dtype) and pd.NaT in NA_VALUES:
        NA_VALUES = NA_VALUES.copy()
        NA_VALUES.remove(pd.NaT)
    unique_values = (
        series.unique().head(10) if is_dask(series) else series.unique()[:10]
    )

    samples = []
    for x in unique_values:
        flag = False
        for y in NA_VALUES:
            if x is y:
                flag = True
        if not flag:
            samples.append(x)
    samples = samples[:5]

    if samples:
        return samples
    else:
        return "*Null Variable"


class FeatureAnalysisMixin:
    """Feature Analysis Mixin class."""

    MIN_CUTOFF_FOR_KEY_HEURISTIC = MIN_CUTOFF_FOR_KEY_HEURISTIC

    # @fail_gracefully(_LOGGER)
    # @measure_time(_LOGGER)
    def variable_summary(self, cols=None):
        """Returns the variable summary.

        The function computes a few key metric that describes the dataset along
        with some heuristic comments.

        Parameters
        ----------
        cols : list, default=[]
        list of columns in the dataframe for analysis. By default all are used.

        Returns
        -------
        summary: pandas.DataFrame
        """
        if not cols:
            cols = list(self.data.columns)
        # nrows = len(df)
        vars = [
            SUMMARY_KEY_MAP.variable_names,
            SUMMARY_KEY_MAP.dtype,
            SUMMARY_KEY_MAP.num_unique,
            SUMMARY_KEY_MAP.samples,
        ]
        vs = td.DataFrame(columns=vars, index=cols)
        vs[SUMMARY_KEY_MAP.variable_names] = cols
        for col in cols:
            s = self.data[col]
            vs.loc[col, SUMMARY_KEY_MAP.num_unique] = compute_if_dask(s.nunique())
            vs.at[col, SUMMARY_KEY_MAP.samples] = compute_if_dask(
                _get_non_null_samples(s, self.NA_VALUES)
            )
        # # FIXME: This logic fails if the column is already of type np.datetime64
        col_dtypes = self.data.dtypes.astype(str).loc[cols].tolist()

        vs[SUMMARY_KEY_MAP.dtype] = col_dtypes
        return vs

    # @fail_gracefully(_LOGGER)
    # @measure_time(_LOGGER)
    def numeric_summary(self, data=None, cols=None):
        """Returns the numeric variable summary.

        The function provides a detailed summary stats of all the
        numeric variables in the dataset.

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.

        Returns
        -------
        df: pandas.DataFrame

        """
        if data is None:
            data = self.data
        if not cols:
            cols = [
                col
                for col in data.numeric_columns
                if col not in get_bool_cols(self.data)
            ]

        if cols:
            summary_df = (
                data[cols]
                .describe(percentiles=[0.25, 0.5, 0.75], include=np.number)
                .T.reset_index()
            )
            summary_df = summary_df.drop("count", axis=1)
            new_column_names = [
                SUMMARY_KEY_MAP.variable_names,
                # SUMMARY_KEY_MAP.num_missing,
                SUMMARY_KEY_MAP.mean_value,
                SUMMARY_KEY_MAP.std_deviation,
                SUMMARY_KEY_MAP.min_value,
                SUMMARY_KEY_MAP.percentile_25,
                SUMMARY_KEY_MAP.percentile_50,
                SUMMARY_KEY_MAP.percentile_75,
                SUMMARY_KEY_MAP.max_value,
            ]
            summary_df = summary_df.rename(
                columns=dict(zip(list(summary_df.columns), new_column_names))
            )
            # df[SUMMARY_KEY_MAP.perc_missing] = (
            #         df[SUMMARY_KEY_MAP.num_missing] / float(self.data.shape[0]) * 100
            # )
            vs = self.variable_summary(cols)
            summary_df = vs.merge(summary_df, on=SUMMARY_KEY_MAP.variable_names)
        else:
            summary_df = "No Numerical columns in the data"
        gc.collect()
        return summary_df

    # @fail_gracefully(_LOGGER)
    # @measure_time(_LOGGER)
    def non_numeric_summary(self, data=None, cols=None, segmented=False):
        """Returns the non-numeric variable summary.

        The function provides a detailed summary of all the non-numeric
        variables in the dataset. If there are no non-numeric variables,
        it returns an empty dataframe.

        Parameters
        ----------
        data: pd.dataFrame, default = None
           If you want to explicitly use any other data
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        segmented: bool, default = False
            For sgemented set it to True
        Returns
        -------
        summary_df: pandas.DataFrame
        """
        df = self.data
        if data is not None:
            df = data
        # df = df[[
        #     col for col in df.columns if col not in df.numeric_columns
        #     ]]
        non_num_cols = [
            col
            for col in df.columns
            if col not in df.select_dtypes(include=np.number).columns.tolist()
        ]
        if (
            segmented
        ):  # FIXME: Remove segmented check here (get_bool_cols seems to be malfunctioning)
            bool_cols = [col for col in df.columns if col not in non_num_cols]
            cat_cols = non_num_cols + bool_cols
        else:
            cat_cols = list(set(non_num_cols + get_bool_cols(df)))
        if cols:
            cat_cols = [col for col in cat_cols if col in cols]
        if cat_cols:
            if non_num_cols:
                summary_df = (
                    df[non_num_cols]
                    .astype("category")
                    .describe(exclude=np.number)
                    .T.reset_index()
                )
                computed_cols = summary_df["index"].values
            else:
                summary_df = td.DataFrame()
                computed_cols = []
            # summary_df["count"] = self.data.shape[0] - summary_df["count"]
            left_over_cols = [col for col in cat_cols if col not in computed_cols]
            if left_over_cols:
                for col in left_over_cols:
                    series = df[col]
                    series_stats = series.astype("category", copy=False).describe()
                    series_stats = td.DataFrame(series_stats).T.reset_index()
                    summary_df = td.concat([summary_df, series_stats])
                summary_df.reset_index(drop=True, inplace=True)
            # summary_df[SUMMARY_KEY_MAP.perc_missing] = (
            # 	    summary_df[SUMMARY_KEY_MAP.num_missing] /
            # 	    float(self.data.shape[0]) * 100
            # )
            summary_df = summary_df.rename(
                columns={
                    "index": SUMMARY_KEY_MAP.variable_names,
                    # "count": SUMMARY_KEY_MAP.num_missing,
                    # "unique": SUMMARY_KEY_MAP.unique,
                    "top": SUMMARY_KEY_MAP.mode_value,
                    "freq": SUMMARY_KEY_MAP.mode_freq,
                }
            )
            summary_df[f"{SUMMARY_KEY_MAP.mode_freq} %"] = (
                summary_df[SUMMARY_KEY_MAP.mode_freq].astype(float)
                * 100
                / summary_df["count"].astype(float)
            )
            summary_df = summary_df.drop(["count", "unique"], axis=1)
            vs = self.variable_summary(cols=cat_cols)
            summary_df = vs.merge(summary_df, on=SUMMARY_KEY_MAP.variable_names)
        else:
            summary_df = "No categorical columns"
        del df
        gc.collect()
        return summary_df

    # @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def density_plots(
        self,
        cols=None,
        file_path="density_plots.pdf",
        return_plots=True,
        save_plots=False,
    ):  # noqa
        """Returns a density plot and a table for each of the numeric variables.

        The plots are compiled as pdf into working directory.
        If there are no numeric variables
        then "No Numeric Variables" message will be displayed
        in the python console.

        Output as follows:
            `Density plot` - A plot describing the density distribution
            `Table` - Variable name, Mean, Median, Standard Deviation

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        return_plots : bool, default=True
            If True, interactive hv plots are returned.
        save_plots : bool, default=True
            If True, plots are saved to a report.
        file_path : str, default="percentile_plots.pdf"
            Path for the file to be saved. Applicable when save_plots=True.
        """
        from . import FrequencyPlot

        if cols:
            cols = [x for x in cols if x in self.get_numeric_columns()]
        else:
            cols = self.get_numeric_columns()
        plot = FrequencyPlot(self.data).get_plots(cols)
        if save_plots:
            save_plot(plot, file_path)
        if return_plots:
            return plot
        else:
            del plot
            gc.collect()

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def non_numeric_frequency_plot(
        self,
        cols=None,
        file_path="non_numeric_frequency_plot.pdf",
        return_plots=True,
        save_plots=False,
    ):  # noqa
        """Generate a bar plot of frequency distribution for all non-numeric variables.

        Output as follows
            X.axis - Frequency of occurrence
            Y.axis - Top 10 most frequent Levels of the variable considered

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        return_plots : bool, default=True
            If True, interactive hv plots are returned.
        save_plots : bool, default=False
            If True, plots are saved to a report.
        file_path : str, default="percentile_plots.pdf"
            Path for the file to be saved. Applicable when save_plots=True.
        """
        from . import FrequencyPlot

        if cols:
            cols = [
                x
                for x in cols
                if x in self.get_non_numeric_columns() + get_bool_cols(self.data)
            ]
        else:
            cols = self.get_non_numeric_columns() + get_bool_cols(self.data)
        plot = FrequencyPlot(self.data).get_plots(cols)
        if save_plots:
            plot_dict = get_plot_dict_for_dynamicmap(plot)
            create_report(
                plot_dict,
                name="non_numeric_frequency_plot",
                path=file_path,
            )
        if return_plots:
            return plot
        else:
            del plot
            gc.collect()

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def percentile_plots(
        self,
        cols=None,
        file_path="percentile_plots.pdf",
        return_plots=True,
        save_plots=False,
    ):  # noqa
        """Generate a percentile plot and 3 tables for each of the numeric variables.

        If there are no numeric variables then "No Numeric Variables"
        message will be displayed in the python console. Output is as follows:

        Percentile plot - A plot describing the percentile distribution
        Table 1 - Variable name, Mean, Meadian, Standard Deviation, Minimum and Maximum
        Table 2 - Bottom 5 values(0.2% to 1% range)
        Table 3 - Top 5 values(99.0% to 99.8% range)

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        return_plots : bool, default=True
            If True, interactive hv plots are returned.
        save_plots : bool, default=True
            If True, plots are saved to a report.
        file_path : str, default="percentile_plots.pdf"
            Path for the file to be saved. Applicable when save_plots=True.
        """
        from . import PercentilePlot

        plots = PercentilePlot(self.data).get_plots(cols)
        if save_plots:
            save_plot(plots, file_path)
        if return_plots:
            return plots
        else:
            del plots
            gc.collect()

    def numeric_distributions(self, cols=None):
        """Generate plots for numeric columns.

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        """
        if not cols:
            cols = self.get_numeric_columns()
        else:
            cols = [col for col in cols if col in self.get_numeric_columns()]
        from . import FrequencyPlot, PercentilePlot

        density = FrequencyPlot(self.data)
        percentile = PercentilePlot(self.data)

        def get_distribution(col):
            density_plot = density.get_plots([col])
            percentile_plot = percentile.get_plots([col])
            if type(density_plot) is type(percentile_plot):
                return (density_plot + percentile_plot).cols(1)
            else:
                return percentile_plot

        import holoviews as hv

        if cols:
            plot = hv.DynamicMap(get_distribution, kdims=["col"]).redim.values(col=cols)
        else:
            plot = "No numeric columns in data"
        return plot

    def feature_distributions(self, cols=None):
        """Generate feature distributions.

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        """
        # cols = cols or list(self.data.columns)
        self.feature_distributions_result = {}
        if self.get_numeric_columns():
            num_result = self.numeric_distributions(cols)
            if not isinstance(num_result, str):
                self.feature_distributions_result[
                    "numeric_variables"
                ] = get_plot_dict_for_dynamicmap(num_result)
            else:
                self.feature_distributions_result[
                    "numeric_variables"
                ] = "No numeric variables in data."
        else:
            self.feature_distributions_result[
                "numeric_variables"
            ] = "No numeric variables in data."
        if self.get_non_numeric_columns():
            non_num_result = self.non_numeric_frequency_plot(cols)
            if not isinstance(non_num_result, str):
                self.feature_distributions_result[
                    "non_numeric_variables"
                ] = get_plot_dict_for_dynamicmap(non_num_result)
            else:
                self.feature_distributions_result[
                    "non_numeric_variables"
                ] = "No categorical variables in data."
        else:
            self.feature_distributions_result[
                "non_numeric_variables"
            ] = "No categorical variables in data."
        return self.feature_distributions_result

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def feature_normality(self, cols=[]):
        """Generates Normal Probability plots for numeric columns.

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        """
        cols = cols or list(self.data.columns)
        from . import Normality

        if self.get_numeric_columns():
            req_cols = list(set(cols) & set(self.get_numeric_columns()))
            data = self.data[req_cols]
            plot = Normality().fit(data).get_plot()
            del data
            gc.collect()
            return plot
        else:
            return (
                "Normality plot cannot be generated without numeric columns in the data"
            )

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def target_distribution(self, y=None):
        """Plot distribution  of target variables.

        Parameters
        ----------
        y : str/list, default=None
            str/list of target column(s). By default definition in class are used.

        """
        if y and y != self._current_y:
            assert isinstance(y, str)
            self._set_current_y(y)
        from . import TargetDistribution

        obj = TargetDistribution(self.is_classification).fit(self.data[self._current_y])
        plot = obj.get_plot()
        del obj
        gc.collect()
        return plot

    # @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def feature_analysis(self, cols=[], save_as=None, save_path=""):
        """Univariate analysis for the columns.

        Generate summary_stats, distributions and normality tests for columns.

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.

        y_continuous : bool, default=None
            Set to False, for classificaiton target

        save_as : str, default=None
            You need to pass only extension here ".html" or ".xlsx". If none results will not be saved.
        save_path : str, default=''
            Location where report to be saved. By default report saved in working directory.

        Examples
        --------
        >>> from tigerml.eda import EDAReport
        >>> import pandas as pd
        >>> df = pd.read_csv("titanic.csv")
        >>> an = EDAReport(df, y = 'Survived', y_continuous = False)
        >>> an.feature_analysis()
        """
        # FIX ME: type doesn't have any significance.
        report = {}
        # if type in ["raw", "both"]: # type = <class: 'type'>
        report["summary_stats"] = {}
        # report['summary_stats']['variable_summary'] = self.variable_summary()
        report["summary_stats"]["numeric_variables"] = [self.numeric_summary(cols=cols)]
        report["summary_stats"]["non_numeric_variables"] = [
            self.non_numeric_summary(cols=cols)
        ]
        # if type in ['processed', 'both']:
        #     self.preprocess_data(self._current_y)
        report["distributions"] = self.feature_distributions(cols=cols)
        report["feature_normality"] = self.feature_normality(cols=cols)
        self.feature_analysis_report = report

        # for y in ys:
        #     report['target_distribution of {}'.format(y)] = self.target_distribution(y=y)
        if save_as:
            default_report_name = "feature_analysis_report_at_{}".format(
                time_now_readable()
            )
            save_path = append_file_to_path(save_path, default_report_name + save_as)
            create_report(
                report,
                path=save_path,
                format=save_as,
            )
        return report
