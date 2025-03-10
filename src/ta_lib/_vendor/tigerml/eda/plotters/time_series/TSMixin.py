import collections
import gc
import holoviews as hv
import logging
import numpy as np
import pandas as pd
from math import ceil
from tigerml.core.plots import hvPlot
from tigerml.core.utils import fail_gracefully, get_bool_cols, measure_time

_LOGGER = logging.getLogger(__name__)


class TSMixin:
    """Ts mixin class."""

    @property
    def analysis_columns(self):
        """Returns List of columns for analysis from data."""
        if self.ts_identifiers:
            return [
                col
                for col in self.data.numeric_columns
                if col
                not in self.ts_identifiers + [self.ts_column] + get_bool_cols(self.data)
            ]
        else:
            return [
                col
                for col in self.data.numeric_columns
                if col not in [self.ts_column] + get_bool_cols(self.data)
            ]

    def _ts_sorter(self, data=None):
        """Returns data sorted based on time col."""
        df = data if data is not None else self.data
        df = df.dropna(subset=[self.ts_column]).reset_index(drop=True)
        if self.ts_identifiers:
            return (
                df.groupby(self.ts_identifiers)
                .apply(lambda x: x.sort_values([self.ts_column], ascending=True))
                .reset_index(drop=True)
            )
        else:
            return df.sort_values(by=[self.ts_column]).reset_index(drop=True)

    # @measure_time(_LOGGER)
    def _compute_periodicity(self, data=None):
        """Returns time period."""
        df = data if data is not None else self.data

        def periodicity(df, ts_column):
            """Returns data with minimum of consecutive_diff."""
            df_consecutive_diff = df[ts_column].diff()
            return df_consecutive_diff[df_consecutive_diff > np.timedelta64(0)].min()

        if self.ts_identifiers and data is None:
            self._periodicity = df.groupby(self.ts_identifiers).apply(
                lambda sdf: periodicity(sdf, self.ts_column)
            )
            return self._periodicity
        else:
            period = periodicity(df, self.ts_column)
            if data is None:
                self._periodicity = period
            return period

    @fail_gracefully(_LOGGER)
    def last_period_in_series(self, data=None):
        """Returns last period in a time series column.

        Parameters
        ----------
        data : data frame, default = None
            Time series data
        """
        df = data if data is not None else self.data
        if self.ts_identifiers and data is None:
            result_series = df.groupby(self.ts_identifiers).apply(
                lambda sdf: self.last_period_in_series(sdf)
            )
            result_series.name = "last_period"
            return result_series
        else:
            return max(df[self.ts_column])

    def plot_last_period_occurrence(self):
        """Returns a bar plot showing Date vs number of times last period occurs."""
        if not self.ts_identifiers:
            pass
        else:
            plot = hvPlot(
                self.last_period_in_series().reset_index().last_period.value_counts()
            ).bar(xlabel="Date", ylabel="# times last period occurs", rot=45)
            return plot

    @fail_gracefully(_LOGGER)
    def first_period_in_series(self, data=None):
        """Returns the first period in the time series data.

        For each ts_identifiers, if given else gives first
        period in the time series data.

        Parameters
        ----------
        data : data frame, default=None
            if not given, takes data from the TSAnalyser class

        Returns
        -------
        df: pandas.DataFrame if ts_identifiers were given, else date time value
        """
        df = data if data is not None else self.data
        if self.ts_identifiers and data is None:
            result_series = df.groupby(self.ts_identifiers).apply(
                lambda sdf: self.first_period_in_series(sdf)
            )
            result_series.name = "first_period"
            return result_series
        else:
            return min(df[self.ts_column])

    def plot_first_period_occurrence(self):
        """Returns a bar plot showing Date vs number of times first period occurs."""
        if not self.ts_identifiers:
            pass
        else:
            plot = hvPlot(
                self.first_period_in_series().reset_index().first_period.value_counts()
            ).bar(xlabel="Date", ylabel="# times first period occurs", rot=45)
            return plot

    @fail_gracefully(_LOGGER)
    def no_of_periods(self, data=None):
        """Returns number of periods in the data.

        Parameters
        ----------
        data : data frame, default = None
            Time series data
        """
        df = data if data is not None else self.data
        if self.ts_identifiers and data is None:
            return df.groupby(self.ts_identifiers).apply(
                lambda sdf: self.no_of_periods(sdf)
            )
        else:
            first_period = self.first_period_in_series(df)
            last_period = self.last_period_in_series(df)
            periodicity = self._compute_periodicity(df)
            return int(((last_period - first_period) / periodicity) + 1)

    @fail_gracefully(_LOGGER)
    def missing_periods(self, data=None, return_values=False):
        """Returns a dict of missing periods in the data.

        Parameters
        ----------
        data : data frame, default = None
            Time series data
        return_values : bool, default = False
            return missing values, if given True
        """

        def calc_missing(row, ts_col, periodicity):
            start = row[ts_col] - row["consecutive_diff"]
            step = periodicity
            end = ceil(row["consecutive_diff"] / periodicity)
            return [start + (x * step) for x in range(1, end)]

        def get_missing_periods(df, ts_column):
            df["consecutive_diff"] = df[ts_column].diff()
            periodicity = self._compute_periodicity(df)
            missing_periods_df = df[df["consecutive_diff"] > periodicity]
            if len(missing_periods_df) > 0:
                missing_periods_df["missing_periods"] = missing_periods_df.apply(
                    lambda row: calc_missing(row, ts_column, periodicity), axis=1
                )
                from tigerml.core.utils import flatten_list

                missing_periods = flatten_list(
                    missing_periods_df["missing_periods"].values.tolist()
                )
            else:
                missing_periods = []
            missing_df_ = pd.DataFrame(
                [[len(missing_periods), missing_periods]],
                columns=["no_of_missing_periods", "missing_values"],
            )
            return missing_df_

        df = data if data is not None else self.data
        if self.ts_identifiers and data is None:
            missing_df = (
                self.data.groupby(self.ts_identifiers)
                .apply(lambda sdf: get_missing_periods(sdf, self.ts_column))
                .unstack()
                .swaplevel(0, 1, axis=1)[0]
            )
            if not return_values:
                missing_df = missing_df[["no_of_missing_periods"]]
            missing_df = missing_df[missing_df["no_of_missing_periods"] > 0]
            if missing_df.empty:
                missing_df = "No missing periods in series"
            return missing_df
        else:
            missing_df = get_missing_periods(df, self.ts_column)
            if missing_df["no_of_missing_periods"][0] == 0:
                return "No missing periods in series"
            else:
                return_dict = {
                    "no_of_missing_periods": missing_df["no_of_missing_periods"][0],
                    "missing_periods": self.show_missing_periods(
                        data=df, missing_df=missing_df
                    ),
                }
                if return_values:
                    return_dict["missing_values"] = missing_df["missing_values"][0]
                return return_dict

    def show_missing_periods(self, data=None, missing_df=None):
        """Returns the hvPlot for the missing periods in time series data.

        Parameters
        ----------
        data : data frame, default = None
            Time series data
        missing_df : data frame, default = None
            data frame containing missing values
        """
        df = data if data is not None else self.data
        time_values = df[self.ts_column]
        if missing_df is None:
            if data is not None:
                missing_plot = self.missing_periods(data=df)
                if isinstance(missing_plot, str):
                    return missing_plot
                return missing_plot["missing_periods"]
            else:
                if self.ts_identifiers:
                    # raise ValueError('Segment data has to be passed to get missing periods plot')
                    return "Segment data has to be passed to get missing periods plot"
                else:
                    missing_plot = self.missing_periods()
                    if isinstance(missing_plot, str):
                        return missing_plot
                    return missing_plot["missing_periods"]
        else:
            missing_times = missing_df.missing_values[0]
            # missing_df.drop('missing_values', axis=1, inplace=True)
            df = pd.Series([1] * len(time_values), index=time_values.sort_values())
            df = pd.concat(
                [df, pd.Series([0] * len(missing_times), index=sorted(missing_times))]
            )
            df.sort_index(inplace=True)
            # df.name = 'No of Occurences'
            # df2.name = 'No of Occurences'
        return hvPlot(df.reset_index()).line(ylabel="No of Occurences", xlabel="Time")

    @fail_gracefully(_LOGGER)
    def get_time_repetitions(self, data=None, return_values=False):
        """Returns the data frame showing repeated periods for each ts_identifiers.

        If given else gives dictionary with  no of repetitions in the time series data.

        Parameters
        ----------
        data : data frame, default=None
            if not given, takes data from the TSAnalyser class

        return_values : bool, default=False
            if True, the the return data frame has dictinary of repeated periods

        Returns
        -------
        df: pandas.DataFrame if ts_identifiers were given, else returns dictionary
        """
        # For each series return the instances where time col has any repetitions. IF not, return False.
        # Time repetition shouldn't happen in TS modeling.
        def get_duplicate_values(time_series):
            # items_without_na = [x for x in df_ if x not in NA_VALUES]
            time_series.dropna(inplace=True)
            duplicate_items = {
                item: count
                for item, count in collections.Counter(time_series).items()
                if count > 1
            }
            duplicate_df = pd.DataFrame(
                [[len(duplicate_items), duplicate_items]],
                columns=["no_of_repetitions", "repeating_periods"],
            )
            return duplicate_df

        df = data if data is not None else self.data
        if self.ts_identifiers and data is None:
            result_df = (
                df.groupby(self.ts_identifiers)
                .apply(lambda seg_df: get_duplicate_values(seg_df[self.ts_column]))
                .unstack()
                .swaplevel(0, 1, axis=1)[0]
            )
            if not return_values:
                result_df = result_df[["no_of_repetitions"]]
            result_df = result_df[result_df["no_of_repetitions"] > 0]
            no_of_repeats = result_df.no_of_repetitions
            if result_df.empty:
                no_of_repeats = "No repetitions in series"
            return no_of_repeats
        else:
            duplicates_df = get_duplicate_values(df[self.ts_column])
            if duplicates_df["no_of_repetitions"][0] == 0:
                no_of_repeats = "No repetitions in series"
                return no_of_repeats
            else:
                return_dict = {
                    "no_of_repetitions": duplicates_df["no_of_repetitions"][0],
                    "repeating_periods": self.show_time_repetitions(data=df),
                }
                if return_values:
                    return_dict["repeating_values"] = duplicates_df[
                        "repeating_periods"
                    ][0]
                return return_dict

    def show_time_repetitions(self, data=None):
        """Returns hvPlot for time repetitions.

        Parameters
        ----------
        data : data frame, default = None
            Time series data
        """
        df = data if data is not None else self.data
        time_values = df[self.ts_column]
        if data is None and self.ts_identifiers:
            return "Segment data has to be passed to get time repetitions plot"
        else:
            df = pd.Series([1] * len(time_values), index=time_values.sort_values())
            df = df.reset_index().groupby(self.ts_column).sum()
            df.name = "No of occurences"
        return hvPlot(df).line()

    def get_conf_lines(self):
        """Marks the position along the y-axis at the upper confidence interval and lower confidence interval."""
        if self.ts_identifiers:
            mean_len = round(
                self.data.groupby(self.ts_identifiers)[self.ts_column].count().mean()
            )
            conf_interval_limit = 1.96 / np.sqrt(mean_len)
        else:
            conf_interval_limit = 1.96 / np.sqrt(len(self.data))
        conf_interval_upper = conf_interval_limit
        conf_interval_lower = -conf_interval_limit
        import holoviews as hv

        hlines = hv.HLine(conf_interval_upper).opts(
            line_dash="dashed", color="red"
        ) * hv.HLine(conf_interval_lower).opts(line_dash="dashed", color="red")
        return hlines

    @fail_gracefully(_LOGGER)
    def get_acf_plot(self, lags=None):
        """Returns acf plot.

        Parameters
        ----------
        lags : list, default=None
            .if lags was given it calculates correlation util lag
            for maximum value in list else it will calculate until
            50 lags.

        Returns
        -------
        holoview plot
        """
        from statsmodels.tsa.stattools import acf

        if self.ts_identifiers:
            x = self.data.groupby(self.ts_identifiers).size()
            n = np.min(x.values)
        else:
            n = self.data.shape[0] - 1

        if lags is None:
            nlags = min(50, n)
        else:
            nlags = min(n, max(lags))

        def get_acf_df(df, y):
            return pd.DataFrame.from_dict(
                {"lags": range(0, nlags + 1), "correlation": acf(df[y], nlags=nlags)}
            )

        if self.ts_identifiers:
            lag_corrs = self.data.groupby(self.ts_identifiers).apply(
                lambda sdf: get_acf_df(sdf, self._current_y)
            )
            lag_acf = (
                lag_corrs.reset_index()
                .groupby("lags")["correlation"]
                .apply(
                    lambda corrs: pd.Series([corrs.min(), corrs.mean(), corrs.max()])
                    .to_frame()
                    .T
                )
            )
            lag_acf.rename(
                columns=dict(
                    zip(
                        list(lag_acf.columns),
                        ["min_correlation", "mean_correlation", "max_correlation"],
                    )
                ),
                inplace=True,
            )
            mean_line = hvPlot(lag_acf).line(
                y="mean_correlation", xlabel="Number of lags", ylabel="Correlation"
            )
            min_max_area = hvPlot(lag_acf).area(
                y="min_correlation", y2="max_correlation", alpha=0.7
            )
            plot = mean_line * min_max_area
        else:
            lag_acf = get_acf_df(self.data, self._current_y).set_index("lags")
            plot = hvPlot(lag_acf).line(xlabel="Number of lags", ylabel="Correlation")
        hlines = self.get_conf_lines()
        return plot * hlines

    @fail_gracefully(_LOGGER)
    def get_pacf_plot(self, lags=None):
        """Returns pacf plot.

        Parameters
        ----------
        lags : list, default=None
            .if lags was given it calculates correlation util lag
            for maximum value in list else it will calculate until
            50 lags.

        Returns
        -------
        holoview plot
        """
        from statsmodels.tsa.stattools import pacf

        if self.ts_identifiers:
            x = self.data.groupby(self.ts_identifiers).size()
            n = np.min(x.values)
        else:
            n = self.data.shape[0] // 2 - 1

        if lags is None:
            nlags = min(50, n)
        else:
            nlags = min(n, max(lags))

        def get_pacf_df(df, y):
            return pd.DataFrame.from_dict(
                {
                    "lags": range(0, nlags + 1),
                    "correlation": pacf(df[y], nlags=nlags, method="ols"),
                }
            )

        if self.ts_identifiers:
            lag_corrs = self.data.groupby(self.ts_identifiers).apply(
                lambda sdf: get_pacf_df(sdf, self._current_y)
            )
            lag_pacf = (
                lag_corrs.reset_index()
                .groupby("lags")["correlation"]
                .apply(
                    lambda corrs: pd.Series([corrs.min(), corrs.mean(), corrs.max()])
                    .to_frame()
                    .T
                )
            )
            lag_pacf.rename(
                columns=dict(
                    zip(
                        list(lag_pacf.columns),
                        ["min_correlation", "mean_correlation", "max_correlation"],
                    )
                ),
                inplace=True,
            )
            mean_line = hvPlot(lag_pacf).line(
                y="mean_correlation", xlabel="Number of lags", ylabel="Correlation"
            )
            min_max_area = hvPlot(lag_pacf).area(
                y="min_correlation", y2="max_correlation", alpha=0.7
            )
            plot = mean_line * min_max_area
        else:
            lag_pacf = get_pacf_df(self.data, self._current_y).set_index("lags")
            plot = hvPlot(lag_pacf).line(xlabel="Number of lags", ylabel="Correlation")
        hlines = self.get_conf_lines()
        return plot * hlines

    def lag_analysis(self, lags=None, quick=True):
        """Returns the dictionary having acf pacf and lag analysis plots for each indipendent variable.

        Parameters
        ----------
        lags : list, default=None
            .if lags was given it calculates correlation util lag
            for maximum value in list else it will calculate until
            50 lags.

        quick : bool, default=True
            if False, gives lag analysis plots for each indipendent variable

        Returns
        -------
        dictionary of holoview plots
        """
        lag_analysis_dict = {
            "ACF_plot": self.get_acf_plot(lags),
            "PACF_plot": self.get_pacf_plot(lags),
        }
        if not quick:
            idv_lag_analysis = {}
            lag_analysis_dict["lags_with_idvs"] = idv_lag_analysis
            for col in self.analysis_columns:
                if col != self._current_y:
                    idv_lag_analysis[col] = self.lag_with_idv(col)
        return lag_analysis_dict

    def lag_with_idv(self, idv, data=None, lags=None):
        """Returns plot on number of lags vs correlation coefficient.

        Parameters
        ----------
        idv : string.
            Column in the data
        data : data frame, default = None
            Time series data
        lags : list, default=None
            If lags was given it calculates correlation util lag
            for maximum value in list else it will calculate until
            50 lags.
        """
        if lags is None:
            nlags = 50
        else:
            nlags = max(lags)

        def create_lagged_df(df, idv, lags):
            for lag in lags:
                df[lag] = df[idv].shift(lag)
            df.rename(columns={idv: 0}, inplace=True)
            return df

        df = data if data is not None else self.data
        df = df[[idv, self._current_y] + self.ts_identifiers]
        if self.ts_identifiers and data is None:
            df = df.groupby(self.ts_identifiers).apply(
                lambda sdf: create_lagged_df(sdf, idv, range(1, nlags + 1))
            )
            corr_table = self.correlation_table(
                data=df,
                x_vars=[x for x in df.columns if x != self._current_y],
                y_vars=self._current_y,
            )
            corr_table = corr_table[
                corr_table["Variable 2"] == self._current_y
            ].reset_index(drop=True)
            corr_table.sort_values(by=["Variable 1"], inplace=True)
            cols = ["mean", "min", "max"]
            lag_df = corr_table[cols].rename(
                columns={col: f"{col}_correlation" for col in cols}
            )
            mean_line = hvPlot(lag_df).line(
                y="mean_correlation", xlabel="Number of lags", ylabel="Correlation"
            )
            min_max_area = hvPlot(lag_df).area(
                y="min_correlation", y2="max_correlation", alpha=0.7
            )
            plot = mean_line * min_max_area
        else:
            df = create_lagged_df(df, idv, range(1, nlags + 1))
            corr_table = self.correlation_table(
                data=df,
                x_vars=[x for x in df.columns if x != self._current_y],
                y_vars=self._current_y,
            )
            corr_table = corr_table[
                corr_table["Variable 2"] == self._current_y
            ].reset_index(drop=True)
            corr_table.sort_values(by=["Variable 1"], inplace=True)
            plot = hvPlot(corr_table).line(
                y="Corr Coef", x="Variable 1", xlabel="Number of lags"
            )
        hlines = self.get_conf_lines()
        return plot * hlines

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def get_outliers(self, data=None, cols=[], get_indeces=False):
        """Returns the dictionary with no of outliers and outlier plot for the y_cols.

        Parameters
        ----------
        data: DataFrame
        cols: list (optional)
        default value - y_cols.
        get_indeces: bool, default: False
        if True return's the list of  outliers points in the dictionary

        Returns
        -------
        dictionary
        """
        try:
            from luminol import anomaly_detector
        except ModuleNotFoundError:
            raise Exception(
                "luminol package not found. Please install with - pip install luminol"
            )
        cols = cols or self.y_cols
        selected_col = []
        for col_ in cols:
            if col_ not in self.data.numeric_columns:
                _LOGGER.info(
                    f"{col_} is not a numeric column hence outlier detection is skipped."
                )
            else:
                selected_col += [col_]
        cols = selected_col
        if len(cols) == 0:
            return "No numeric column passed for outlier detection."

        if not hasattr(self, "_periodicity"):
            self._compute_periodicity()

        def detect_outliers(series, periodicity):
            detector = anomaly_detector.AnomalyDetector(series.to_dict())
            outliers = detector.get_anomalies()
            anomaly_dict = {}
            scores = []
            for outlier in outliers:
                time_period = outlier.get_time_window()
                if time_period[1] - time_period[0] > 0:
                    current_time = time_period[0]
                    while current_time <= time_period[1]:
                        try:
                            value = series.loc[current_time]
                            if "Series" in str(type(value)):
                                for val in value:
                                    anomaly_dict.update({current_time: val})
                                    # scores.append(outlier.anomaly_score)
                            else:
                                anomaly_dict.update({current_time: value})
                            scores.append(outlier.anomaly_score)
                        except KeyError:
                            pass
                        current_time += periodicity.total_seconds()
                else:
                    value = series.loc[time_period[0]]
                    anomaly_dict.update({time_period[0]: value})
                    scores.append(outlier.anomaly_score)
            outliers_df = pd.DataFrame()
            outliers_df[series.name] = list(anomaly_dict.values())
            outliers_df["index"] = list(anomaly_dict.keys())
            outliers_df["index"] = pd.to_datetime(
                outliers_df["index"].astype(np.int64) * (10**9)
            )
            outliers_df.set_index("index", inplace=True)
            outliers_df["score"] = scores
            return outliers_df

        def outlier_indeces(df, analysis_columns, periodicity):
            # print('Running for {}'.format(df.name))
            if periodicity.delta % (10**9) > 0:
                if self.ts_identifiers:
                    if hasattr(df, "name"):
                        _LOGGER.info(
                            f"Outlier detection not done for segment {df.name} due to non-zero micro/nanoseconds "
                            f"in the series."
                        )
                        return None
                    else:
                        return (
                            "Outlier detection not done for the data due to non-zero micro/nanoseconds in "
                            "the series."
                        )
                else:
                    return "Outlier detection not done for the data due to non-zero micro/nanoseconds in the series."
            length = len(df)
            outliers_len = []
            outliers_perc = []
            outliers_score = []
            for col in analysis_columns:
                series = pd.Series(
                    df[col].values,
                    index=df[self.ts_column].astype(np.int64) / (10**9),
                )
                series.name = col
                if len(series) > len(set(series.index.values)):
                    _LOGGER.info(
                        "Time values repeat. Taking the first value for each time."
                    )
                    series = series.groupby(series.index).first()
                outliers = detect_outliers(series, periodicity)
                outliers_len += [len(outliers)]
                outliers_perc += [round(len(outliers) * 100 / length, 2)]
                outliers_score += [outliers["score"].to_dict()]
            outliers_dict = {
                "No of Outliers": pd.Series(outliers_len, index=cols),
                "Percentage of Outliers": pd.Series(outliers_perc, index=cols),
                "Outliers Score": pd.Series(outliers_score, index=cols),
            }
            outliers_data_ = pd.concat(outliers_dict, axis=1)
            return outliers_data_

        df = data if data is not None else self.data
        if self.ts_identifiers and data is None:
            outlier_df = df.groupby(self.ts_identifiers).apply(
                lambda sdf: outlier_indeces(sdf, cols, self._periodicity.loc[sdf.name])
            )
            if outlier_df.empty:
                outlier_df = (
                    "Outlier detection not performed for any of the segment "
                    "due to non-zero micro/nanoseconds in all the series."
                )
            else:
                outlier_df = outlier_df.unstack()
                nil_cols = (
                    outlier_df["No of Outliers"]
                    .columns[
                        outlier_df["No of Outliers"].apply(lambda col: col.sum() <= 0)
                    ]
                    .tolist()
                )
                if not get_indeces:
                    outlier_df = outlier_df[
                        ["No of Outliers", "Percentage of Outliers"]
                    ]
                if nil_cols:
                    outlier_df.drop(nil_cols, axis=1, level=1, inplace=True)
                if outlier_df.empty:
                    outlier_df = "No outliers in the data"
            return outlier_df
        else:
            if self.ts_identifiers:
                outlier_data = outlier_indeces(
                    df, cols, self._compute_periodicity(data=df)
                )
            else:
                outlier_data = outlier_indeces(df, cols, self._periodicity)
            if type(outlier_data) == str:
                return outlier_data
            outlier_df = outlier_data["Outliers Score"]
            outlier_counts = outlier_data["No of Outliers"]
            if not outlier_counts[outlier_counts > 0].empty:
                outlier_cols = outlier_counts[outlier_counts > 0].index.values.tolist()
                return_dict = {
                    "no_of_outliers": outlier_data[["No of Outliers"]],
                    "outlier_points": self.show_outliers(
                        data=df,
                        cols=outlier_cols,
                        outliers_data=outlier_df[outlier_cols],
                    ),
                }
                if get_indeces:
                    return_dict["Outliers Score"] = outlier_data[["Outliers Score"]]
                return return_dict
            else:
                return "No outliers in the data"

    def outliers_plot(self):
        """Returns a bar plot showing percentage of outliers in segments."""
        if not self.ts_identifiers:
            pass
        else:
            if isinstance(self.get_outliers(), str):
                return self.get_outliers()
            else:
                break_value = [
                    0,
                    5,
                    10,
                    15,
                    20,
                    25,
                    30,
                    35,
                    40,
                    45,
                    50,
                    55,
                    60,
                    65,
                    70,
                    75,
                    80,
                    85,
                    90,
                    95,
                    100,
                ]
                lab_value = [
                    "0-5%",
                    "5-10%",
                    "10-15%",
                    "15-20%",
                    "20-45%",
                    "25-30%",
                    "30-35%",
                    "35-40%",
                    "40-45%",
                    "45-50%",
                    "50-55%",
                    "55-60%",
                    "60-65%",
                    "65-70%",
                    "70-75%",
                    "75-80%",
                    "80-85%",
                    "85-90%",
                    "90-95%",
                    "95-100%",
                ]
                cuts = pd.cut(
                    self.get_outliers()["Percentage of Outliers"][self.y_cols[0]]._data,
                    bins=break_value,
                    labels=lab_value,
                    right=True,
                )
                cuts = cuts.value_counts().reindex(lab_value)
                remaining_cols = self.get_outliers().shape[0] - cuts.sum()
                cuts = pd.concat(
                    [pd.Series([remaining_cols], index=["No Outiers"]), cuts]
                )
                plot = hvPlot(cuts).bar(
                    rot=45,
                    title="Percentage of Outliers in Segments",
                    xlabel="percentage of outliers",
                    ylabel="# of segments",
                )
                return plot

    def show_outliers(self, data=None, cols=[], outliers_data=None):
        """Returns the hvPlot plotting values with outlier's plotted in red color.

        Parameters
        ----------
        data: DataFrame
        cols: list (optional)
        default value - y_cols.
        outliers_data: pandas.DataFrame, default: None
        if given, will calculate the plot directly from outliers_data.

        Returns
        -------
         plt: `hvplot`

        """
        outlier_plots = {}
        if outliers_data is None:
            if data is not None:
                outlier_plots = self.get_outliers(data=data, cols=cols)
                if isinstance(outlier_plots, str):
                    return outlier_plots
                return outlier_plots["outlier_points"]
            else:
                if self.ts_identifiers:
                    # raise ValueError('Segment data has to be passed to get outliers plot')
                    return "Segment data has to be passed to get outliers plot"
                else:
                    outlier_plots = self.get_outliers(cols=cols)
                    if isinstance(outlier_plots, str):
                        return outlier_plots
                    return outlier_plots["outlier_points"]
        else:
            outliers = outliers_data
        data_ = data if data is not None else self.data
        for col in outliers.index:
            data = data_[[self.ts_column, col]].set_index(self.ts_column)
            outlier_df = data[col].loc[list(outliers[col].keys())].reset_index()
            outlier_df["score"] = outlier_df.apply(
                lambda row: outliers[col][row[self.ts_column]], axis=1
            )
            col_plot = hvPlot(data[col]).line()
            col_outlier_plot = hv.Points(outlier_df, vdims=["score"]).opts(
                hv.opts.Points(color="red", size="score")
            )
            # col_outlier_plot
            outlier_plots[col] = col_plot * col_outlier_plot
        return outlier_plots

    @fail_gracefully(_LOGGER)
    def get_change_points(self, data=None, cols=[], get_indeces=False):
        """Returns the dictionary with no of change points.

        Parameters
        ----------
        data: DataFrame
        cols: list (optional)
        default value - empty list.
        get_indeces: bool, default: False
        if True return's the list of change points in the dictionary

        Returns
        -------
        dictionary
        """
        try:
            import ruptures as rpt
        except ModuleNotFoundError:
            raise Exception(
                "ruptures package not found. Please install with - pip install ruptures"
            )
        cols = cols or self.y_cols
        selected_col = []
        for col_ in cols:
            if col_ not in self.data.numeric_columns:
                _LOGGER.info(
                    f"{col_} is not a numeric column hence change point detection is skipped."
                )
            else:
                selected_col += [col_]
        cols = selected_col
        if len(cols) == 0:
            return "No numeric column passed for change point detection."

        def detect_change_points(series):
            model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
            algo = rpt.Window(width=int(0.02 * len(series)), model=model).fit(
                series.values
            )
            my_bkps = algo.predict(pen=np.log(len(series)) * 2 * (series.std() ** 2))
            if len(series) in my_bkps:
                my_bkps.remove(len(series))
            indices = [
                series.reset_index().iloc[ind - 1][self.ts_column] for ind in my_bkps
            ]
            # values = [series[ind - 1] for ind in my_bkps]
            # cp_df = pd.DataFrame(values, index=indices)
            # return cp_df
            return indices

        def change_point_indeces(df, cols):
            # print('Running for {}'.format(df.name))
            if int(0.02 * len(set(df[self.ts_column]))) / 2 < 2:
                # ensures width/2 >2 in algo = rpt.Window(width=int(0.02 * len(series)), model=model).fit(series.values)
                if self.ts_identifiers:
                    if hasattr(df, "name"):
                        _LOGGER.info(
                            f"Change point detection not done for segment {df.name} due to lack of minimum no. of"
                            f"observations in the series."
                        )
                        return None
                    else:
                        return (
                            "Change point detection not done due to lack of minimum no. of "
                            "observations in the series."
                        )
                else:
                    return "Change point detection not done due to lack of minimum no. of observations in the series."
            change_pts_len = []
            change_pts_list = []
            for col in cols:
                # print('Computing for column - {}'.format(col))
                series = pd.Series(df[col].values, index=df[self.ts_column])
                if len(series) > len(set(series.index.values)):
                    _LOGGER.info(
                        "Time values repeat. Taking the first value for each time."
                    )
                    series = series.groupby(series.index).first()
                change_points = detect_change_points(series)
                change_pts_len += [len(change_points)]
                change_pts_list += [change_points]
            cp_dict = {
                "No of Change Points": pd.Series(change_pts_len, index=cols),
                "Change Points": pd.Series(change_pts_list, index=cols),
            }
            cp_data_ = pd.concat(cp_dict, axis=1)
            return cp_data_

        df = data if data is not None else self.data
        if self.ts_identifiers and data is None:
            cp_df = df.groupby(self.ts_identifiers).apply(
                lambda sdf: change_point_indeces(sdf, cols)
            )
            if cp_df.empty:
                cp_df = (
                    "Change Point detection not performed for any of the segment "
                    "due to lack of minimum no. of observations in all the series."
                )
            else:
                cp_df = cp_df.unstack()
                nil_cols = (
                    cp_df["No of Change Points"]
                    .columns[
                        cp_df["No of Change Points"].apply(lambda col: col.sum() <= 0)
                    ]
                    .tolist()
                )
                if not get_indeces:
                    cp_df = cp_df[["No of Change Points"]]
                if nil_cols:
                    cp_df.drop(nil_cols, axis=1, level=1, inplace=True)
                if cp_df.empty:
                    cp_df = "No change points in the data"
            return cp_df
        else:
            cp_data = change_point_indeces(df, cols)
            if type(cp_data) == str:
                return cp_data
            else:
                cp_series = cp_data["Change Points"]
                cp_counts = cp_data["No of Change Points"]
                if not cp_counts[cp_counts > 0].empty:
                    cp_cols = cp_counts[cp_counts > 0].index.values.tolist()
                    return_dict = {
                        "no_of_change_points": cp_data[["No of Change Points"]],
                        "change_points": self.show_change_points(
                            data=df, cp_data=cp_series[cp_cols], cols=cp_cols
                        ),
                    }
                    if get_indeces:
                        return_dict["Change Points"] = cp_data[["Change Points"]]
                    return return_dict
                else:
                    return "No change points in the data"

    def show_change_points(self, data=None, cols=[], cp_data=None):
        """Show trend change points in timesries.

        Plot of points with vertical marklines indicating an unexpected change in the values of the y_cols.

        Parameters
        ----------
        data: DataFrame
        cols: list (optional)
        default value - empty list.
        cp_data: pandas.DataFrame, default: None
            change point data, if given it will create plot directly from cp_data

        Returns
        -------
        plt: `hvplot`
        """
        cp_plots = {}
        if cp_data is None:
            if data is not None:
                cp_plots = self.get_change_points(data=data, cols=cols)
                if isinstance(cp_plots, str):
                    return cp_plots
                return cp_plots["change_points"]
            else:
                if self.ts_identifiers:
                    # raise ValueError('Segment data has to be passed to get change_points plot')
                    return "Segment data has to be passed to get change_points plot"
                else:
                    cp_plots = self.get_change_points(cols=cols)
                    if isinstance(cp_plots, str):
                        return cp_plots
                    return cp_plots["change_points"]
        else:
            cp_series = cp_data
        data_ = data if data is not None else self.data
        for col in cp_series.index:
            data = data_[[self.ts_column, col]].set_index(self.ts_column)
            cp_indeces = cp_series[col]
            col_plot = hvPlot(data[col]).line()
            for ind in cp_indeces:
                col_plot = col_plot * hv.VLine(ind).opts(
                    hv.opts.VLine(color="red", line_width=0.5)
                )
            cp_plots[col] = col_plot
        return cp_plots

    @fail_gracefully(_LOGGER)
    def ts_summary(self):
        """Returns summary of the time series data."""
        summary_dict = {}
        summary_dict["first_period"] = self.first_period_in_series()
        summary_dict["last_period"] = self.last_period_in_series()
        summary_dict["no_of_periods"] = self.no_of_periods()
        summary_dict["periodicity"] = self._compute_periodicity()

        plot = None
        missing_periods = self.missing_periods()
        if isinstance(missing_periods, dict):
            summary_dict["missing_periods"] = missing_periods["no_of_missing_periods"]
            plot = missing_periods["missing_periods"]
        elif isinstance(missing_periods, str):
            summary_dict["missing_periods"] = missing_periods
        else:
            summary_dict["missing_periods"] = missing_periods[
                list(missing_periods.columns)[0]
            ]
        repeting_periods = self.get_time_repetitions()
        if isinstance(repeting_periods, dict):
            summary_dict["repeating_periods"] = repeting_periods["no_of_repetitions"]
            if plot:
                plot = plot * repeting_periods["repeating_periods"]
            else:
                plot = repeting_periods["repeating_periods"]
        else:
            summary_dict["repeating_periods"] = repeting_periods
        if self.ts_identifiers:
            ts_summary = pd.DataFrame.from_dict(summary_dict)
        else:
            ts_summary = pd.DataFrame(
                summary_dict.values(), index=summary_dict.keys(), columns=["Values"]
            )
        del summary_dict
        gc.collect()
        if not plot:
            return [ts_summary.reset_index()]
        else:
            return {"summary": ts_summary, "visualization": plot}

    @measure_time(_LOGGER)
    def get_time_series_analysis(self):
        """Generate outliers change points first_period_occurrences.

        And last_period_occurrences for time series data.

        Returns
        -------
        dictionary of all outputs.
        """
        ts_analysis_dict = {"time_periods_summary": self.ts_summary()}
        if self.ts_identifiers:

            # ts_analysis_dict['missing_periods'] = self.missing_periods()
            # ts_analysis_dict['repeating_periods'] = self.get_time_repetitions()
            ts_analysis_dict["outliers_in_time_series(es)"] = self.get_outliers()
            ts_analysis_dict["change_points"] = self.get_change_points()
            ts_analysis_dict["outliers_plot"] = self.outliers_plot()
            ts_analysis_dict[
                "first_period_occurrences"
            ] = self.plot_first_period_occurrence()
            ts_analysis_dict[
                "last_period_occurrences"
            ] = self.plot_last_period_occurrence()
        else:
            # ts_analysis_dict['missing_periods'] = self.missing_periods()
            # ts_analysis_dict['repeating_periods'] = self.get_time_repetitions()
            ts_analysis_dict["outliers_in_time_series(es)"] = self.get_outliers()
            ts_analysis_dict["change_points"] = self.get_change_points()

        return ts_analysis_dict

    def get_ts_key_drivers(self, quick=True):
        """Returns dictionary having acf pacf and lag analysis.

        Plots for each indipendent variable for each dependent variable.

        Returns
        -------
        dictionary of holoview plots.
        """
        ts_key_drivers = {}
        for col in self.y_cols:
            ts_key_drivers[col] = {}
            self._set_current_y(col)
            ts_key_drivers[col]["lag_analysis"] = self.lag_analysis(quick=quick)
        return ts_key_drivers
