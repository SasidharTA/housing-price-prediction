import logging
import numpy as np
import pandas as pd
import tigerml.core.dataframe as td
from collections import defaultdict
from tigerml.core.plots import hvPlot
from tigerml.core.preprocessing import Outlier
from tigerml.core.reports import create_report
from tigerml.core.utils import (
    append_file_to_path,
    compute_if_dask,
    fail_gracefully,
    measure_time,
    time_now_readable,
)
from tigerml.core.utils.constants import NA_VALUES, SUMMARY_KEY_MAP

from ...helpers import is_missing, split_sets

_LOGGER = logging.getLogger(__name__)


class HealthMixin:
    """Health mixin class."""

    NA_VALUES = NA_VALUES

    @property
    def na_definition(self):
        """
        Returns a list of NA Values that are considered as missing values.

        Returns
        -------
        list: returns list of NA Values

        """
        return self.NA_VALUES

    def add_to_na_definition(self, value):
        """
        Appends custom NA_VALUES to the list that are considered as missing values.

        Parameters
        ----------
        value: Value that has to be added to NA_VALUES list.

        Returns
        -------
        list: returns list of NA Values

        """
        if value in self.NA_VALUES:
            _LOGGER.info("Value already exists in NA definition")
        else:
            self.NA_VALUES.append(value)
            return self.NA_VALUES

    def remove_from_na_definition(self, value):
        """
        Removes NA_VALUES from the list that are not considered as missing values.

        Parameters
        ----------
        value: Value that has to be removed from NA_VALUES list.

        Returns
        -------
        list: returns list of NA Values

        """
        try:
            self.NA_VALUES.remove(value)
            return self.NA_VALUES
        except ValueError:
            _LOGGER.error("Value does not exist in NA definition")

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def duplicate_columns(self):
        """List duplicate columns in dataframe.

        Returns the duplicate columns that are present for each
        variable in the dataset. If there are no such duplicate
        columns, "No Duplicate Variables" message is displayed.

        Parameters
        ----------
        dataset: pandas.DataFrame

        Returns
        -------
        dups: pandas.DataFrame
        """
        df = self.data
        dups = defaultdict(list)
        sets = split_sets(
            df, df.columns, 0
        )  # splitting columns based on first element matching
        for i in range(1, 4):
            big_sets = [set for set in sets if len(set) > 100]
            if big_sets:
                for set in big_sets:
                    new_sets = split_sets(df, set, i)
                    sets.remove(set)
                    sets += new_sets
            else:
                break
        # final_sets = []
        multi_sets = [set for set in sets if len(set) > 1]
        # for set in multi_sets:
        #     new_sets = split_sets(df, set, "mean")
        #     final_sets += new_sets
        # sets = final_sets
        # multi_sets = [set for set in sets if len(set) > 1]
        # print("Lengths of the sets obtained - {}".format([len(set) for set in sets]))
        for set in multi_sets:
            cols = set
            _LOGGER.info("processing set of - {}".format(cols))
            while cols:
                current_col = cols[0]
                current_col_data = df[current_col]
                remaining_cols = cols[1:]
                for col2 in remaining_cols:
                    other_col_data = df[col2]
                    if (
                        str(df.dtypes[current_col]) == "category"
                        and str(df.dtypes[col2]) == "category"
                        and (
                            len(current_col_data.cat.categories)
                            != len(other_col_data.cat.categories)
                        )
                    ):
                        continue
                    if compute_if_dask(
                        (
                            df[current_col].iloc[:1000].astype(str)
                            == df[col2].iloc[:1000].astype(str)
                        ).all()
                    ) and compute_if_dask(
                        (df[current_col].astype(str) == df[col2].astype(str)).all()
                    ):
                        dups[SUMMARY_KEY_MAP.variable_names].append(current_col)
                        dups[SUMMARY_KEY_MAP.duplicate_col].append(col2)
                        cols.remove(col2)
                cols.remove(current_col)
        dups = pd.DataFrame(dups)
        if dups.empty:
            dups = "No duplicate variables"
        else:
            dups = dups.loc[
                dups[SUMMARY_KEY_MAP.variable_names].isin(
                    dups[SUMMARY_KEY_MAP.duplicate_col]
                )
                == False  # noqa
            ]
            dups = (
                dups.groupby(SUMMARY_KEY_MAP.variable_names)
                .agg({SUMMARY_KEY_MAP.duplicate_col: lambda s: ", ".join(s)})
                .reset_index()
            )
        self.duplicate_columns_result = dups
        return dups

    def _compute_data_health(self):
        """Returns data health."""
        df = self.data
        dtypes = td.DataFrame(df.dtypes.rename("dtype"))
        dtypes[SUMMARY_KEY_MAP.dtype] = "*Unknown*"
        dtypes.loc[
            dtypes.dtype.astype(str).str.contains("float|int"),
            SUMMARY_KEY_MAP.dtype,
        ] = "Numeric"
        dtypes.loc[
            dtypes.dtype.astype(str).str.contains("date"), SUMMARY_KEY_MAP.dtype
        ] = "Date"
        dtypes.loc[
            dtypes[SUMMARY_KEY_MAP.dtype].isin(["Numeric", "Date"]) == False,  # noqa
            SUMMARY_KEY_MAP.dtype,
        ] = "Others"

        no_of_columns = len(df.columns)
        pie1 = dtypes.groupby(SUMMARY_KEY_MAP.dtype).size()
        pie1 = (pie1 / no_of_columns).to_dict()
        if not ("Numeric" in pie1.keys()):
            pie1.update({"Numeric": 0})
        if not ("Others" in pie1.keys()):
            pie1.update({"Others": 0})

        datatype_dict = {}

        for i in sorted(pie1):
            datatype_dict.update({i: pie1[i]})
        pie1 = datatype_dict

        missing_value = is_missing(df, self.NA_VALUES).sum().sum()
        pie2 = missing_value / float(compute_if_dask(df.size))
        pie2 = {"Available": (1 - pie2), "Missing": pie2}
        # pie2 = pd.DataFrame(pie2, index=['Missing Values'])
        # pie3
        duplicate_value = df.duplicated().sum() / len(df)
        # pie3 = (duplicate_value / float(compute_if_dask(df.shape[0])))
        # pie3 = {"Unique": (1 - pie3), "Duplicate": pie3}

        # pie3 should be multiplied with no_of_rows because it checks the duplicate obs. all other
        # plots in the function work on column basis.
        pie3 = duplicate_value
        pie3 = {"Unique": (1 - pie3), "Duplicate": pie3}

        pie4 = self.duplicate_columns()
        if type(pie4) == str:
            pie4 = {"Unique": 1, "Duplicate": 0}
            duplicate_var = 0
        else:
            duplicate_var = 1 + (", ".join(pie4.Duplicates).count(","))
            pie4 = duplicate_var / float(compute_if_dask(df.shape[1]))
            pie4 = {"Unique": (1 - pie4), "Duplicate": pie4}

        data_dict = {
            "Datatypes": pie1,
            "Missing Values": pie2,
            "Duplicate Values": pie3,
            "Duplicate Columns": pie4,
        }
        from tigerml.core.utils import flatten_list

        df_dict = {
            "type": flatten_list([[x] * len(data_dict[x]) for x in data_dict.keys()]),
            "labels": list(pie1.keys())
            + list(pie2.keys())
            + list(pie3.keys())
            + list(pie4.keys()),
            "values": [i * 100 for i in list(pie1.values())]
            + [i * 100 for i in list(pie2.values())]
            + [i * 100 for i in list(pie3.values())]
            + [i * 100 for i in list(pie4.values())],
        }
        df = pd.DataFrame(df_dict)
        df = df.set_index(["type", "labels"])
        return df

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def data_health(self):
        """Return data health summary.

        Data health plot containing - Data types, Missing data points, Duplicate Observations, Duplicate
        Variables. The function prints 4 pie charts to describe the datatype distribution and the health of the dataset.

        Returns
        -------
        plot : `hvplot`
            The plot contains 4 subplots with:

            Data types             - The percentage distribution among different
                                     data types
            Missing data points    - The percentage of missing data compared to
                                     the total data points available(`nrow x ncol`)
            Duplicate Observations - The percentage of rows that are an exact
                                     replica of another row
            Duplicate Variables    - The percentage of columns that are an exact
                                     replica of another column
        """

        df = self._compute_data_health()
        return self._plot_data_health(df)

    def _plot_data_health(self, df):
        # this multipliers resolves the issue of duplicate columns as it's values are multiplied by 1 and others
        # with no_of_columns. which was needed for the correct metrics.
        final_plot = None
        for metric in df.index.get_level_values(0).unique():
            plot = (
                hvPlot(df.loc[metric].T)
                .bar(stacked=True, title=metric, height=100, invert=True)
                .opts(xticks=list([i for i in range(df.shape[1])]))
            )
            if final_plot:
                final_plot += plot
            else:
                final_plot = plot
        return final_plot.cols(1).opts(title="Data Shape:" + str(self.data.shape))

    def _missing_values(self):
        """Returns a pandas dataframe with the information about missing values in the dataset."""
        df = self.data
        # df = df.apply(lambda s: compute_if_dask(is_missing(s, self.NA_VALUES).sum()))
        df = is_missing(df, self.NA_VALUES).sum()
        df = df.reset_index()
        df = df.rename(
            columns=dict(
                zip(
                    list(df.columns),
                    [SUMMARY_KEY_MAP.variable_names, SUMMARY_KEY_MAP.num_missing],
                )
            )
        )
        df[SUMMARY_KEY_MAP.perc_missing] = (
            df[SUMMARY_KEY_MAP.num_missing] / float(self.data.shape[0]) * 100
        )
        return df

    def _compute_missing_plot(self):
        return self.missing_value_summary()

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def missing_value_summary(self):
        """Returns the summary of missing values.

        This function describes the share of missing values for each variable
        in the dataset. If there are no missing values, "No Missing Values"
        message is displayed, else a table containing the percentage of
        missing for all variables with missing values are displayed

        Returns
        -------
        df: pandas.DataFrame
        """
        df = self._missing_values()
        df = df.loc[df[SUMMARY_KEY_MAP.num_missing] != 0].reset_index(drop=True)
        df = df.rename(
            columns={
                SUMMARY_KEY_MAP.num_missing: SUMMARY_KEY_MAP.num_missing
                + " (out of "
                + str(len(self.data))
                + ")"
            }
        )
        if df.empty:
            return "No Missing Values"
        else:
            return df

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def missing_plot(self):
        """Missing value summary plot.

        Plot form of `missing_value_summary`.

        Returns
        -------
        f: `hvplot`
            missing_plot returns a bar plot with the following axis:

            X.axis - % of missing observation bucket
            Y.axis - Number of variables

        """
        # plt.close('all')
        df = self.data
        missing_values = self._missing_values()
        break_value = [0, 5, 10, 20, 30, 40, 50, 100]
        lab_value = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%"]
        cuts = pd.cut(
            missing_values[SUMMARY_KEY_MAP.perc_missing],
            bins=break_value,
            labels=lab_value,
            right=True,
        )
        cuts = cuts.value_counts().reindex(lab_value)
        remaining_cols = len(df.columns) - cuts.sum()
        cuts = td.concat([td.Series([remaining_cols], index=["No Missing"]), cuts])
        plot = hvPlot(cuts).bar(
            rot=0,
            title="Missing Variables - Data Shape:" + str(self.data.shape),
            xlabel="# of missing observations",
            ylabel="# of variables",
        )
        return plot

    def get_outliers_df(self):
        """Returns the data frame with outlier analysis table for self.data."""
        data = self.data
        return HealthMixin.get_outliers_df_for_data(data)

    @staticmethod
    def get_outliers_df_for_data(data):
        """Returns the data frame with outlier analysis table for any provided data."""
        if pd.__version__ >= "1.0.0":
            pd.set_option("mode.use_inf_as_na", True)
        outlier_col_labels = [
            "< (mean-3*std)",
            "> (mean+3*std)",
            "< (1stQ - 1.5 * IQR)",
            "> (3rdQ + 1.5 * IQR)",
        ]
        mean_outliers = Outlier(method="mean").fit(data).get_outlier_nums(data)
        median_outliers = Outlier(method="median").fit(data).get_outlier_nums(data)
        outliers_df = pd.DataFrame.from_dict(mean_outliers)
        outliers_df = pd.concat([outliers_df, pd.DataFrame.from_dict(median_outliers)])
        outliers_df = outliers_df.reset_index(drop=True).T
        outliers_df.rename(
            columns=dict(zip(list(outliers_df.columns), outlier_col_labels)),
            inplace=True,
        )
        outliers_df["-inf"] = 0
        outliers_df["+inf"] = 0
        if pd.__version__ >= "1.0.0":
            pd.set_option("mode.use_inf_as_na", False)
        outliers_df["-inf"].loc[outliers_df.index] = (data == -np.inf).sum().values
        outliers_df["+inf"].loc[outliers_df.index] = (data == +np.inf).sum().values
        outliers_sum = outliers_df.sum(axis=1)
        outliers_df = outliers_df[outliers_sum > 0]
        if outliers_df.empty:
            return "No Outlier Values"
        outliers_df.insert(
            loc=0, column="N", value=[data.shape[0]] * (outliers_df.shape[0])
        )
        outliers_df.index.name = "Feature"
        return outliers_df

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def health_analysis(self, save_as=None, save_path=""):
        """Data health report.

        Compiles outputs from data_health, missing_plot, missing_value_summary and get_outliers_df as a report.

        Parameters
        ----------
        save_as : str, default=None
            You need to pass only extension here ".html" or ".xlsx". If none results will not be saved.
        save_path : str, default=''
            Location where report to be saved. By default report saved in working directory.
            This should be without extension just complete path where you want to save, file name will be taken by default.

        Examples
        --------
        >>> from tigerml.eda import EDAReport
        >>> import pandas as pd
        >>> df = pd.read_csv("titatic.csv")
        >>> an = EDAReport(df)
        >>> an.health_analysis()
        """
        # data_shape = str(self.data.shape)
        self.health_analysis_report = {}
        self.health_analysis_report.update({"health_plot": self.data_health()})
        self.health_analysis_report.update({"missing_plot": self.missing_plot()})
        self.health_analysis_report.update(
            {"missing_value_summary": self.missing_value_summary()}
        )
        self.health_analysis_report.update(
            {
                "duplicate_columns": self.duplicate_columns_result
                if hasattr(self, "duplicate_columns_result")
                else self.duplicate_columns()
            }
        )
        self.health_analysis_report.update(
            {"outliers_in_features": self.get_outliers_df()}
        )
        if save_as:
            default_report_name = "health_analysis_report_at_{}".format(
                time_now_readable()
            )
            save_path = append_file_to_path(save_path, default_report_name + save_as)
            create_report(
                self.health_analysis_report,
                path=save_path,
                format=save_as,
            )
        return self.health_analysis_report
