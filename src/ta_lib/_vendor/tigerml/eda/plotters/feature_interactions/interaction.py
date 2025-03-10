import holoviews as hv
import logging
import numpy as np
import pandas as pd
import tigerml.core.dataframe as td
from functools import partial
from sklearn.metrics.cluster import mutual_info_score
from tigerml.core.common import compute_correlations
from tigerml.core.plots import hvPlot
from tigerml.core.utils import get_x_y_vars, normalized
from tigerml.core.utils.constants import SUMMARY_KEY_MAP
from tigerml.core.utils.pandas import (
    get_bool_cols,
    get_cat_cols,
    get_dt_cols,
    get_num_cols,
)

_LOGGER = logging.getLogger(__name__)


def get_bivariate_plot(df, dtypes, corr_df, x_var, y_var):
    MAX_CAT_LEVELS = 20
    title = None
    if x_var == y_var:
        data = df[x_var].to_frame()
        y_var = f"{x_var}_copy"
        data[y_var] = df[x_var]
        dtypes[y_var] = dtypes[x_var]
        if corr_df is not None:
            title = "Correlation: 1"
    else:
        data = df[[x_var, y_var]]
        if corr_df is not None:
            corr_df_ = (
                corr_df.set_index(
                    [SUMMARY_KEY_MAP.variable_1, SUMMARY_KEY_MAP.variable_2]
                ).copy()
                if SUMMARY_KEY_MAP.variable_1 in corr_df.columns
                else corr_df.copy()
            )
            try:
                title = f"Correlation: {round(corr_df_.loc[x_var, y_var][SUMMARY_KEY_MAP.corr_coef], 3)}"
            except KeyError:
                try:
                    title = f"Correlation: {round(corr_df_.loc[y_var, x_var][SUMMARY_KEY_MAP.corr_coef], 3)}"
                except KeyError:
                    title = f"Correlation: NA"
    clean_data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[x_var, y_var], how="all"
    )
    for col in [x_var, y_var]:
        if dtypes[col] == "cat" and data[col].nunique() > MAX_CAT_LEVELS:
            from tigerml.core.dataframe.helpers import detigerify

            top_levels = (
                data[col]
                .value_counts()
                .sort_values(ascending=False)
                .head(MAX_CAT_LEVELS)
                .index.tolist()
            )
            other_levels = (
                data[col][~detigerify(data[col].isin(top_levels))].unique().tolist()
            )
            others_placeholder = f"Others ({len(other_levels)} levels)"
            if str(data[col].dtype) == "category":
                data[col] = data[col].cat.add_categories(others_placeholder)
            data[col][~detigerify(data[col].isin(top_levels))] = others_placeholder
            data[col] = data[col].astype(str)
        if dtypes[col] == "bool":
            data[col] = data[col].astype(str)
    plotter = hvPlot(data)
    if (dtypes[x_var] == "conti") and (dtypes[y_var] == "conti"):
        if len(df) > 5000:
            fig = plotter.hexbin(x=x_var, y=y_var, width=600)
        else:
            fig = plotter.scatter(x=x_var, y=y_var, width=600)
        # fig = hvPlot(pd.DataFrame.from_dict({'x': [1,2], 'y': [1,2]})).line()
    elif ((dtypes[x_var] == "conti") and (dtypes[y_var] in ["cat", "bool"])) or (
        (dtypes[x_var] in ["cat", "bool"]) and (dtypes[y_var] == "conti")
    ):
        kwargs = {}
        if dtypes[x_var] == "conti":
            kwargs["invert"] = True
        group_col, count_col = (
            (x_var, y_var) if dtypes[y_var] == "conti" else (y_var, x_var)
        )
        fig = plotter.violin(y=count_col, by=group_col, **kwargs)
        try:
            # Fixing No. of bins (n) = N^(0.7) based on
            # [1] B. C. Ross "Mutual Information between Discrete and Continuous Data Sets". PLoS ONE 9(2), 2014
            # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357#s3
            n_bins = round(len(clean_data[count_col]) ** 0.7)
            binned_data = pd.cut(clean_data[count_col], bins=n_bins).astype(str)
            mutual_info = mutual_info_score(clean_data[group_col].values, binned_data)
            title = (
                f"Mutual Information: "
                f"{round(mutual_info, 3)} ('{count_col}' is binned for computation)"
            )
        except ValueError:
            title = f"Mutual Information: NA"
    elif dtypes[x_var] in ["cat", "bool"] and dtypes[y_var] in [
        "cat",
        "bool",
    ]:
        group_col, count_col = (x_var, y_var)
        if dtypes[x_var] == "cat":
            data[x_var] = data[x_var].astype(str)
        if dtypes[y_var] == "cat":
            data[y_var] = data[y_var].astype(str)
        fig = hvPlot(
            data.groupby([group_col, count_col])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
        ).bar(
            ylabel="Number of " "Observations",
            x=group_col,
            y="count",
            by=count_col,
            stacked=True,
            rot=45,
        )
        try:
            title = (
                f"Mutual Information: "
                f"{round(mutual_info_score(clean_data[group_col].values, clean_data[count_col].values), 3)}"
            )
        except ValueError:
            title = f"Mutual Information: NA"
    else:
        fig = f"Error unable to identify right plot for {x_var}, {y_var}"
        _LOGGER.error(fig)
    if title:
        fig.opts(title=title)
    return fig


def get_bivariate_plot_for_comb(df, dtypes, comb_str):
    y_var, x_var = comb_str.split(" (corr: ")[0].split(" vs ")
    return get_bivariate_plot(df, dtypes, None, x_var, y_var)


def get_bivariate_plot_fixed_x(df, dtypes, x_var, y_label):
    y_var = y_label.split(" (corr: ")[0]
    return get_bivariate_plot(df, dtypes, None, x_var, y_var)


def get_bivariate_plot_fixed_y(df, dtypes, y_var, x_label):
    x_var = x_label.split(" (corr: ")[0]
    return get_bivariate_plot(df, dtypes, None, x_var, y_var)


def sort_list_by_corr(keys):
    sorted_keys = sorted(
        keys, key=lambda val: val.split("(corr: ")[-1][:-1], reverse=True
    )
    return sorted_keys


class JointPlot:
    """Returns bivariate plots of x_vars and y_vars depending on their datatype.

    Generates univariate plots for both x_vars and y_vars to see their distribution.
    """

    def __init__(self, data):
        """Class initialization.

        Parameters
        ----------
        data: pd.DataFrame
            It is from where we'll fetch the data_columns.
        """
        self.data = data
        self.CAT_TYPE = "cat"
        self.BOOL_TYPE = "bool"
        self.NUM_TYPE = "conti"
        self.buckets = {
            "conti_v_conti": [],
            "conti_v_cat": [],
            "cat_v_conti": [],
            "cat_v_cat": [],
        }

    def compute_dtypes(self, x_vars, y_vars, df):
        """Returns data types."""
        dtypes = dict.fromkeys(set(x_vars).union(set(y_vars)), "conti")
        bool_cols = get_bool_cols(df)
        cat_cols = get_cat_cols(df)
        dt_cols = get_dt_cols(df)
        for col in cat_cols + dt_cols:
            dtypes[col] = "cat"
        for col in bool_cols:
            dtypes[col] = "bool"
        self.dtypes = dtypes

    def compute_corr_df(self, x_vars, y_vars, abs_corr_thresold=None, top_n=None):
        """Returns corr_df."""
        self.corr_df = CorrelationTable(self.data).get_plot(
            x_vars=x_vars, y_vars=y_vars
        )
        if len(self.corr_df) > 0:
            self.corr_df.sort_values(
                SUMMARY_KEY_MAP.corr_coef, ascending=False, inplace=True
            )
        else:
            self.corr_df = pd.DataFrame.from_dict(
                {
                    SUMMARY_KEY_MAP.variable_1: [],
                    SUMMARY_KEY_MAP.variable_2: [],
                    SUMMARY_KEY_MAP.corr_coef: [],
                    SUMMARY_KEY_MAP.abs_corr_coef: [],
                }
            )
        if top_n:
            self.corr_df = self.corr_df.iloc[:top_n]
        elif abs_corr_thresold is not None and abs_corr_thresold > 0:
            self.corr_df = self.corr_df[
                self.corr_df[SUMMARY_KEY_MAP.abs_corr_coef] >= abs_corr_thresold
            ]

    def get_bucketed_combs(self, x_vars, y_vars, abs_corr_thresold, top_n):
        """Gets bucketed combs."""
        self.compute_corr_df(x_vars, y_vars, abs_corr_thresold, top_n)
        combinations = self.corr_df[
            [SUMMARY_KEY_MAP.variable_1, SUMMARY_KEY_MAP.variable_2]
        ].values.tolist()
        self.corr_df.set_index(
            [SUMMARY_KEY_MAP.variable_1, SUMMARY_KEY_MAP.variable_2], inplace=True
        )
        for x_var, y_var in combinations:
            x_type = (
                "cat" if self.dtypes[x_var] == self.BOOL_TYPE else self.dtypes[x_var]
            )
            y_type = (
                "cat" if self.dtypes[y_var] == self.BOOL_TYPE else self.dtypes[y_var]
            )
            self.buckets[f"{y_type}_v_{x_type}"].append((x_var, y_var))

    def get_bucketed_xy_vars(self, x_vars, y_vars):
        """Returns categorical column names to bucket."""
        dtypes = self.dtypes
        x_cat_cols = [x for x in x_vars if dtypes[x] in [self.CAT_TYPE, self.BOOL_TYPE]]
        x_num_cols = list(set(x_vars) - set(x_cat_cols))
        y_cat_cols = [y for y in y_vars if dtypes[y] in [self.CAT_TYPE, self.BOOL_TYPE]]
        y_num_cols = list(set(y_vars) - set(y_cat_cols))
        if x_num_cols and y_num_cols:
            self.buckets[f"{self.NUM_TYPE}_v_{self.NUM_TYPE}"] = [
                x_num_cols,
                y_num_cols,
            ]
        if x_cat_cols and y_num_cols:
            self.buckets[f"{self.NUM_TYPE}_v_{self.CAT_TYPE}"] = [
                x_cat_cols,
                y_num_cols,
            ]
        if x_num_cols and y_cat_cols:
            self.buckets[f"{self.CAT_TYPE}_v_{self.NUM_TYPE}"] = [
                x_num_cols,
                y_cat_cols,
            ]
        if x_cat_cols and y_cat_cols:
            self.buckets[f"{self.CAT_TYPE}_v_{self.CAT_TYPE}"] = [
                x_cat_cols,
                y_cat_cols,
            ]

    def create_dmap_for_combs(self, df):
        """Returns dmap for combs."""
        dmap_dict = {}
        for key in self.buckets:
            if not self.buckets[key]:
                continue
            combinations = self.buckets[key]
            combinations = [
                "{} vs {} (corr: {})".format(
                    y_var,
                    x_var,
                    round(self.corr_df.loc[x_var, y_var][SUMMARY_KEY_MAP.corr_coef], 3),
                )
                for x_var, y_var in combinations
            ]
            plot = hv.DynamicMap(
                partial(get_bivariate_plot_for_comb, df, self.dtypes), kdims=["y_vs_x"]
            )
            # kwargs = {key: combinations}
            plot = plot.redim.values(y_vs_x=sort_list_by_corr(combinations))
            plot.opts(width=700, height=400)
            dmap_dict[key] = plot
        return dmap_dict

    def create_dmap_for_xy(self, df):
        """Returns dmap for categorical variables to bucket."""
        dmap_dict = {}
        for key in self.buckets:
            if not self.buckets[key]:
                continue
            x_vars, y_vars = self.buckets[key]
            self.compute_corr_df(x_vars, y_vars)
            self.corr_df.set_index(
                [SUMMARY_KEY_MAP.variable_1, SUMMARY_KEY_MAP.variable_2], inplace=True
            )
            if len(x_vars) == 1 and len(y_vars) == 1:
                # key = f'{x_vars[0]} vs {y_vars[0]}'
                # plot_dict = {key: None}
                # title = f'Correlation: {round(self.corr_df.loc[x_vars[0], y_vars[0]][SUMMARY_KEY_MAP.corr_coef], 3)}'
                dmap_dict[key] = get_bivariate_plot(
                    df, self.dtypes, self.corr_df, x_vars[0], y_vars[0]
                )
            else:
                if len(x_vars) == 1 or len(y_vars) == 1:
                    if len(x_vars) == 1:
                        fixed_input = x_vars[0]
                        varying_input = y_vars
                        correlation_index = SUMMARY_KEY_MAP.variable_2
                        bivariate_func = get_bivariate_plot_fixed_x
                        kdims = ["y_label"]
                    else:
                        fixed_input = y_vars[0]
                        varying_input = x_vars
                        correlation_index = SUMMARY_KEY_MAP.variable_1
                        bivariate_func = get_bivariate_plot_fixed_y
                        kdims = ["x_label"]
                    # corr_df = CorrelationTable(self.data).get_plot(x_vars=x_vars, y_vars=y_vars)
                    self.corr_df = self.corr_df.reset_index().set_index(
                        correlation_index
                    )
                    input_vals = []
                    for val in varying_input:
                        if (
                            val != fixed_input
                        ):  # ignore if current column is same as fixed column
                            if (
                                val in self.corr_df.index.values
                            ):  # check if column exists
                                corr = round(
                                    self.corr_df.loc[val][SUMMARY_KEY_MAP.corr_coef], 3
                                )
                            else:
                                try:  # Check if an encoded version of the column exists
                                    index = [
                                        col
                                        for col in self.corr_df.index.values
                                        if col
                                        in [
                                            f"target_encoded_{col}",
                                            "label_encoded_{col}",
                                        ]
                                    ][0]
                                    corr = round(
                                        self.corr_df.loc[index][
                                            SUMMARY_KEY_MAP.corr_coef
                                        ],
                                        3,
                                    )
                                except IndexError:
                                    corr = "NA"
                            label = f"{val} (corr: {corr})"
                            input_vals.append(label)
                    input_vals = sort_list_by_corr(input_vals)
                    dim_values = {kdims[0]: input_vals}
                    func = partial(bivariate_func, df, self.dtypes, fixed_input)
                else:
                    kdims = ["x_var", "y_var"]
                    dim_values = {"x_var": x_vars, "y_var": y_vars}
                    func = partial(get_bivariate_plot, df, self.dtypes, self.corr_df)
                plot = hv.DynamicMap(func, kdims=kdims)
                plot = plot.redim.values(**dim_values)
                plot.opts(width=700, height=400)
                dmap_dict[key] = plot
        return dmap_dict

    def get_plots(
        self,
        x_vars=None,
        y_vars=None,
        abs_corr_thresold=0,
        top_n=None,
        return_dict=False,
    ):
        """Returns `Scatter` plot if both x_var and y_var are Continuous.

        `violin` plot if one of them is Continuous and other one is either Boolean or Categorical
        `heatmap` if both x_var and y_var are both Categorical or Boolean.
        `bar` plot if one of them is Categorical and other one is Boolean.

        Parameters
        ----------
        x_vars : list of variables for which we need to plot the bivariate plots.
        y_vars : list of variables for which we need to plot the bivariate plots.

        Returns
        -------
        plot : `hvplot`
            interactive plots depending on the datatype of `x_var` and `y_var`
        """
        x_vars = x_vars or []
        y_vars = y_vars or []
        x_vars, y_vars, req_cols = get_x_y_vars(list(self.data.columns), x_vars, y_vars)
        df = self.data[req_cols]
        self.compute_dtypes(x_vars, y_vars, df)
        if return_dict:
            plot_dict = {}
            self.compute_corr_df(x_vars, y_vars, abs_corr_thresold, top_n)
            if abs_corr_thresold > 0 or top_n:
                combinations = self.corr_df[
                    [SUMMARY_KEY_MAP.variable_1, SUMMARY_KEY_MAP.variable_2]
                ].values.tolist()
                self.corr_df.set_index(
                    [SUMMARY_KEY_MAP.variable_1, SUMMARY_KEY_MAP.variable_2],
                    inplace=True,
                )
                for x_var, y_var in combinations:
                    key = f"{y_var} vs {x_var}"
                    # title = f'Correlation: {round(self.corr_df.loc[x_var, y_var][SUMMARY_KEY_MAP.corr_coef], 3)}'
                    plot_dict[key] = get_bivariate_plot(
                        df, self.dtypes, self.corr_df, x_var, y_var
                    )
            else:
                if len(x_vars) < len(y_vars):
                    first_list = x_vars
                    second_list = y_vars
                else:
                    first_list = y_vars
                    second_list = x_vars
                for var_1 in first_list:
                    plot_dict[var_1] = {}
                    for var_2 in second_list:
                        # title = f'Correlation: {round(self.corr_df.loc[x_var, y_var][SUMMARY_KEY_MAP.corr_coef], 3)}'
                        plot_dict[var_1][var_2] = get_bivariate_plot(
                            df, self.dtypes, self.corr_df, var_1, var_2
                        )
                if len(plot_dict) == 1:
                    plot_dict = list(plot_dict.values())[0]
            return plot_dict
        else:
            if abs_corr_thresold > 0 or top_n:
                self.get_bucketed_combs(x_vars, y_vars, abs_corr_thresold, top_n)
                dmap_dict = self.create_dmap_for_combs(df)
            else:
                self.get_bucketed_xy_vars(x_vars, y_vars)
                dmap_dict = self.create_dmap_for_xy(df)
            if len(dmap_dict) > 1:
                import panel as pn

                plot = pn.Column()
                for key in dmap_dict:
                    plot.append(
                        pn.pane.Markdown(
                            "### {}".format(
                                key.replace("conti", "Continuous Variables")
                                .replace("cat", "Categorical Variables")
                                .replace("_", " ")
                            )
                        )
                    )
                    plot.append(dmap_dict[key])
            elif len(dmap_dict) == 1:
                plot = list(dmap_dict.values())[0]
            else:
                plot = "No correlation values"
            return plot


class CovarianceHeatmap:
    """Returns a Heatmap of covariance for the variables in the given data."""

    def fit(self, data):
        """Returns an updated object with initialization of df which is the input data.

        Parameters
        ----------
        data : pd.DataFrame
        """
        self.df = data
        return self

    def get_plot_data(self):
        """Returns a pandas dataframe which contains the covariance after the data is normalized."""
        from tigerml.core.utils import compute_if_dask

        cov = self.df.cov()
        return normalized(compute_if_dask(cov))

    def get_plot(self):
        """Returns an interactive heatmap of covariance after the data is normalized."""
        data = self.get_plot_data()
        return hvPlot(data).heatmap(rot=45, height=450)


class CorrelationTable:
    """Returns the correlation of the variables in the given data in the form of a table."""

    def __init__(self, data):
        """Class initialization.

        Parameters
        ----------
        data: pd.DataFrame
        """
        self.data = data

    def get_plot(self, x_vars=None, y_vars=None):
        """Computes correlation for plot."""
        return compute_correlations(self.data, x_vars, y_vars)


class CorrelationHeatmap:
    """Returns a Heatmap of correlation for the variables in the given data."""

    def __init__(self, data):
        """Class initialization.

        Parameters
        ----------
        data: pd.DataFrame
        """
        self.data = data

    def get_plot_data(self, x_vars=None, y_vars=None):
        """Returns a correlation tables for x_vars and y_vars which are taken from  self.data."""
        x_vars, y_vars, req_cols = get_x_y_vars(list(self.data.columns), x_vars, y_vars)
        df = self.data[req_cols]
        df = df[get_num_cols(df)]
        if not df.empty:
            corr_df = df.corr()
            return corr_df

    def get_plot(self, x_vars=None, y_vars=None):
        """Returns a correlation tables for x_vars and y_vars which are taken from  self.data.

        Parameters
        ----------
        x_vars : list of variables for which we need to plot the Heatmap.
        y_vars : list of variables for which we need to plot the Heatmap.

        Returns
        -------
        Heatmap : Containing the correlation of all variables with each other.
        """
        corr_df = self.get_plot_data(x_vars, y_vars)
        heatmap = hvPlot(corr_df).heatmap(rot=45, height=450)
        return heatmap
