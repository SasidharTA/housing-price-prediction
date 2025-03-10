import holoviews as hv
import numpy as np
import pandas as pd
from functools import partial
from scipy.stats import shapiro
from tigerml.core.plots import hvPlot
from tigerml.core.plots.bokeh import add_to_secondary, finalize_axes_right
from tigerml.core.utils import compute_if_dask
from tigerml.core.utils.pandas import get_cat_cols, get_num_cols


def non_numeric_frequency_plot(df, col):
    """Returns a dictionary of interactive frequency plots and summary table for non-numeric cols.

    Parameters
    ----------
    df : pd.DataFrame
    cols : `list`
        default : empty, takes the requested columns in the given dataframe, otherwise takes all the columns given
        in the list.

    Returns
    -------
    plot : `dict`
        for all the non-numeric required columns in the list if it is not empty else all non-numeric from the given data.
    """
    MAX_LEVELS = 20
    top_levels = False
    series = df[col]
    summary_df = series.describe().to_frame().T.round(2)
    summary_table = hvPlot(summary_df).table(
        columns=list(summary_df.columns), height=60, width=600
    )
    freq_data = series.value_counts().head(MAX_LEVELS).sort_values(ascending=True)
    if series.nunique() > MAX_LEVELS:
        top_levels = True
        freq_data.index = freq_data.index.astype(str)
        freq_data[f"Others ({series.nunique() - MAX_LEVELS} levels)"] = (
            series.value_counts().sum() - freq_data.sum()
        )
    freq_plot = hvPlot(freq_data).bar(
        title="Frequency Plot for {}{}".format(
            col, f" (top {MAX_LEVELS})" if top_levels else ""
        ),
        invert=True,
        width=600,
    )
    plot = (freq_plot + summary_table).cols(1)
    return plot


def density_plot(df, col):
    """Returns a dict of interactive density plots and numeric summary for the given columns or all numeric columns.

     For the given data if the cols list is empty.

    Parameters
    ----------
    df : pd.DataFrame
    cols : `list`
        default : empty, takes the requested columns in the given dataframe, otherwise takes all the columns given
        in the list.

    Returns
    -------
    plot : `dict`
        for all the requested numeric columns defined in the list if it is not empty else all non-numeric from the given data.
    """
    series = df[col]
    if np.any(series):
        series = series.dropna()
    summary_df = series.describe().to_frame().T.round(2)
    summary_table = hvPlot(summary_df).table(
        columns=list(summary_df.columns), height=60, width=600
    )
    try:
        hist_plot = hv.Histogram(np.histogram(series, bins=20))
        density_plot = hvPlot(series).kde(
            title="Density Plot for {}".format(col), width=600
        )
        hooks = [add_to_secondary, finalize_axes_right]
        complete_plot = hist_plot.options(
            color="#00fff0", xlabel=col
        ) * density_plot.options(hooks=hooks)
        plot = (complete_plot + summary_table).cols(1)
    except Exception as e:
        plot = "Could not generate. Error - {} ".format(e)
    return plot


class FrequencyPlot:
    """Returns a non numeric frequency plot for categorical variables and density plot for numeric variables."""

    def __init__(self, data):
        """Class initializer.

        Parameters
        ----------
        data : pd.DataFrame
        """
        self.data = data

    # def get_plot(self, type, col):
    #     assert type in ['cat', 'num']
    #     if type == 'cat':
    #         return non_numeric_frequency_plot(self.data, col)
    #     else:
    #         return density_plot(self.data, col)

    def get_plots(self, cols=None):
        """Return a plot dict which gives a non-numeric freq plot if the key is a categorical else density plot.

        Parameters
        ----------
        cols: `list`
            list of requested columns

        Returns
        -------
        plot : `plot`
            if variables are categorical : non_numeric_freq_plot
            if variables are continuous continuous : density plot
        """
        # if cols:
        #     df = self.data[cols]
        # else:
        df = self.data
        if cols is None:
            cols = list(df.columns)
        num_cols = [col for col in cols if col in get_num_cols(df)]
        cat_cols = [col for col in cols if col in get_cat_cols(df)]
        # for categorical columns
        import holoviews as hv

        if cat_cols:
            if len(cat_cols) > 1:
                plot = hv.DynamicMap(
                    partial(non_numeric_frequency_plot, self.data), kdims=["col"]
                )
                plot = plot.redim.values(col=cat_cols)
            else:
                plot = non_numeric_frequency_plot(self.data, cat_cols[0])

        # for numerical columns
        elif num_cols:
            if len(num_cols) > 1:
                plot = hv.DynamicMap(partial(density_plot, self.data), kdims=["col"])
                plot = plot.redim.values(col=num_cols)
            else:
                plot = density_plot(self.data, num_cols[0])
        else:
            plot = "No columns in data"
        return plot


def percentile_plot(df, col):
    series = df[col]
    t2_cellText = (
        series.describe(percentiles=[0.002, 0.004, 0.006, 0.008, 0.01])
        .iloc[3:-2]
        .round(2)
        .reset_index()
        .values
    ).T
    t2 = pd.DataFrame(t2_cellText).set_index(pd.Index(["Percentile", "Value"]))
    t2 = t2.rename(columns=t2.loc["Percentile"]).drop("Percentile")
    t3_cellText = (
        series.describe(percentiles=[0.99, 0.992, 0.994, 0.996, 0.998])
        .iloc[5:-1]
        .round(2)
        .reset_index()
        .values
    ).T
    t3 = pd.DataFrame(t3_cellText).set_index(pd.Index(["Percentile", "Value"]))
    t3 = t3.rename(columns=t3.loc["Percentile"]).drop("Percentile")
    t4 = series.describe(percentiles=np.linspace(0, 1, 21)).iloc[4:-1]
    t4.index = [i for i in range(0, 101, 5)]
    t4.index = [i for i in range(0, 101, 5)]
    p_tables = (
        hvPlot(t2).table(columns=list(t2.columns), height=70, width=600)
        + hvPlot(t3).table(columns=list(t3.columns), height=70, width=600)
    ).cols(1)
    plot = (
        hvPlot(t4).bar(width=600, title="Percentile Plot for {}".format(col)) + p_tables
    ).cols(1)
    return plot


class PercentilePlot:
    """Returns the plot of percentile between 0 and 100 for every variable in the dataset.

    Percentiles are calculated for the numerical variables and then displayed as an interactive percentile plot.
    """

    def __init__(self, data):
        """Class initializer.

        Parameters
        ----------
        data : pd.Dataframe
        """
        self.data = data

    def get_plots(self, cols=None):
        """Returns a percentile plot and two tables which have the values for the extreme percentiles for the variables.

        Parameters
        ----------
        cols : `list`
            could be given a list of variables from `self.data` to be plotted.

        Returns
        -------
        plot : `dict`
            containing two percentile tables and a percentile plot for every numerical column.
        """
        # if cols:
        #     df = self.data[cols]
        # else:
        #     df = self.data
        df = self.data
        if cols is None:
            cols = list(df.columns)
        num_cols = [col for col in cols if col in get_num_cols(df)]
        if not num_cols:
            return "No numeric columns in data."
        if len(num_cols) > 1:
            import holoviews as hv

            plot = hv.DynamicMap(
                partial(percentile_plot, self.data), kdims=["col"]
            ).redim.values(col=num_cols)
        else:
            plot = percentile_plot(self.data, num_cols[0])
        return plot


class Normality:
    """Returns the plot of Shapiro feature rankings to check the normality of the variables."""

    def __init__(self, orientation="h"):
        """Class initializer.

        Parameters
        ----------
        orientation : `string`
            takes the arguments `h` and `v` which changes the orientation of the plot, horizontal or vertical.
        """
        self.orientation = orientation

    def fit(self, X, y=None):
        """Fits the variables in the Shapiro ranking algorithm and returns an updated object.

        Parameters
        ----------
        X : pd.DataFrame
            independent variables
        y : pd.Series
            dependent variable, optional argument

        Returns
        -------
        self : the updated object
        """
        self.X = X
        self.y = y
        self.ranks_ = self.rank(self.X)
        return self

    def rank(self, X):
        """Returns a numpy array of the feature rankings.

        The higher the rank the closer it is to being normally distributed.

        Parameters
        ----------
        X: pd.DataFrame
            Independent variables

        Returns
        -------
        An array of normality rankings of all columns in X
        """
        return np.array([shapiro(X[col])[0] for col in X.columns])

    def get_plot_data(self):
        """Returns the ranking array of features."""
        ranks = self.ranks_
        features = list(self.X.columns)
        normality_ranks = (
            pd.DataFrame({"features": features, "ranks": ranks})
            .sort_values(by="ranks")
            .set_index(["features"])
        )
        return normality_ranks

    def get_plot(self):
        """Returns the bar plot of the ranking array of features."""
        normality_ranks = self.get_plot_data()

        # orientation is horizontal
        if self.orientation == "h":
            plot = hvPlot(normality_ranks).bar(
                invert=True,
                title="Shapiro ranking of {} features".format(self.X.shape[1]),
            )
            return plot
        # orientation is vertical
        elif self.orientation == "v":
            plot = hvPlot(normality_ranks).bar(
                invert=False,
                title="Shapiro ranking of {} features".format(self.X.shape[1]),
            )
            return plot


class TargetDistribution:
    """Returns the distribution of the `target variable`."""

    def __init__(self, is_classification):
        """Assigns the model object depending on if the target is categorical or numerical.

        Parameters
        ----------
        is_classification : `bool`
            If the target is categorical then it takes `True` else `False`
        """
        self.is_classification = is_classification

    def fit(self, y):  # Needs to be updated
        """Returns the updated object after fitting the model in target.

        Parameters
        ----------
        y : pd.Series
            Target or Dependent Variable

        Returns
        -------
        self : updated object
        """
        self.y = y
        return self

    def get_plot(self):
        """Returns the plot with the target distribution.

        If `self.classification == True` then it's a barplot of value_counts else a histogram.
        """
        if self.is_classification:
            plot = hvPlot(self.y.value_counts()).bar(title="Target Distribution")
        else:
            plot = hvPlot(self.y).hist(title="Target Distribution")

        return plot
