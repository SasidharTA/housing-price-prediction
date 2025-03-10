"""Description: Multiple Model Comparison."""

import holoviews as hv
import logging
import pandas as pd
from datetime import datetime
from hvplot import hvPlot
from itertools import combinations
from tigerml.core.reports import create_report
from tigerml.core.scoring import SCORING_OPTIONS
from tigerml.core.utils import fail_gracefully

hv.extension("bokeh", "matplotlib")
hv.output(widget_location="bottom")

_LOGGER = logging.getLogger(__name__)


def verify_data_columns(cols, data):
    """Verify if all the required columns are present in data."""
    missing_cols = list(set(cols) - set(data.columns))
    if len(missing_cols):
        raise ValueError("These columns are not in 'data':", missing_cols)


def all_possible_combinations(x: list, min_n: int = 0, max_n: int = None):
    """Create all possible combinations from a list of elements.

    Parameters
    ----------
    x : list
        List of elements

    min_n : int, default=0
        Minimum number of elements to combine;
        should be a positive integer not exceeding `len(x)`.

    max_n : int, default=None
        Maximum number of elements to combine;
        if None, takes value of `len(x)`;
        should be a positive integer not exceeding `len(x)`
        with min_n <= max_n .
    """
    if max_n is None:
        max_n = len(x)
    if (min_n > len(x)) or (max_n > len(x)):
        raise ValueError("'min_n' or 'max_n' cannot be greater than length of 'x'")
    if (min_n < 0) or (max_n < 0):
        raise ValueError("'min_n' or 'max_n' cannot be less than 0")
    if min_n > max_n:
        raise ValueError("'min_n' cannot be greater than 'max_n'")
    list_combns = []
    for r in range(min_n, max_n + 1):
        list_combns.extend(list(combinations(x, r)))
    return list_combns


def simplify_dataframe(df: pd.DataFrame, col_sep: str = " - "):
    """Simplify DataFrame having MultiIndex type index and/or columns."""
    df_new = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df_new.columns = [col_sep.join(col).strip() for col in df_new.columns]
    if not isinstance(df.index, pd.RangeIndex):
        df_new.reset_index(inplace=True)
    df_new.columns.names = [None]
    df_new.index.names = [None]
    return df_new


def get_summary_df(data: pd.DataFrame, percentiles: list = None):
    """Get basic statistics summary of a DataFrame."""
    if percentiles is None:
        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    summary_df = data.describe(percentiles=percentiles).T
    summary_df["count"] = summary_df["count"].astype(int)
    summary_df.index.name = "Metric"
    summary_df.columns = summary_df.columns.str.title()
    return summary_df


def density_with_box_plot(data: pd.DataFrame, x: str, by: str = None):
    """Create a combination of density and box plot.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data for plot.

    x : str
        Column of `data` to use as x-axis.

    by : str, default=None
        Column in the `data` to group by.
    """
    plotter = hvPlot(data, cmap="Category10")
    # Density plot
    density_plot = plotter.density(x, by=by, height=250, xlabel="", legend="top_right")
    # Box plot
    box_plot = plotter.box(x, by=by, invert=True, height=100, xlabel="")
    if by is not None:
        box_plot.opts(box_color=by, show_legend=False, height=125)
    combined_plot = (density_plot + box_plot).cols(1)
    return combined_plot


def scatter_plot(
    data: pd.DataFrame, x: str, y: str, hover_cols: list = None, by: str = None
):
    """Create a scatter plot.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data for plot.

    x : str
        Column of `data` to use as x-axis.

    y : str
        Column of `data` to use as y-axis.

    hover_cols : list, list of strings, default=None
        List of columns or index names to show on hover
        in addition to x and y.

    by : str, default=None
        Column in the `data` to group by.
    """
    if hover_cols is None:
        hover_cols = []
    plotter = hvPlot(data, cmap="Category10")
    if by is None:
        plot = plotter.scatter(x, y, hover_cols=hover_cols)
    else:
        plot = plotter.scatter(x, y, hover_cols=hover_cols, by=by, legend="top_right")
    return plot


def create_heatmap(
    data: pd.DataFrame, x: str, y: str, C: str, colormap: str = None, title: str = ""
):
    """Create a heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data for plot.

    x : str
        Column or index name of `data` to use as x-axis.

    y : str
        Column or index name of `data` to use as y-axis.

    C : str
        Column in the `data` to use as values for color intensity in heatmap.

    colormap : str, default=None
        Name of the colormap supported by `hvplot` to use for the plot.

    title : str, default=""
        Title string for the plot.
    """
    plotter = hvPlot(data, cmap=colormap)
    plot = plotter.heatmap(x=x, y=y, C=C, rot=90, title=title)
    return plot


class MultiModelComparisonRegression:
    """Comparison of multiple Regression models by groups.

    Compare multiple model metrics using actual vs predicted values of
    multiple Regression models segmented by one or more grouping parameters.

    Default metrics are:
        - MAPE: mean absolute percentage error
        - WMAPE: weighted mean absolute percentage error
        - MAE: mean absolute error
        - RMSE: root mean squared error

    Use `add_metric` method to add any custom metric for comparison.
    Use `remove_metric` method to remove any metric from the default metrics
    or the custom metrics added already.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the grouping columns along with
        actual and predicted values.

    group_cols : list, list of strings
        List of column names to be used as grouping parameters.

    y_true_col : str
        Name of the column containing actual values.

    y_pred_col : str
        Name of the column containing predicted values.

    y_base_col : str, default = None
        Name of the column containing baseline prediction values.

    Examples
    --------
    >>> # Import the required modules
    >>> import pandas as pd
    >>> import numpy as np
    >>> from tigerml.model_eval import MultiModelComparisonRegression

    >>> # Load the data
    >>> # Download the csv file from following Google Drive location
    >>> # https://drive.google.com/file/d/1ZQhtKQYmvOiRm2y33edOpjzipPFx4JYb
    >>> results_file = "../data/store_item_daily_predictions.csv"
    >>> results_df = pd.read_csv(results_file, parse_dates=['date'])
    >>> results_df.info()
    >>> # Make sure all the grouping columns are of str type
    >>> # as it will increase visibility of all values on heatmap axis
    >>> results_df['item'] = results_df['item'].astype(str)
    >>> results_df.head()

    >>> # Initialize the model comparison object and get report (w/o baseline)
    >>> mmcr = MultiModelComparisonRegression(
    ...     data=results_df,
    ...     group_cols=['store', 'item'],
    ...     y_true_col='actuals',
    ...     y_pred_col='predicted')
    >>> mmcr.get_report()

    >>> # Create a dummy baseline predictions column
    >>> np.random.seed(42)
    >>> noise = np.random.choice(range(10), size=results_df.shape[0])
    >>> baseline = results_df[['actuals', 'predicted']].mean(axis=1) + noise
    >>> results_df['baseline'] = baseline
    >>> results_df.head()

    >>> # Compare against baseline predictions and get report
    >>> mmcr2 = MultiModelComparisonRegression(
    ...     data=results_df,
    ...     group_cols=['store', 'item'],
    ...     y_true_col='actuals',
    ...     y_pred_col='predicted',
    ...     y_base_col='baseline')
    >>> mmcr2.get_report()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        group_cols: list,
        y_true_col: str,
        y_pred_col: str,
        y_base_col: str = None,
    ):
        reqd_cols = group_cols + [y_true_col, y_pred_col]
        if y_base_col is not None:
            reqd_cols.append(y_base_col)
        verify_data_columns(reqd_cols, data)

        self.data = data
        self.group_cols = group_cols
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.y_base_col = y_base_col

        self.has_baseline = True if self.y_base_col is not None else False

        default_metrics = ["MAPE", "WMAPE", "MAE", "RMSE"]
        self.metrics = {}
        for metric in default_metrics:
            self.metrics[metric] = SCORING_OPTIONS.regression.get(metric)

        self.metrics_df = None
        self.element_tree = {}

    def _check_if_metrics_df_exists(self):
        if self.metrics_df is None:
            raise Exception(
                "'metrics_df' has not been computed yet. "
                "Use .compute_metrics_all_groups() method first."
            )

    def add_metric(
        self,
        metric_name,
        metric_func,
        more_is_better,
        display_format=None,
        default_params={},
    ):
        """Add custom metric for multiple model comparison.

        Parameters
        ----------
        metric_name : str
            Metric name

        metric_func : func
            function to claculate metrics

        more_is_better : bool, default True
            metrics value direction

        display_format : table_styles, default None
            metric display format

        default_params : dict, default {}
            parameters help to calculate the metric

        Examples
        --------
        >>> def adjusted_r2(y, yhat, idv):
        ...     from sklearn.metrics import r2_score
        ...     r2 = r2_score(y, yhat)
        ...     n = len(y)
        ...     adjusted_r_squared = 1 - (1 - r2) * (n - 1) / (n - idv - 1)
        ...     return adjusted_r_squared

        >>> self.add_metric(
        ...     "Adj R^2", adjusted_r2, more_is_better=True,
        ...     default_params={"idv": 13}
        ... )
        """
        self.metrics[metric_name] = {
            "string": metric_name,
            "func": metric_func,
            "more_is_better": more_is_better,
            "format": display_format,
            "default_params": default_params,
        }

    def remove_metric(self, metric_name):
        """Remove the selected metric.

        Parameters
        ----------
        metric_name : str
            Metric name
        """
        self.metrics.pop(metric_name)

    def compute_metrics(self, actual: str, predicted: str, group_by: list):
        """Compute metrics by groups.

        Parameters
        ----------
        actual : str
            Column name containing the actual values.

        predicted : str
            Column name containing the predicted values.

        group_by : list, list of strings
            List of column names to be used as grouping parameters.
        """
        metrics = self.metrics

        def _func(x):
            metrics_dict = {}
            for metric_name, metric_details in metrics.items():
                func = metric_details["func"]
                if "default_params" in metric_details:
                    default_params = metric_details["default_params"]
                else:
                    default_params = {}
                params = [x[actual], x[predicted]]
                metrics_dict[metric_name] = func(*params, **default_params)
            return pd.Series(metrics_dict)

        if group_by:
            metrics_df = self.data.groupby(group_by).apply(_func)
        else:
            metrics_df = _func(self.data).to_frame().T

        _info = "Computed metrics {} for '{}' vs '{}' grouped by columns {}".format(
            metrics_df.columns.tolist(), actual, predicted, group_by
        )
        _info += " --- Shape: {}".format(metrics_df.shape)
        _LOGGER.info(_info)
        # print(metrics_df.head())

        return metrics_df

    def compute_metrics_all_groups(self):
        """Compute metrics for all group combinations.

        The function creates all possible groups of 1 or 2 columns from
        `group_cols` and computes metrics at group level for all group
        combinations.
        """
        self.metrics_df = {}
        group_combns = all_possible_combinations(
            self.group_cols, min_n=1, max_n=min(2, len(self.group_cols))
        )
        for group_by_cols in group_combns:
            # Metrics for current predictions
            df_current = self.compute_metrics(
                actual=self.y_true_col,
                predicted=self.y_pred_col,
                group_by=list(group_by_cols),
            )
            if not self.has_baseline:
                self.metrics_df[group_by_cols] = df_current.reset_index()
            else:
                # Metrics for baseline predictions
                df_baseline = self.compute_metrics(
                    actual=self.y_true_col,
                    predicted=self.y_base_col,
                    group_by=list(group_by_cols),
                )
                df = pd.concat(
                    {"Current": df_current, "Baseline": df_baseline},
                    axis=0,
                    names=["Prediction Type"],
                )
                self.metrics_df[group_by_cols] = df.reset_index()
            # print(self.metrics_df[group_by_cols].head())

    @fail_gracefully(_LOGGER)
    def get_metrics_table(self, group_by):
        """Get simplified metrics table."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        if self.has_baseline:
            table = df.pivot_table(index=list(group_by), columns=["Prediction Type"])
            table = simplify_dataframe(table, col_sep=" - ")
        else:
            table = simplify_dataframe(df)
        return table

    @fail_gracefully(_LOGGER)
    def get_metrics_summary(self, group_by):
        """Summary statistics of metrics at overall level."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        if self.has_baseline:
            by = "Prediction Type"
            metric_summary = df.groupby(by).apply(lambda x: get_summary_df(x))
        else:
            metric_summary = get_summary_df(df)
        metric_summary = simplify_dataframe(metric_summary)
        return metric_summary

    @fail_gracefully(_LOGGER)
    def plot_metrics_density(self, group_by):
        """Create all metrics density plots."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        plots = {}
        for metric in self.metrics:
            by = "Prediction Type" if self.has_baseline else None
            plots[metric] = density_with_box_plot(df, x=metric, by=by)
        return plots

    @fail_gracefully(_LOGGER)
    def plot_metrics_scatter(self, group_by):
        """Create all metric vs metric scatter plots."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        # select_combns = None
        select_combns = [("MAPE", "MAE"), ("WMAPE", "RMSE")]
        plots = {}
        for m1, m2 in combinations(self.metrics, 2):
            if select_combns is not None:
                if ((m1, m2) in select_combns) or ((m2, m1) in select_combns):
                    pass
                else:
                    continue
            by = "Prediction Type" if self.has_baseline else None
            plot = scatter_plot(df, m1, m2, hover_cols=list(group_by), by=by)
            plots[f"{m1} vs {m2}"] = [plot]
        return plots

    @fail_gracefully(_LOGGER)
    def plot_bivariate_heatmap(self, group_by):
        """Create all bi-variate Heatmap plots."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        metrics = list(self.metrics.keys())
        if self.has_baseline:
            df_current = df[df["Prediction Type"] == "Current"].set_index(
                list(group_by)
            )[metrics]
            df_baseline = df[df["Prediction Type"] == "Baseline"].set_index(
                list(group_by)
            )[metrics]
            df = df_current - df_baseline
        else:
            df = df.set_index(list(group_by))[metrics]
        # Variables to use as axes in heatmap
        x, y = df.index.names
        n_levels = df.index.levshape
        if n_levels[0] > n_levels[1]:
            x, y = y, x
        # Create dict of plots for each metric
        plots = {}
        for metric in self.metrics:
            more_is_better = self.metrics[metric]["more_is_better"]
            if self.has_baseline:
                title = (
                    f"Distribution of differences in " f"{metric} (Current - Baseline)"
                )
                colormap = "RdYlGn" if more_is_better else "RdYlGn_r"
                plot = create_heatmap(
                    df, x=x, y=y, C=metric, colormap=colormap, title=title
                )
                plot.opts(symmetric=True, colorbar=True)
            else:
                title = f"Distribution of {metric}"
                colormap = "Reds_r" if more_is_better else "Reds"
                plot = create_heatmap(
                    df, x=x, y=y, C=metric, colormap=colormap, title=title
                )
            plots[metric] = [plot]
        return plots

    def _create_common_elements(self, key, level):
        self.element_tree[level]["Metric Summary"] = self.get_metrics_summary(
            group_by=key
        )
        self.element_tree[level]["Metric Distribution"] = self.plot_metrics_density(
            group_by=key
        )
        self.element_tree[level]["Metric Table"] = self.get_metrics_table(group_by=key)
        self.element_tree[level]["Bi-Metric Scatter Plots"] = self.plot_metrics_scatter(
            group_by=key
        )

    def _create_1d_elements(self, key, level):
        pass

    def _create_2d_elements(self, key, level):
        self.element_tree[level]["Bi-variate Heatmap"] = self.plot_bivariate_heatmap(
            group_by=key
        )

    def _create_report_elements(self):
        self.element_tree.clear()
        for key in self.metrics_df.keys():
            level = "{} level".format(" x ".join(key))
            self.element_tree[level] = {}
            self._create_common_elements(key, level)
            if len(key) == 1:
                self._create_1d_elements(key, level)
            elif len(key) == 2:
                self._create_2d_elements(key, level)
            else:
                pass

    def _generate_report_name(self, with_timestamp=False):
        prefix = "MultiModelComparisonReport"
        name_parts = [prefix, "Regression"]
        if self.has_baseline:
            name_parts.append("with_Baseline")
        if with_timestamp:
            name_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_name = "--".join(name_parts)
        return report_name

    def get_report(self, file_path="", with_timestamp=False, format=".html", **kwargs):
        """Create consolidated report on Model Evaluation.

        Parameters
        ----------
        file_path : str, default=''
            location with filename where report to be saved. By default is auto generated from system timestamp and saved in working directory.
        with_timestamp : bool, default=False
            Adds an auto generated system timestamp to name of the report.
        format : str, default='.html'
            format of report to be generated. possible values '.xlsx', '.html'
        excel_params : dict
            Dictionary containing the following keys if the format is ".xlsx".
            If a key is not provided, it will take the default values.

                - have_plot : boolean; default False.
                    If True, keep the plots in image format in excel report.
                - n_rows : int; default 100.
                    Number of sample rows to keep for plot types containing all the records in data (for example, density plot, scatter plot etc.)
        """
        if self.metrics_df is None:
            self.compute_metrics_all_groups()
        self._create_report_elements()
        # report_element = {"Multiple Model Comparison - Regression": self.element_tree}
        report_element = self.element_tree
        if not file_path:
            file_path = self._generate_report_name(with_timestamp=with_timestamp)

        if format == ".xlsx":
            keys_to_combine = [
                (key, "Metric Distribution") for key in report_element.keys()
            ]

            from tigerml.core.utils import convert_to_tuples

            convert_to_tuples(keys_to_combine, report_element)

        create_report(report_element, name=file_path, format=format, **kwargs)


class MultiModelComparisonClassification:
    """Comparison of multiple Classification models by groups.

    Compare multiple model metrics using actual vs predicted values of
    multiple Classification models segmented by one or more grouping parameters.

    Default metrics are:
        - Accuracy
        - F1_Score
        - Precision
        - Recall
        - ROC AUC Score

    Use `add_metric` method to add any custom metric for comparison.
    Use `remove_metric` method to remove any metric from the default metrics
    or the custom metrics added already.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the grouping columns along with
        actual and predicted values.

    group_cols : list, list of strings
        List of column names to be used as grouping parameters.

    y_true_col : str
        Name of the column containing actual values.

    y_pred_col : str
        Name of the column containing predicted values.

    y_base_col : str, default = None
        Name of the column containing baseline prediction values.

    Examples
    --------
    >>> # Import the required modules
    >>> import pandas as pd
    >>> import numpy as np
    >>> from tigerml.model_eval import MultiModelComparisonClassification

    >>> # Load the data
    >>> # Download the csv file from following Google Drive location
    >>> # https://drive.google.com/file/d/1wkOBKKbEM7ZSn9hCktuPm3BEu1OKCl-P
    >>> results_file = "../data/Titanic_predictions.csv"
    >>> results_df = pd.read_csv(results_file)
    >>> results_df.info()
    >>> # Make sure all the grouping columns are of str type
    >>> # as it will increase visibility of all values on heatmap axis
    >>> results_df['Pclass'] = results_df['Pclass'].astype(str)
    >>> results_df['Sex'] = results_df['Sex'].astype(str)
    >>> results_df.head()

    >>> # Initialize the model comparison object and get report (w/o baseline)
    >>> mmcc = MultiModelComparisonClassification(
    ...     data=results_df,
    ...     group_cols=['Pclass', 'Sex'],
    ...     y_true_col='actuals',
    ...     y_pred_col='predicted')
    >>> mmcc.get_report()

    >>> # Create a dummy baseline predictions column
    >>> np.random.seed(42)
    >>> results_df['baseline'] = np.random.choice(range(2), size=results_df.shape[0])
    >>> results_df.head()

    >>> # Compare against baseline predictions and get report
    >>> mmcc2 = MultiModelComparisonClassification(
    ...     data=results_df,
    ...     group_cols=['Pclass', 'Sex'],
    ...     y_true_col='actuals',
    ...     y_pred_col='predicted',
    ...     y_base_col='baseline')
    >>> mmcc2.get_report()
    """

    # TODO: Update class to accept predicted probabilities and prediction thresholds
    # TODO: Bring all common functionalities under one base class for both classification and regression and then extend the base class for both

    def __init__(
        self,
        data: pd.DataFrame,
        group_cols: list,
        y_true_col: str,
        y_pred_col: str,
        y_base_col: str = None,
    ):
        reqd_cols = group_cols + [y_true_col, y_pred_col]
        if y_base_col is not None:
            reqd_cols.append(y_base_col)
        verify_data_columns(reqd_cols, data)

        self.data = data
        self.group_cols = group_cols
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.y_base_col = y_base_col

        if self.data[self.y_true_col].nunique() > 2:
            self.multi_class = True
        else:
            self.multi_class = False

        self.has_baseline = True if self.y_base_col is not None else False

        default_metrics = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
        self.metrics = {}
        for metric in default_metrics:
            if self.multi_class:
                self.metrics[metric] = SCORING_OPTIONS.multi_class.get(metric)
            else:
                self.metrics[metric] = SCORING_OPTIONS.classification.get(metric)

        self.metrics_df = None
        self.element_tree = {}

    def _check_if_metrics_df_exists(self):
        if self.metrics_df is None:
            raise Exception(
                "'metrics_df' has not been computed yet. "
                "Use .compute_metrics_all_groups() method first."
            )

    def add_metric(
        self,
        metric_name,
        metric_func,
        more_is_better,
        display_format=None,
        default_params={},
    ):
        """Add custom metric for multiple model comparison.

        Parameters
        ----------
        metric_name : str
            Metric name

        metric_func : func
            function to claculate metrics

        more_is_better : bool, default True
            metrics value direction

        display_format : table_styles, default None
            metric display format

        default_params : dict, default {}
            parameters help to calculate the metric

        Examples
        --------
        >>> def adjusted_r2(y, yhat, idv):
        ...     from sklearn.metrics import r2_score
        ...     r2 = r2_score(y, yhat)
        ...     n = len(y)
        ...     adjusted_r_squared = 1 - (1 - r2) * (n - 1) / (n - idv - 1)
        ...     return adjusted_r_squared

        >>> self.add_metric(
        ...     "Adj R^2", adjusted_r2, more_is_better=True,
        ...     default_params={"idv": 13}
        ... )
        """
        self.metrics[metric_name] = {
            "string": metric_name,
            "func": metric_func,
            "more_is_better": more_is_better,
            "format": display_format,
            "default_params": default_params,
        }

    def remove_metric(self, metric_name):
        """Remove the selected metric.

        Parameters
        ----------
        metric_name : str
            Metric name
        """
        self.metrics.pop(metric_name)

    def compute_metrics(self, actual: str, predicted: str, group_by: list):
        """Compute metrics by groups.

        Parameters
        ----------
        actual : str
            Column name containing the actual values.

        predicted : str
            Column name containing the predicted values.

        group_by : list, list of strings
            List of column names to be used as grouping parameters.
        """
        metrics = self.metrics

        def _func(x):
            metrics_dict = {}
            for metric_name, metric_details in metrics.items():
                func = metric_details["func"]
                if "default_params" in metric_details:
                    default_params = metric_details["default_params"]
                else:
                    default_params = {}
                params = [x[actual], x[predicted]]
                metrics_dict[metric_name] = func(*params, **default_params)
            return pd.Series(metrics_dict)

        if group_by:
            metrics_df = self.data.groupby(group_by).apply(_func)
        else:
            metrics_df = _func(self.data).to_frame().T

        _info = "Computed metrics {} for '{}' vs '{}' grouped by columns {}".format(
            metrics_df.columns.tolist(), actual, predicted, group_by
        )
        _info += " --- Shape: {}".format(metrics_df.shape)
        _LOGGER.info(_info)
        # print(metrics_df.head())

        return metrics_df

    def compute_metrics_all_groups(self):
        """Compute metrics for all group combinations.

        The function creates all possible groups of 1 or 2 columns from
        `group_cols` and computes metrics at group level for all group
        combinations.
        """
        self.metrics_df = {}
        group_combns = all_possible_combinations(
            self.group_cols, min_n=1, max_n=min(2, len(self.group_cols))
        )
        for group_by_cols in group_combns:
            # Metrics for current predictions
            df_current = self.compute_metrics(
                actual=self.y_true_col,
                predicted=self.y_pred_col,
                group_by=list(group_by_cols),
            )
            if not self.has_baseline:
                self.metrics_df[group_by_cols] = df_current.reset_index()
            else:
                # Metrics for baseline predictions
                df_baseline = self.compute_metrics(
                    actual=self.y_true_col,
                    predicted=self.y_base_col,
                    group_by=list(group_by_cols),
                )
                df = pd.concat(
                    {"Current": df_current, "Baseline": df_baseline},
                    axis=0,
                    names=["Prediction Type"],
                )
                self.metrics_df[group_by_cols] = df.reset_index()
            # print(self.metrics_df[group_by_cols].head())

    @fail_gracefully(_LOGGER)
    def get_metrics_table(self, group_by):
        """Get simplified metrics table."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        if self.has_baseline:
            table = df.pivot_table(index=list(group_by), columns=["Prediction Type"])
            table = simplify_dataframe(table, col_sep=" - ")
        else:
            table = simplify_dataframe(df)
        return table

    @fail_gracefully(_LOGGER)
    def get_metrics_summary(self, group_by):
        """Summary statistics of metrics at overall level."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        if self.has_baseline:
            by = "Prediction Type"
            metric_summary = df.groupby(by).apply(lambda x: get_summary_df(x))
        else:
            metric_summary = get_summary_df(df)
        metric_summary = simplify_dataframe(metric_summary)
        return metric_summary

    @fail_gracefully(_LOGGER)
    def plot_metrics_density(self, group_by):
        """Create all metrics density plots."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        plots = {}
        for metric in self.metrics:
            by = "Prediction Type" if self.has_baseline else None
            plots[metric] = density_with_box_plot(df, x=metric, by=by)
        return plots

    @fail_gracefully(_LOGGER)
    def plot_metrics_scatter(self, group_by):
        """Create all metric vs metric scatter plots."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        # select_combns = None
        select_combns = [("precision", "recall")]
        plots = {}
        for m1, m2 in combinations(self.metrics, 2):
            if select_combns is not None:
                if ((m1, m2) in select_combns) or ((m2, m1) in select_combns):
                    pass
                else:
                    continue
            by = "Prediction Type" if self.has_baseline else None
            plot = scatter_plot(df, m1, m2, hover_cols=list(group_by), by=by)
            plots[f"{m1} vs {m2}"] = [plot]
        return plots

    @fail_gracefully(_LOGGER)
    def plot_bivariate_heatmap(self, group_by):
        """Create all bi-variate Heatmap plots."""
        self._check_if_metrics_df_exists()
        df = self.metrics_df[group_by]
        metrics = list(self.metrics.keys())
        if self.has_baseline:
            df_current = df[df["Prediction Type"] == "Current"].set_index(
                list(group_by)
            )[metrics]
            df_baseline = df[df["Prediction Type"] == "Baseline"].set_index(
                list(group_by)
            )[metrics]
            df = df_current - df_baseline
        else:
            df = df.set_index(list(group_by))[metrics]
        # Variables to use as axes in heatmap
        x, y = df.index.names
        n_levels = df.index.levshape
        if n_levels[0] > n_levels[1]:
            x, y = y, x
        # Create dict of plots for each metric
        plots = {}
        for metric in self.metrics:
            more_is_better = self.metrics[metric]["more_is_better"]
            if self.has_baseline:
                title = (
                    f"Distribution of differences in " f"{metric} (Current - Baseline)"
                )
                colormap = "RdYlGn" if more_is_better else "RdYlGn_r"
                plot = create_heatmap(
                    df, x=x, y=y, C=metric, colormap=colormap, title=title
                )
                plot.opts(symmetric=True, colorbar=True)
            else:
                title = f"Distribution of {metric}"
                colormap = "Reds_r" if more_is_better else "Reds"
                plot = create_heatmap(
                    df, x=x, y=y, C=metric, colormap=colormap, title=title
                )
            plots[metric] = [plot]
        return plots

    def _create_common_elements(self, key, level):
        self.element_tree[level]["Metric Summary"] = self.get_metrics_summary(
            group_by=key
        )
        self.element_tree[level]["Metric Distribution"] = self.plot_metrics_density(
            group_by=key
        )
        self.element_tree[level]["Metric Table"] = self.get_metrics_table(group_by=key)
        self.element_tree[level]["Bi-Metric Scatter Plots"] = self.plot_metrics_scatter(
            group_by=key
        )

    def _create_1d_elements(self, key, level):
        pass

    def _create_2d_elements(self, key, level):
        self.element_tree[level]["Bi-variate Heatmap"] = self.plot_bivariate_heatmap(
            group_by=key
        )

    def _create_report_elements(self):
        self.element_tree.clear()
        for key in self.metrics_df.keys():
            level = "{} level".format(" x ".join(key))
            self.element_tree[level] = {}
            self._create_common_elements(key, level)
            if len(key) == 1:
                self._create_1d_elements(key, level)
            elif len(key) == 2:
                self._create_2d_elements(key, level)
            else:
                pass

    def _generate_report_name(self, with_timestamp=False):
        prefix = "MultiModelComparisonReport"
        name_parts = [prefix, "Classification"]
        if self.has_baseline:
            name_parts.append("with_Baseline")
        if with_timestamp:
            name_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_name = "--".join(name_parts)
        return report_name

    def get_report(self, file_path="", with_timestamp=False, format=".html", **kwargs):
        """Create consolidated report on Model Evaluation.

        Parameters
        ----------
        file_path : str, default=''
            location with filename where report to be saved. By default is auto generated from system timestamp and saved in working directory.
        with_timestamp : bool, default=False
            Adds an auto generated system timestamp to name of the report.
        format : str, default='.html'
            format of report to be generated. possible values '.xlsx', '.html'
        excel_params : dict
            Dictionary containing the following keys if the format is ".xlsx".
            If a key is not provided, it will take the default values.
            - have_plot : boolean; default False.
              If True, keep the plots in image format in excel report.
            - n_rows : int; default 100.
              Number of sample rows to keep for plot types containing all the records in data (for example, density plot, scatter plot etc.)
        """
        if self.metrics_df is None:
            self.compute_metrics_all_groups()
        self._create_report_elements()
        # report_element = {"Multiple Model Comparison - Regression": self.element_tree}
        report_element = self.element_tree
        if not file_path:
            file_path = self._generate_report_name(with_timestamp=with_timestamp)

        if format == ".xlsx":
            keys_to_combine = [
                (key, "Metric Distribution") for key in report_element.keys()
            ]

            from tigerml.core.utils import convert_to_tuples

            convert_to_tuples(keys_to_combine, report_element)

        create_report(report_element, name=file_path, format=format, **kwargs)
