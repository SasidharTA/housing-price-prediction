import gc
import logging
from tigerml.core.plots import save_plot as sp
from tigerml.core.reports import create_report
from tigerml.core.utils import (
    append_file_to_path,
    fail_gracefully,
    measure_time,
    time_now_readable,
)

_LOGGER = logging.getLogger(__name__)


class FeatureInteractionsMixin:
    """Feature Interactions Mixin class."""

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def correlation_heatmap(
        self,
        x_vars=None,
        y_vars=None,
        data=None,
        save_plot=False,
        file_path="correlation_heatmap.html",
    ):
        """Build correlation heatmap between the numeric variables.

        Parameters
        ----------
        x_vars: list, default=[]
            List of all columns  you want to keep on x axis. By default all are used.
        y_vars: list, default=[]
            List of all columns  you want to keep on y axis. By default all are used.
        data: pd.DataFrame, default = None
            If you want to use any other dataset explicitly rather then through self attribute.
        save_plot: bool, default= False
            Option to save the result, set it to True for saving.
        file_path: str, default= "correlation_heatmap.html"
            Pass a complete file path with extension.
        """
        from . import CorrelationHeatmap

        if data is None:
            data = self.data
        heatmap = CorrelationHeatmap(data).get_plot(x_vars, y_vars)
        # from tigerml.core.plots import autosize_plot
        # heatmap = autosize_plot(heatmap)
        if save_plot:
            sp(heatmap, file_path)
        return heatmap

    @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def correlation_table(self, x_vars=None, y_vars=None):
        """Return the pairwise Pearson correlation coefficient between the numeric variables for self.data.

        This considers only variables with non-zero standard deviation.

        Parameters
        ----------
        x_vars : list, default=None
            list of columns in the dataframe for analysis. By default all are used.
        y_vars : list, default=None
            By default definition in class are used.
        """
        if y_vars is None:
            y_vars = self.y_cols
        return FeatureInteractionsMixin.correlation_table_for_data(
            self.data, x_vars, y_vars
        )

    @staticmethod
    def correlation_table_for_data(data, x_vars=None, y_vars=None):
        """Return the pairwise Pearson correlation coefficient between the numeric variables for any given data.

        This considers only variables with non-zero standard deviation.

        Parameters
        ----------
        data : Pandas Dataframe
        x_vars : list, default=None
        list of columns in the dataframe for analysis. By default all are used.
        y_vars : list, default=None
        By default definition in class are used.
        """
        from . import CorrelationTable

        return CorrelationTable(data).get_plot(x_vars, y_vars)

    # @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def bivariate_plots(
        self,
        x_vars=None,
        y_vars=None,
        abs_corr_thresold=0,
        top_n=None,
        file_path=None,
        return_plots=True,
        save_plots=False,
        return_dict=False,
    ):  # noqa
        """Build bivariate plots for all x & y columns.

        The plots are compiled as PDF into working directory.

        Parameters
        ----------
        x_vars : list, default=None
            list of x_columns in the dataframe for analysis. By default all are used.
        y_vars : list, default=None
            By default definition in class are used.
        return_plots : bool, default=True
            If True, interactive hv plots are returned.
        save_plots : bool, default=True
            If True, plots are saved to a report.
        file_path : str, default="percentile_plots.pdf"
            Path for the file to be saved. Applicable when save_plots=True.

        Returns
        -------
        Scatter Plot     - When both X & Y are continuous
        Grouped Bar Plot - When both X & Y are categorical/ binary
        Box Plot         - When one of X & Y is continuous
                           and the other is categorical
        Butterfly Plot   - When one of X & Y is binary and
                           the other is categorical
        """
        # top_n = None
        # if not x_vars and not y_vars and len(self.data.columns) >= 10:
        #     top_n = 100
        from . import JointPlot

        if save_plots:
            return_dict = True

        plots = JointPlot(self.data).get_plots(
            x_vars,
            y_vars,
            top_n=top_n,
            abs_corr_thresold=abs_corr_thresold,
            return_dict=return_dict,
        )
        if save_plots:
            create_report(
                plots,
                name="bivariate_plots",
                path=file_path,
            )
            # sp(plots, file_path)
        if return_plots:
            return plots
        else:
            del plots
            gc.collect()

    # @fail_gracefully(_LOGGER)
    def covariance_heatmap(self, cols=None):
        """Build covariance heatmap between the numeric variables.

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        """
        from . import CovarianceHeatmap

        if cols:
            cols = [cols] if isinstance(cols, str) else cols
            data = self.data[
                [x for x in self.get_numeric_columns(self.data) if x in cols]
            ]
        else:
            data = self.data[self.get_numeric_columns(self.data)]
        plot = CovarianceHeatmap().fit(data).get_plot()
        # del data
        gc.collect()
        return plot

    # @fail_gracefully(_LOGGER)
    @measure_time(_LOGGER)
    def feature_interactions(self, cols=None, save_as=None, save_path=""):
        """Feature interactions report.

        Compiles outputs from correlation_table, correlation_heatmap,
        covariance_heatmap and bivariate_plots as a report.

        Parameters
        ----------
        cols : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
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
        >>> an = EDAReport(df, y = 'Survived', y_continuous = False)
        >>> an.feature_interactions()
        """
        self._preprocess_data(self._current_y)
        self.feature_interactions_report = {}
        self.feature_interactions_report["correlation_table"] = [
            self.correlation_table(x_vars=cols, y_vars=cols)
        ]
        self.feature_interactions_report["correlation_heatmap"] = [
            self.correlation_heatmap(x_vars=cols, y_vars=cols)
        ]
        self.feature_interactions_report["covariance_heatmap"] = [
            self.covariance_heatmap(cols=cols)
        ]
        top_n = None
        if len(self.data.columns) > 7:  # If total possibilities > 50
            top_n = 50
        plot_dict = self.bivariate_plots(
            x_vars=cols, y_vars=cols, return_dict=True, top_n=top_n
        )
        key = (
            f"bivariate_plots (Top {top_n} Correlations)"
            if top_n
            else "bivariate_plots"
        )
        self.feature_interactions_report[key] = plot_dict
        if save_as:
            default_report_name = "feature_interactions_report_at_{}".format(
                time_now_readable()
            )
            save_path = append_file_to_path(save_path, default_report_name + save_as)
            create_report(
                self.feature_interactions_report,
                path=save_path,
                format=save_as,
            )
        return self.feature_interactions_report
