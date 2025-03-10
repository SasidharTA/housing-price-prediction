import holoviews as hv
import logging
import numpy as np
import pandas as pd
from hvplot import hvPlot
from tigerml.core.scoring import SCORING_OPTIONS, compute_residual
from tigerml.core.utils import fail_gracefully

from .base import Evaluator

_LOGGER = logging.getLogger(__name__)


def create_scatter(x, y, x_label="predicted", y_label="residuals"):
    """Returns an interactive scatterplot for residuals from the model.

    Parameters
    ----------
    test_data: boolean, if True residual plot will be created from test_data.
    """
    plotter = hvPlot(pd.DataFrame({x_label: x, y_label: y}))
    if len(x) > 5000:
        plot_type = "hexbin"
    else:
        plot_type = "scatter"
    plot = plotter(x=x_label, y=y_label, kind=plot_type)
    return plot


def create_residuals_histogram(residuals, x_label="residuals"):
    """Returns an interactive histogram for residuals from the model.

    Parameters
    ----------
    test_data: boolean, if True histogram for residual plot will be created from test_data.
    """
    # Histogram showing the distribution of residuals with test data
    hist = (
        hvPlot(pd.DataFrame({x_label: residuals})).hist(
            x_label, width=150, invert=True, height=380, alpha=0.7
        )
    ).opts(xrotation=45)
    return hist


class RegressionEvaluation(Evaluator):
    """Regression evaluation class."""

    def __init__(
        self,
        model=None,
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        yhat_train=None,
        yhat_test=None,
        residual_train=None,
        residual_test=None,
    ):
        """
        Regression evaluation class.

        Parameters
        ----------
        model : a `Scikit-Learn` Regressor
            Should be an instance of a `regressor`.
        """
        super().__init__(model, x_train, y_train, x_test, y_test, yhat_train, yhat_test)
        self.metrics = SCORING_OPTIONS.regression.copy()
        self.plots = {
            "residual_plot": self.residuals_plot,
            "actual_v_predicted": self.prediction_error_plot,
        }

        super().remove_metric("Explained Variance")
        self.residual_train = residual_train
        self.residual_test = residual_test
        if residual_train is None:
            self.residual_train = compute_residual(y_train, yhat_train)
        if self.has_test and residual_test is None:
            self.residual_test = compute_residual(y_test, yhat_test)

    @fail_gracefully(_LOGGER)
    def residuals_plot(self):
        """Returns a scatter plot and distribution for the residuals of the model.

        A residual plot shows the residuals on the vertical axis and the
        independent variable on the horizontal axis.
        If the points are randomly dispersed around the horizontal axis, a linear
        regression model is appropriate for the data; otherwise, a non-linear
        model is more appropriate.
        """
        # Residual Plot for TRAINING DATA
        line = hv.HLine(0)
        line.opts(
            color="black",
            line_width=2.0,
        )

        train_scatter = create_scatter(
            self.yhat_train,
            self.residual_train,
        ).opts(title="Train Data")
        train_hist = create_residuals_histogram(residuals=self.residual_train)

        train_scatter = (train_scatter * line).opts(
            legend_position="top_left", width=380, height=380, xlabel="predicted"
        )
        train_plot = (train_scatter + train_hist).cols(2)

        if self.has_test:
            test_scatter = create_scatter(
                self.yhat_test,
                self.residual_test,
            ).opts(title="Test Data")
            test_hist = create_residuals_histogram(residuals=self.residual_test)
            test_scatter = (test_scatter * line).opts(
                legend_position="top_left", width=380, height=380, xlabel="predicted"
            )
            test_plot = (test_scatter + test_hist).cols(2)
            return train_plot + test_plot
        # Histogram showing the distribution of residuals with train data
        # scatters = test_scatter * train_scatter
        # hists = test_hist * train_hist
        return train_plot

    def _compute_best_fit(self):
        """Returns a dataframe containing the best fit line for the model."""
        if self.has_test:
            data = self.y_test
            pred_data = self.yhat_test
        else:
            data = self.y_train
            pred_data = self.yhat_train
        y_test = data[:, np.newaxis]
        slope, _, _, _ = np.linalg.lstsq(y_test, pred_data, rcond=-1)
        min_y = data.min()
        max_y = data.max()
        y = [min_y, max_y]
        best_fit = [min_y * slope[0], max_y * slope[0]]
        return pd.DataFrame({"actuals": y, "predicted": best_fit})

    def _create_identity_line(self):
        """Returns a dashed identity line for the predictin-error plot."""
        min_vals = []
        max_vals = []
        if self.has_test:
            min_vals.append(self.y_test.min())
            min_vals.append(self.yhat_test.min())
            max_vals.append(self.y_test.max())
            max_vals.append(self.yhat_test.max())
        else:
            min_vals.append(self.y_train.min())
            min_vals.append(self.yhat_train.min())
            max_vals.append(self.y_train.max())
            max_vals.append(self.yhat_train.max())
        min_val = min(min_vals)
        max_val = max(max_vals)
        identity = [min_val, max_val]
        # Identity Line
        id_line = hvPlot(
            pd.DataFrame({"actuals": identity, "predicted": identity})
        ).line(x="actuals", y="predicted", label="identity")
        id_line.opts(
            color="red",
            line_dash="dashed",
            line_width=2.0,
        )
        return id_line

    @fail_gracefully(_LOGGER)
    def prediction_error_plot(self):
        """Returns a scatter plot between actual and predicted values of the model.

        The prediction error visualizer plots the actual targets from the `dataset`
        against the predicted values generated by our model(s). This visualizer is
        used to detect noise or `heteroscedasticity` along a range of the target
        domain.
        """
        train_scatter = create_scatter(
            self.y_train, self.yhat_train, x_label="actuals", y_label="predicted"
        )
        train_identity_line = self._create_identity_line()
        train_plot = (train_scatter * train_identity_line).opts(
            legend_position="top_left",
            width=500,
            height=500,
        )
        plot = train_plot.opts(title="Train Data")
        # best_fit_line = self.create_best_fit_line()
        if self.has_test:
            test_scatter = create_scatter(
                self.y_test, self.yhat_test, x_label="actuals", y_label="predicted"
            )
            test_identity_line = self._create_identity_line()
            test_plot = (test_scatter * test_identity_line).opts(
                legend_position="top_left",
                width=500,
                height=500,
            )
            plot = (plot + test_plot.opts(title="Test Data")).cols(2)
        return plot

    @fail_gracefully(_LOGGER)
    def get_metrics(self, cutoff_value=None):
        """A dataframe containing all evaluation metrics."""
        metrics_dict = {}
        for metric in self.metrics.keys():
            metric_details = self.metrics[metric]
            func = metric_details["func"]
            default_params = {}
            if "default_params" in metric_details:
                default_params = metric_details["default_params"]
            metrics_dict[metric] = {}
            if self.has_train:
                params = []
                params.append(self.y_train)
                params.append(self.yhat_train)
                metrics_dict[metric]["train"] = round(
                    func(*params, **default_params), 4
                )
            if self.has_test:
                params = []
                params.append(self.y_test)
                params.append(self.yhat_test)
                metrics_dict[metric]["test"] = round(func(*params, **default_params), 4)
            # if 'mape' in label.lower() and metrics_dict[label] > 1:
            #     import pdb
            #     pdb.set_trace()
        dict_of_df = {k: pd.DataFrame([v]) for k, v in metrics_dict.items()}
        metrics_df = pd.concat(dict_of_df, axis=1)
        metrics_df.columns.set_names(["metric", "dataset"], inplace=True)
        return metrics_df

    @fail_gracefully(_LOGGER)
    def get_plots(self, cutoff_value=None):
        """Returns a dictionary of plots to be used for regression report."""
        report_dict = {}
        for plot_ in self.plots.keys():
            func = self.plots[plot_]
            if self.has_test:
                report_dict[plot_] = [func()]
            else:
                report_dict[plot_] = func()
        return report_dict


class RegressionComparisonMixin:
    """Regression comparison mixin class."""

    def perf_metrics(self):
        """Returns a HTML table for the regression metrics for all the models given as list input.

        Returns
        -------
        Performance metrices table: HTMLTable
        """
        self.performance_metrics = pd.DataFrame()
        for model_name in self.reporters:
            current_metrics = self.reporters[model_name].evaluator.get_metrics()
            current_metrics.index = [model_name]
            self.performance_metrics = pd.concat(
                [self.performance_metrics, current_metrics], axis=0
            )
        self.performance_metrics.columns = self.performance_metrics.columns.droplevel(
            level=1
        )  # no train test
        from tigerml.core.reports.html import HTMLTable, preset_styles

        table = HTMLTable(self.performance_metrics)
        bad_metrics = ["MAPE", "WMAPE", "MAE", "RMSE"]
        table.apply_conditional_format(
            cols=[
                x
                for x in self.performance_metrics.columns
                if all([col not in x for col in bad_metrics])
            ],
            style=preset_styles.more_is_good_2colors,
        )
        table.apply_conditional_format(
            cols=[
                x
                for x in self.performance_metrics.columns
                if any([col in x for col in bad_metrics])
            ],
            style=preset_styles.less_is_good_2colors,
        )

        return table

    def pred_error(self):
        """Returns a dictionary of plots of prediction error for multiple model input which can be used for comparison.

        Returns
        -------
        plots dictionary: dict
        """
        scatters = None
        best_fits = None
        if len(self.reporters) > 4:
            col_length = 3
            plot_size = 400
        else:
            col_length = 2
            plot_size = 400
        for model_name in self.reporters:
            if best_fits is None:
                best_fits = self.reporters[model_name].evaluator._create_identity_line()
            current_plot = (
                self.reporters[model_name]
                .evaluator.prediction_error_plot()
                .opts(title=model_name)
            ).opts(
                title=model_name, width=plot_size, height=plot_size, show_legend=False
            )
            current_best_fit = self.reporters[model_name].evaluator._compute_best_fit()
            best_fit_line = hvPlot(current_best_fit).line(
                x="actuals", y="predicted", label=f"{model_name}"
            )
            best_fits *= best_fit_line
            if scatters is None:
                scatters = current_plot
            else:
                scatters = scatters + current_plot
        scatters = scatters.cols(col_length)
        return {"best_fit": best_fits, "actual_vs_predicted": [scatters]}

    def residual_plot(self):
        """Returns a dictionary of plots of residuals for multiple model input which can be used for comparison.

        Returns
        -------
        plots dictionary: dict
        """
        scatters = None
        if len(self.reporters) > 4:
            col_length = 3 * 2
        else:
            col_length = 2 * 2
        for model_name in self.reporters:
            current_plot = self.reporters[model_name].evaluator.residuals_plot()
            current_plot[0].opts(title=model_name)
            if scatters is None:
                scatters = current_plot
            else:
                scatters = scatters + current_plot
        scatters = scatters.cols(col_length)
        return {"predicted_vs_residuals": [scatters]}

    def residual_distribution(self):
        """Returns residual distribution for multiple models input as an interactive plot.

        Returns
        -------
        Residual distribution plot: hvPlot
        """
        dataset = pd.DataFrame()
        for model_name in self.reporters:
            dataset = pd.concat(
                [
                    dataset,
                    pd.Series(self.reporters[model_name].evaluator.residual_train),
                ],
                axis=1,
            )
        dataset.columns = self.reporters.keys()
        residual_distribution = dataset.hvplot.kde(
            alpha=0.7,
            ylabel="density",
            xlabel="residuals",
            title="Residual Distribution",
            legend="top_right",
        )
        return residual_distribution

    def get_performance_report(self):
        """Return a consolidate dictionary contains regression specific comparative matrices values and different.

        Performance plots for all the input models.

        Returns
        -------
        Models performance plots: dict
        """
        perf_dict = {}
        perf_dict["performance_metrics"] = [self.perf_metrics()]
        perf_dict["prediction_error"] = self.pred_error()
        perf_dict["residual_plot"] = self.residual_plot()
        perf_dict["residual_distribution"] = self.residual_distribution()
        return perf_dict
