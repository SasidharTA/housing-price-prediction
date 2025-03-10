"""Description: Evaluation Base class."""

import numpy as np
from tigerml.core.scoring import TEST_PREFIX, TRAIN_PREFIX
from tigerml.core.utils.modeling import is_fitted


class Evaluator:
    """Evaluator class."""

    def __init__(
        self,
        model=None,
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        yhat_train=None,
        yhat_test=None,
        display_labels=None,
    ):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.yhat_train = yhat_train
        self.yhat_test = yhat_test
        self.datasets = []
        self.display_labels = display_labels
        self.datasets.append(TRAIN_PREFIX)
        if self.y_test is not None:
            self.datasets.append(TEST_PREFIX)

    @property
    def has_train(self):
        """Returns train prefix for Evaluator class."""
        return TRAIN_PREFIX in self.datasets

    @property
    def has_test(self):
        """Retuurns test prefix for Evaluator class."""
        return TEST_PREFIX in self.datasets

    def remove_metric(self, metric_name):
        """Thie method is used to remove the metric.

        Parameters
        ----------
            metric_name : str
                Metric name
        """
        self.metrics.pop(metric_name)

    def remove_plot(self, plot_name):
        """Thie method is used to remove the plot.

        Parameters
        ----------
            plot_name : str
                plot name
        """
        self.plots.pop(plot_name)

    def add_metric(
        self,
        metric_name,
        metric_func,
        more_is_better,
        display_format=None,
        default_params={},
    ):
        """Thie method is used to add custom metric.

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
        def adjusted_r2(y, yhat, idv):
            from sklearn.metrics import r2_score
            r2 = r2_score(y, yhat)
            adjusted_r_squared = 1 - (1 - r2) * (len(y) - 1) / (len(y) - idv - 1)
            return adjusted_r_squared


        self.add_metric(
            "Adj R^2", adjusted_r2, more_is_better=True, default_params={"idv": 13}
        )
        """

        self.metrics[metric_name] = {
            "string": metric_name,
            "func": metric_func,
            "more_is_better": more_is_better,
            "format": display_format,
            "default_params": default_params,
        }

    def add_plot(self, plot_name, plot_func):
        """Thie method is used to add custom metric.

        Parameters
        ----------
        plot_name : str
            plot name name
        plot_func : func
            function to get the plot


        Examples
        --------
        from tigerml.model_eval.plotters.evaluation.regression import create_scatter
        def plot_func():
            train_plot = create_scatter(regOpt1.y_train, regOpt1.yhat_train, x_label="y train", y_label="yhat train")
            test_plot = create_scatter(regOpt1.y_test, regOpt1.yhat_test, x_label="y test", y_label="yhat test")
            return train_plot + test_plot


        self.add_plot("y vs y hat", plot_func)
        """
        self.plots[plot_name] = plot_func

    def fit(self, X, y):
        """
        Fits the model.

        Parameters
        ----------
        X : X_train, pd.DataFrame of shape `n x m`
            A matrix of n instances with m features
        y : y_train, pd.Series of length `n`
            A series of target values

        Returns
        -------
        self : `ResidualsPlot`
            The visualizer instance
        """
        # Train Data
        self.X_train = X
        if type(y).__module__.startswith("pandas"):
            self.y_train = y.values.ravel()
        else:
            self.y_train = y

        if self.process_train:
            if not is_fitted(self.model, X.iloc[0].values.ravel().reshape(1, -1)):
                self.model.fit(self.X_train, np.ravel(self.y_train))
            self.train_pred = self.model.predict(self.X_train)
        else:
            self.train_pred = self.y_hat_train
        self.datasets.append(TRAIN_PREFIX)
        return self

    def score(self, X, y=None):
        """Returns the score of underlying estimator, usually the R-squared value.

        Parameters
        ----------
        X : X_test, pd.DataFrame
            to test for the model performance and get the predictions
        y : y_test, pd.Series
            to compare with the predicted values to finally compute the score.

        Returns
        -------
        score : `float`
            The score of the underlying estimator, usually the R-squared score
            for regression estimators.
        """

        self.X_test = X
        if type(y).__module__.startswith("pandas"):
            self.y_test = y.values.ravel()
        else:
            self.y_test = y
        if self.process_test:
            self.test_pred = self.model.predict(self.X_test)
            self._score = self.model.score(self.X_test, y=self.y_test)
        else:
            self.test_pred = self.y_hat_test
            self._score = None
        # if self.test_pred.ndim > 1:
        #   self.test_pred = self.test_pred[:, 0]
        if self.generate_test_report:
            self.datasets.append(TEST_PREFIX)
        return self._score
