"""Description: Model Report."""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from tigerml.core.dataframe.base import TAPipeline
from tigerml.core.reports import create_report
from tigerml.core.scoring import compute_residual

from .plotters.evaluation import ClassificationEvaluation, RegressionEvaluation
from .plotters.interpretation import (
    Algo,
    ModelInterpretation,
    get_shap_summary_plot,
)

algo_object = Algo()


def verify_y_type(obj):
    if obj is None:
        pass
    elif not (
        isinstance(obj, pd.DataFrame)
        or isinstance(obj, pd.Series)
        or isinstance(obj, np.ndarray)
    ):
        print(type(obj))
        raise TypeError("y should be pd.DataFrame / pd.Series / np.ndarray")


def verify_x_type(obj):
    if obj is None:
        pass
    elif not (isinstance(obj, pd.DataFrame) or isinstance(obj, np.ndarray)):
        raise TypeError("x should be pd.DataFrame / pd.Series / np.ndarray")


def verify_lengths(lhs, rhs):
    if not (rhs is None or lhs is None):
        if len(lhs) != len(rhs):
            raise TypeError("all datasets should be of the same length")


def set_x_type(obj, cols=None):
    if isinstance(obj, np.ndarray):
        if cols is None:
            cols = ["Feature_" + str(i) for i in range(obj.shape[1])]
        obj = pd.DataFrame(obj, columns=cols)
    return obj


def set_y_type(obj, multi_class=False):
    if isinstance(obj, pd.DataFrame):
        if multi_class:
            obj = obj.values
        else:
            obj = obj.iloc[:, 0].values

    elif isinstance(obj, pd.Series):
        obj = obj.values
    else:
        try:
            if multi_class:
                None
            else:
                obj = obj[:, 1]
        except Exception:
            pass
    if obj is not None:
        if not multi_class:
            obj = np.ravel(obj)
    return obj


def print_dict_tree(t, s=0, expand_def=False):
    for key in t:
        print(" " * s * 3 + ("├──" if key != list(t.keys())[-1] else "└──") + str(key))
        if type(t[key]) is dict:
            if expand_def:
                print_dict_tree(t[key], s + 1, expand_def)
            else:
                if not ("func" in t[key].keys()):
                    print_dict_tree(t[key], s + 1, expand_def)


def clean_dict(dict_to_clean):
    dict_ = dict_to_clean.copy()
    for key_ in dict_.keys():
        if type(dict_[key_]) is dict:
            if len(dict_[key_]) == 0:
                dict_to_clean.pop(key_)
            else:
                dict_[key_] = clean_dict(dict_to_clean)
    return dict_to_clean


class ModelReport:
    """Model report class."""

    metrics = {}
    plots = {}

    def __init__(
        self,
        algo: str,
        y_train: pd.Series,
        model=None,
        x_train: pd.DataFrame = None,
        yhat_train: pd.Series = None,
        x_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        yhat_test: pd.Series = None,
        refit: bool = False,
        display_labels: dict = None,
    ):
        self._init_assignments(
            algo,
            y_train,
            model,
            x_train,
            yhat_train,
            x_test,
            y_test,
            yhat_test,
            refit,
            display_labels,
        )
        self._validate_inputs()

        # Fit model
        self._fit_model()
        # compute yhats
        self._compute_yhat()
        # compute residual
        self._compute_residuals()

        if isinstance(model, Pipeline):
            model = TAPipeline(self.model)
        if isinstance(model, TAPipeline):
            if x_train is not None:
                self.x_train = set_x_type(self._apply_pipeline(model, self.x_train))
            if x_test is not None:
                self.x_test = set_x_type(self._apply_pipeline(model, self.x_test))
            self.model = model.get_step(-1)

        if algo_object.is_regression(self.algo):
            self.evaluator = RegressionEvaluation(
                model=self.model,
                x_train=self.x_train,
                y_train=self.y_train,
                x_test=self.x_test,
                y_test=self.y_test,
                yhat_train=self.yhat_train,
                yhat_test=self.yhat_test,
                residual_train=self.residual_train,
                residual_test=self.residual_test,
            )
        elif algo_object.is_classification(self.algo):
            self.evaluator = ClassificationEvaluation(
                model=self.model,
                x_train=self.x_train,
                y_train=self.y_train,
                x_test=self.x_test,
                y_test=self.y_test,
                yhat_train=self.yhat_train,
                yhat_test=self.yhat_test,
                multi_class=self.multi_class,
                display_labels=self.display_labels,
            )

        # Init explainer
        self.explainer = ModelInterpretation(
            model=self.model,
            algo=self.algo,
            process_train=True,
            process_test=False,
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            yhat_train=self.yhat_train,
            yhat_test=self.yhat_test,
            residual_train=self.residual_train,
            residual_test=self.residual_test,
            multi_class=self.multi_class,
            display_labels=self.display_labels,
        )

        self.performance_tree = {
            "metrics": self.evaluator.metrics,
            "plots": self.evaluator.plots,
        }

        self._build_full_element_tree()

    def _init_assignments(
        self,
        algo,
        y_train,
        model,
        x_train,
        yhat_train,
        x_test,
        y_test,
        yhat_test,
        refit,
        display_labels,
    ):
        # Disabling interpreation on test datasets
        self.interpret_test = False
        # Assignments
        self.algo = algo
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.yhat_train = yhat_train
        self.yhat_test = yhat_test
        self.refit = refit
        self.display_labels = display_labels
        self.multi_class = (
            type_of_target(self.y_train) == "multiclass"
            and self.algo == "classification"
        )
        self.residual_train = None
        self.residual_test = None
        # Set report option
        if self.model is not None:
            self.report_option = 1
        else:
            self.report_option = 2
            self.refit = False
        # Reset yhat & residual on refit
        if self.refit:
            self.yhat_train = None
            self.yhat_test = None
            self.residual_train = None
            self.residual_test = None
        # Set has_test
        if self.y_test is None:
            self.has_test = False
            self.interpret_test = False
        else:
            self.has_test = True

        # Disable when no test
        if not (self.has_test):
            self.x_test = None
            self.y_test = None
            self.yhat_test = None
            self.residual_test = None
            self.interpret_test = False

        # Creating dummies for yhat_train and yhat_test for multi_class classification
        if self.report_option == 2 and self.multi_class:
            if len(self.yhat_train.shape) == 1:
                self.yhat_train = np.array(pd.get_dummies(self.yhat_train))
            if self.has_test:
                if len(self.yhat_test.shape) == 1:
                    self.yhat_test = np.array(pd.get_dummies(self.yhat_test))

    def _apply_pipeline(self, pipeline, X):
        return pipeline.get_data_at_step(-1, X)

    def _validate_inputs(self):
        # Validate data types
        for y in [self.y_train, self.y_test, self.yhat_train, self.yhat_test]:
            verify_y_type(y)

        for x in [self.x_train, self.x_test]:
            verify_x_type(x)

        # Set datatypes
        self.x_train = set_x_type(self.x_train)
        self.x_test = set_x_type(self.x_test)

        self.y_train = set_y_type(self.y_train, multi_class=self.multi_class)
        self.y_test = set_y_type(self.y_test, multi_class=self.multi_class)
        self.yhat_train = set_y_type(self.yhat_train, multi_class=self.multi_class)
        self.yhat_test = set_y_type(self.yhat_test, multi_class=self.multi_class)

        # Validate array lengths
        for obj in [self.x_train, self.yhat_train]:
            if obj is not None:
                verify_lengths(self.y_train, obj)
        if self.has_test:
            for obj in [self.x_test, self.yhat_test]:
                if obj is not None:
                    verify_lengths(self.y_test, obj)

        # Refit validations
        if self.refit:
            if self.x_train is None:
                raise ValueError("x_train is mandatory with refit=True")
            if self.has_test and self.x_test is None:
                raise ValueError("x_test is mandatory with refit=True")

        if self.has_test and self.y_test is None:
            raise ValueError("y_test is mandatory with refit=True")

        if self.multi_class:
            if self.display_labels is None:
                self.display_labels = dict(
                    zip(set(self.y_train), [str(i) for i in set(self.y_train)])
                )
                # raise ValueError("display_labels is mandatory for multiclass classification")
            if len(self.display_labels) != len(set(self.y_train)):
                raise ValueError(
                    "display labels and unique y train should be of the same length"
                )

    def show_element_tree(self, expand_def=False):
        """Print element tree."""
        self._build_full_element_tree()
        print_dict_tree(self.element_tree, 0, expand_def)

    def add_metric(
        self,
        metric_name,
        metric_func,
        more_is_better=True,
        display_format=None,
        default_params={},
    ):
        """Add custom metric to.

        Parameters
        ----------
        metric_name : str
            Metric name
        metric_func : func
            function to calculate metrics
        more_is_better : bool, default True
            metrics value direction
        display_format : table_styles, default None
            metric display format
        default_params : dict, default {}
            parameters help to calculate the metric

        Examples
        --------
        >>> def adjusted_r2(y, yhat, idv):
        >>>     from sklearn.metrics import r2_score
        >>>     r2 = r2_score(y, yhat)
        >>>     adjusted_r_squared = 1 - (1 - r2) * (len(y) - 1) / (len(y) - idv - 1)
        >>>     return adjusted_r_squared

        >>> self.add_metric("Adj R^2", adjusted_r2, more_is_better=True, default_params={"idv":13})
        """
        self.evaluator.add_metric(
            metric_name, metric_func, more_is_better, display_format, default_params
        )

    def remove_metric(self, metric_name):
        """Remove metric from evaluation.

        Parameters
        ----------
        metric_name : str
            Metric name
        """
        self.evaluator.remove_metric(metric_name)

    def add_eval_plot(self, plot_name, plot_func):
        """Add custom plot to evaluation.

        Parameters
        ----------
        plot_name : str
            plot name name
        plot_func : func
            function to get the plot. hv plots are recommended for interactions.


        Examples
        --------
        >>> from tigerml.model_eval.plotters.evaluation.regression import create_scatter
        >>> def plot_func():
        >>>     train_plot = create_scatter(regOpt1.y_train, regOpt1.yhat_train, x_label="y train", y_label="yhat train")
        >>>     test_plot = create_scatter(regOpt1.y_test, regOpt1.yhat_test, x_label="y test", y_label="yhat test")
        >>>     return train_plot + test_plot

        >>> self.add_eval_plot("y vs y hat", plot_func)
        """
        self.evaluator.add_plot(plot_name, plot_func)

    def remove_eval_plot(self, plot_name):
        """Remove plot from evaluation.

        Parameters
        ----------
        plot_name : str
            plot name
        """
        self.evaluator.remove_plot(plot_name)

    def _build_full_element_tree(self):
        self.element_tree = {}
        self.element_tree["model_performance"] = {
            "metrics": self.evaluator.metrics,
            "plots": self.evaluator.plots,
        }
        if self.report_option == 1:
            self.element_tree["model_interpretation"] = self.explainer.interpretations

    def _fit_model(self):
        """
        Fit the model evaluator.

        Returns
        -------
        self : returns an instance of self.
        """
        if self.refit:
            try:
                self.model.fit(self.x_train, np.ravel(self.y_train))
            except NotFittedError:
                raise Exception("Unable to fit refit the model Fit failed")

    def _compute_yhat(self):
        if self.report_option == 1:
            try:
                if algo_object.is_regression(self.algo):
                    self.yhat_train = set_y_type(self.model.predict(self.x_train))
                    if self.has_test:
                        self.yhat_test = set_y_type(self.model.predict(self.x_test))
                elif self.multi_class:
                    self.yhat_train = set_y_type(
                        self.model.predict_proba(self.x_train), self.multi_class
                    )
                    if self.has_test:
                        self.yhat_test = set_y_type(
                            self.model.predict_proba(self.x_test), self.multi_class
                        )
                else:
                    if "predict_proba" in dir(self.model):
                        self.yhat_train = set_y_type(
                            self.model.predict_proba(self.x_train)[:, 1]
                        )
                        if self.has_test:
                            self.yhat_test = set_y_type(
                                self.model.predict_proba(self.x_test)[:, 1]
                            )
                    else:
                        self.yhat_train = set_y_type(self.model.predict(self.x_train))
                        if self.has_test:
                            self.yhat_test = set_y_type(self.model.predict(self.x_test))
            except Exception:
                raise Exception(
                    "Prediction failed: Either pass a fitted model or set refit=True"
                )

    def _compute_residuals(self):
        """
        Fit the model evaluator.

        Returns
        -------
        self : returns an instance of self.
        """
        if algo_object.is_regression(self.algo):
            self.residual_train = compute_residual(self.y_train, self.yhat_train)
            if self.has_test:
                self.residual_test = compute_residual(self.y_test, self.yhat_test)

    def get_coefs_table(self):
        """Returns a dataframe having coefficients values from the model for all the features.

        Returns
        -------
        dataframe: pd.DataFrame
        """
        return self.explainer.get_coefs_table()

    def get_feature_importances(self, plot=True, top_n_features=20):
        """Returns feature importance from the model as an interactive hvplot bar chart or as a dataframe having.

        importance values.

        Note: For the linear models, feature importance is computed by coeff * mean(x).

        Parameters
        ----------
        plot: bool, default=True
            flag if plot needs to be return or just importance values as dataframe
        top_n_features: int, default=20
            top N features need to consider out of all features

        Returns
        -------
        feature importance: hvplot object or pd.DataFrame
        """
        return self.explainer.get_feature_importances(
            X=self.explainer.x_train, plot=plot, n=top_n_features
        )

    # def get_interpretation_plots(self, include_shap=False,
    #                              errorbuckets_spec=None,
    #                              include_shap_test_error_analysis=False,
    #                              n_features=20):
    #     """Returns a dictionary of all plots or tables from model interpretation based on specified parameters.
    #
    #     Parameters
    #     ----------
    #     include_shap: bool, default=False
    #         flag if SHAP support need to include for interpretation, if False, several interpretation elements will not
    #         be a part of return
    #     errorbuckets_spec: dict, default=None
    #         specification related to prediction error (in regression) or probability cutoff (in classification) and
    #         top N columns value to be select from feature importance for error analysis
    #     include_shap_test_error_analysis: bool, default=False
    #         flag if error analysis need to perform on test data (subject to availability)
    #     n_features: int, default=20
    #         number of features to be select for error analysis based on feature importance
    #
    #     Returns
    #     -------
    #     dict of plots
    #     """
    #     return self.explainer.get_plots(include_shap=include_shap,
    #                                     errorbuckets_spec=errorbuckets_spec,
    #                                     include_shap_test_error_analysis=include_shap_test_error_analysis,
    #                                     n_features=n_features)
    #
    # def get_error_drivers(self):
    #     """Genrates bivariate plots b/w residuals and features specified.
    #
    #     Returns
    #     -------
    #     dict of hvplot plots
    #     """
    #     return self.explainer.get_error_drivers()
    #
    # def get_errorbucket_profiles(self):
    #     """Generate interactive distributions of different error buckets for all features.
    #
    #     Returns
    #     -------
    #     dict of hvplot layouts
    #     """
    #     return self.explainer.get_errorbucket_profiles()
    #
    # def get_shap_error_analysis(self, test_plot=True):
    #     """Shap distribution (summary) plots of data instances for each error buckets set through `set_errorbuckets_spec()`.
    #
    #     Parameters
    #     ----------
    #     test_plot: bool, default=True
    #         flag if plots need to include for test data (if passed)
    #
    #     Returns
    #     -------
    #     dict of shap summary plots
    #     """
    #     return self.explainer.get_shap_error_analysis(test_plot=test_plot)
    #
    # def set_errorbuckets_spec(self, errorbuckets_spec=None):
    #     """Set the appropriate specifications for error analysis based on the model type.
    #
    #     Parameters
    #     ----------
    #     errorbuckets_spec: dict
    #         specification related to prediction error (in regression) or probability cutoff (in classification) and
    #         top N columns value to be select from feature importance for error analysis
    #     """
    #     return self.explainer.set_errorbuckets_spec(errorbuckets_spec=errorbuckets_spec)
    #
    # def get_dependence_plots(self):
    #     """Create a SHAP dependence plot, colored by an interaction feature.
    #
    #     Plots the value of the feature on the x-axis and the SHAP value of the same feature on the y-axis.
    #     This shows how the model depends on the given feature, and is like a richer extenstion of the classical parital
    #     dependence plots. Vertical dispersion of the data points represents interaction effects.
    #
    #     Returns
    #     -------
    #     shap dependence plot
    #     """
    #     return self.explainer.get_dependence_plots()
    #
    # def shap_distribution(self):
    #     """ Return distribution of shapely values as an interactive shap summary plot for a given data samples.
    #
    #     Returns
    #     -------
    #     shap summary plot
    #     """
    #     return self.explainer.shap_distribution()
    #
    # def shap_feature_contributions(self):
    #     """Return feature importance as an interactive shap summary plot.
    #
    #     Returns
    #     -------
    #     shap summary plot
    #     """
    #     return self.explainer.shap_feature_contributions()

    def get_performance_report(self, cutoff_value=0.5):
        """
        First evaluate the model and then generate the performance matrices and plots for the same.

        Returns
        -------
        performance_report : dict
            a dictionary contains performance matrices values and plots data to be added in the final report.
        """
        # moved to init as metrics
        report_dict = dict()
        metrics = self.evaluator.get_metrics(cutoff_value=cutoff_value)
        if metrics.__class__.__name__ == "DataFrame":
            from tigerml.core.reports import Table

            metrics = (
                metrics.transpose().reset_index(level=[1]).pivot(columns="dataset")[0]
            )
            metrics.index.name = None
            metrics.columns.name = None
            metrics = metrics.astype(str)
            metrics_table = Table(metrics, title="metrics", datatable=False)
            from tigerml.core.reports import table_styles

            metrics_table.apply_cell_format(
                {"width": table_styles.get_max_width},
                cols=list(metrics.columns),
                index=True,
            )
            report_dict["metrics"] = metrics_table
        else:
            report_dict["metrics"] = metrics

        plots_dict = self.evaluator.get_plots(cutoff_value=cutoff_value)
        report_dict["plots"] = plots_dict
        return report_dict

    def get_interpretation_report(
        self,
        include_shap=False,
        errorbuckets_spec=None,
        n_features=20,
        include_shap_test_error_analysis=False,
    ):
        """
        Interpret the evaluated model's performance with or without shap support.

        Parameters
        ----------
        include_shap : bool, default=True
            flag whether shap support need to include or not.
        errorbuckets_spec: dict
            specification related to prediction error (in regression) or probability cutoff (in classification) and
            top N columns value to be select from feature importance for error analysis
        n_features: int, default=20
            number of features to be select for error analysis based on feature importance
        include_shap_test_error_analysis: bool, default=False
            flag if error analysis need to perform on test data (subject to availability)

        Returns
        -------
        interpretation_report : dict
            a dictionary contains information regarding feature importance, dependence plots and prediction error data
            to be added in the final report.
        """
        if hasattr(self, "explainer") and include_shap:
            if self.report_option == 2 or self.multi_class:
                print("SHAP disabled as it is not applicable for the input provided")
                include_shap = False
        return self.explainer.get_plots(
            include_shap=include_shap,
            errorbuckets_spec=errorbuckets_spec,
            n_features=n_features,
            include_shap_test_error_analysis=include_shap_test_error_analysis,
        )

    def _get_report(
        self,
        format=".html",
        file_path="",
        include_shap=False,
        errorbuckets_spec=None,
        n_features=20,
        include_shap_test_error_analysis=False,
        cutoff_value=0.5,
        **kwargs,
    ):
        """Generate a consolidate report of model performance and interpretation."""
        report_dict = dict()
        report_dict["model_performance"] = self.get_performance_report(
            cutoff_value=cutoff_value
        )
        if (len(self.explainer.interpretations) > 0) or self.multi_class:
            report_dict["model_interpretation"] = self.get_interpretation_report(
                include_shap=include_shap,
                errorbuckets_spec=errorbuckets_spec,
                n_features=n_features,
                include_shap_test_error_analysis=include_shap_test_error_analysis,
            )
        if self.multi_class and (self.report_option == 1):
            shap_fe = get_shap_summary_plot(
                self.model, X=self.x_train, native_plot=False
            )
            if "feature_importance" not in report_dict["model_interpretation"].keys():
                report_dict["model_interpretation"]["feature_importance"] = {}
            report_dict["model_interpretation"]["feature_importance"]["from_shap"] = [
                shap_fe
            ]
        # In case of multi-class with report option 2,
        # "model_interpretation" gets created with blank dict
        # Need to remove "model_interpretation" from report_dict
        if self.multi_class and (self.report_option == 2):
            if "model_interpretation" in report_dict.keys():
                report_dict.pop("model_interpretation")
        if format == ".xlsx":
            keys_to_combine = [("model_interpretation", "error_analysis", "residual_analysis", "errorbucket_profiles"),  # noqa
                               ("model_interpretation", "shap_interpretation", "dependence_plots")]  # noqa

            from tigerml.core.utils import convert_to_tuples

            convert_to_tuples(keys_to_combine, report_dict)

        create_report(report_dict, name=file_path, format=format, **kwargs)


class RegressionReport(ModelReport):
    """Model evaluation toolkit for regression models.

    Generate report for model performance, interpretation & diagnostics.

    There are two options:

    * Option 1: Using model object.
        - Must have model, x_train, y_train.
        - Capable to generate full fledged diagnostics listed below.
    * Option 2: Using predictors
        - Must have y_train, yhat_train
        - generates only model performance

    Parameters
    ----------
    model : model or pipeline object, default=None
        Model object should have api similar to sklearn with methods `fit`, `predict`, and `get_model_type`
        If model object has `coef_` or `summary` methods, coefficient table will be enabled in the report.
        If pipeline is passed the last stage of the pipeline is assumed as model and rest of the pipeline is applied on x_train before running the report.

    y_train : pd.DataFrame, array-like of shape (n_samples,)
        The training target values. This is mandatory. When more than one column is sent, the first column is treated as y.

    y_test : Similar to y_train, default=None
        The testing target values. This is mandatory.

    X_train : {pd.DataFrame, array-like} of shape (n_samples, n_features), default=None
        The training input samples. If available model diagnostics are reported.

    X_test : Similar to x_train
        The testing input samples.

    yhat_train : Similar to y_train, default=None
        The training prediction samples. When model is None, this is mandatory.
        If model & x_train are passed, this is not mandatory.

    yhat_test : Similar to y_train, default=None
        The testing prediction samples.

    refit : bool, default=False
        If True, model is retrained with x_train & y_train

    Examples
    --------
    >>> from tigerml.model_eval import RegressionReport
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = fetch_california_housing(return_X_y=True)
    >>> X = pd.DataFrame(X, columns= ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.33, random_state=42)
    >>> reg = LinearRegression().fit(X_train, y_train)
    >>> yhat_train = reg.predict(X_train)
    >>> yhat_test = reg.predict(X_test)
    >>> # Option 1 - with model
    >>> regOpt1 = RegressionReport(y_train=y_train, model=reg, x_train=X_train, x_test=X_test, y_test=y_test)
    >>> regOpt1.get_report(include_shap=True)
    >>> # Option 2 - without model
    >>> regOpt2 = RegressionReport(y_train=y_train, x_train=X_train, x_test=X_test, y_test=y_test, yhat_train=yhat_train, yhat_test=yhat_test)
    >>> regOpt2.get_report(include_shap=True)
    """

    def __init__(
        self,
        y_train: pd.Series,
        model=None,
        x_train: pd.DataFrame = None,
        yhat_train: pd.Series = None,
        x_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        yhat_test: pd.Series = None,
        refit: bool = False,
    ):

        super().__init__(
            algo=algo_object.regression,
            model=model,
            y_train=y_train,
            x_train=x_train,
            yhat_train=yhat_train,
            y_test=y_test,
            x_test=x_test,
            yhat_test=yhat_test,
            refit=refit,
        )

    def residuals_plot(self):
        """Returns a scatter plot and distribution for the residuals of the model.

        A residual plot shows the residuals on the vertical axis and the
        independent variable on the horizontal axis.
        If the points are randomly dispersed around the horizontal axis, a linear
        regression model is appropriate for the data; otherwise, a non-linear
        model is more appropriate.

        Returns
        -------
        plot layout: holoviews plot layout object
            will have residual scatter plot for train data (and test data if passed)
        """
        return self.evaluator.residuals_plot()

    def prediction_error_plot(self):
        """Returns a scatter plot between actual and predicted values of the model.

        The prediction error visualizer plots the actual targets from the `dataset`
        against the predicted values generated by our model(s). This visualizer is
        used to detect noise or `heteroscedasticity` along a range of the target
        domain.

        Returns
        -------
        plot layout: holoviews plot layout object
            will have prediction error scatter plot for train data (and test data if passed)
        """
        return self.evaluator.prediction_error_plot()

    def get_report(
        self,
        format=".html",
        file_path="",
        include_shap=False,
        errorbuckets_spec=None,
        n_features=20,
        name=None,
        **kwargs,
    ):
        """
        Generate a consolidate report of model performance and interpretation.

        The generated report would have matrices values and prediction results under model performance section and
        feature importance under model interpretation section. If shap support is included (include_shap=True) then it
        will include shap values based feature importance, dependence plots and prediction error plots in the report. Refer `SHAP Interpretation <https://tigeranalytics-code-templates.readthedocs-hosted.com/projects/tigerml/en/latest/tigerml/model_eval.html#shap-baseline-ref>`__ for more information.

        It save the final report in the format and at location as specified by the user.

        Parameters
        ----------
        format : str, default='.html'
            format of report to be generated
        file_path : str, default=''
            location with filename where report to be saved. By default name is auto generated from system timestamp and saved in working directory.
        include_shap : bool, default=True
            flag whether shap support need to include or not for additional model interpretation. Note: this takes very long to run for tree models.
        errorbuckets_spec : dict, default={"type": "perc", "edges": [-0.2, 0.2], "labels": ["under predictions (-inf,-0.2]","correct predictions","over predictions [0.2, inf)"],"top_n_cols": 10}
            Specifications for prediction error buckets definitions and error analysis. It should be having following information:
                type - str
                    type of values (one of from 'perc' (for percentage) and 'abs' (for absolute))
                edges - list of floats
                    list of values to defined error buckets (negative for under-prediction and positive for over-prediction definitions)
                labels - list of str
                    list of labels to assign corresponding to each error buckets (should be more than 1 of edges length)
                top_n_cols - int
                    number of top columns to be select from model feature importance for error analysis
        n_features : int, default=20
            No. of top features used for interpretation. Applies to dependency plots and feature importance.
        excel_params : dict
            Dictionary containing the following keys if the format is ".xlsx".
            If a key is not provided, it will take the default values.

                - have_plot : boolean; default False.
                    If True, keep the plots in image format in excel report.
                - n_rows : int; default 100.
                    Number of sample rows to keep for plot types containing all the records in data (for example, density plot, scatter plot etc.)
        """
        if not file_path:
            file_path = "regression_report_at_{}".format(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )

        super()._get_report(
            format=format,
            file_path=file_path,
            include_shap=include_shap,
            errorbuckets_spec=errorbuckets_spec,
            n_features=n_features,
            include_shap_test_error_analysis=False,
            **kwargs,
        )

    def get_metrics(self):
        """Model evaluation metrics.

        Returns
        -------
        pd.Dataframe containing the performance metrics.
        """
        return self.evaluator.get_metrics()

    def get_evaluation_plots(self):
        """Regression evaluation plots as a dictionary."""
        return self.evaluator.get_plots()


class ClassificationReport(ModelReport):
    """Model evaluation toolkit for classification models.

    Generate report for model performance, interpretation & diagnostics.

    There are two options:

    * Option 1: Using model object.
        - Must have model, x_train, y_train.
        - Capable to generate full fledged diagnostics listed below.
    * Option 2: Using predictors
        - Must have y_train, yhat_train
        - generates only model performance

    Parameters
    ----------
    model : model or pipeline object, default=None
        Model object should have api similar to sklearn with methods `fit`, `predict_prba`, and `get_model_type`
        If model object has `coef_` or `summary` methods, coefficient table will be enabled in the report.
        If pipeline is passed the last stage of the pipeline is assumed as model and rest of the pipeline is applied on x_train before running the report.

    y_train : pd.DataFrame, array-like of shape (n_samples,)
        The training target values. This is mandatory. When more than one column is sent, the first column is treated as y.

    y_test : Similar to y_train, default=None
        The testing target values. This is mandatory.

    X_train : {pd.DataFrame, array-like} of shape (n_samples, n_features), default=None
        The training input samples. If available model diagnostics are reported.

    X_test : Similar to x_train
        The testing input samples.

    yhat_train : Similar to y_train, default=None
        Must be probabilities similar to output from `sklearn.predict_proba`.
        When model is None, this is mandatory. If yhat as more than one column, the last column is picked as the predicted class.
        If model & x_train are passed, this is not mandatory.

    yhat_test : Similar to yhat_train, default=None
        The testing prediction samples.

    refit : bool, default=False
        If True, model is retrained with x_train & y_train

    display_labels : dict, default=None
        A dict with class values are keys and class labes as values. By default, both are same.

    features_list : list, default=None
        A list of features keys and class labes as values. By default, both are same.

    Examples
    --------
    >>> from tigerml.model_eval import ClassificationReport
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.33, random_state=42)
    >>> cls = LogisticRegression().fit(X_train, y_train)
    >>> yhat_train = cls.predict(X_train)
    >>> yhat_test = cls.predict(X_test)
    >>> # Option 1 - with model
    >>> clsOpt1 = ClassificationReport(y_train=y_train, model=cls, x_train=X_train, x_test=X_test, y_test=y_test)
    >>> clsOpt1.get_report(include_shap=True)
    >>> # Option 2 - without model
    >>> clsOpt2 = ClassificationReport(y_train=y_train, x_train=X_train, x_test=X_test, y_test=y_test, yhat_train=yhat_train, yhat_test=yhat_test)
    >>> clsOpt2.get_report(include_shap=True)
    """

    def __init__(
        self,
        y_train: pd.Series,
        model=None,
        x_train: pd.DataFrame = None,
        yhat_train: pd.Series = None,
        x_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        yhat_test: pd.Series = None,
        refit: bool = False,
        display_labels: dict = None,
    ):
        super().__init__(
            algo=algo_object.classification,
            model=model,
            y_train=y_train,
            x_train=x_train,
            yhat_train=yhat_train,
            y_test=y_test,
            x_test=x_test,
            yhat_test=yhat_test,
            refit=refit,
            display_labels=display_labels,
        )

    def gains_table(self):
        """Gains table from predicted probabilities.

        This function gives a dataframe with columns as  no of True_positives, false_positive etc under each provided
        quantile which will be helpful to make lift charts

        Returns
        -------
        gains table(s): Table or Dictionary of Tables
            will have table if only Train data available otherwise will return a dictionary having tables for train and
            test data
        """
        if self.multi_class:
            print("Not applicable for multi-class classification")
            return None
        return self.evaluator.gains_table()

    def get_confusion_matrix(self, cutoff_value=0.5):
        """Interactive confusion matrix using holoview.

        Parameters
        ----------
        cutoff_value: float, default=0.5
            default probability cutoff value at which confusion matrix to be shown. Not applicable for multi-class classification.

        Returns
        -------
        plot: holoview's holomap object
        """
        if self.multi_class:
            return self.evaluator.get_confusion_matrix_multiclass()
        else:
            return self.evaluator.get_confusion_matrix(cutoff_value=cutoff_value)

    def get_class_distributions(self, orient=0):
        """Interactive plot of classwise prediction.

        Parameters
        ----------
        orient: int, default=0
            when 1, both train & test come in one row otherwise they are stacked one below the other.

        Returns
        -------
        plot: holoview's holomap object
        """
        if not (self.multi_class):
            print("Not applicable for binary classification")
            return None

        train_plot = self.evaluator.get_class_distributions()["Train Data"][0]
        if self.has_test:
            test_plot = self.evaluator.get_class_distributions()["Test Data"][0]
            if orient == 1:
                train_plot = train_plot + test_plot
            else:
                train_plot = (train_plot + test_plot).cols(1)
        return train_plot

    def get_classification_report(self, orient=0):
        """Interactive plot of classwise metrics.

        Parameters
        ----------
        orient: int, default=0
            when 1, both train & test come in one row otherwise they are stacked one below the other.

        Returns
        -------
        plot: holoview's holomap object
        """
        if not (self.multi_class):
            print("Not applicable for binary classification")
            return None

        train_plot = self.evaluator.get_classification_report()["Train Data"][0]
        if self.has_test:
            test_plot = self.evaluator.get_classification_report()["Test Data"][0]
            if orient == 1:
                train_plot = train_plot + test_plot
            else:
                train_plot = (train_plot + test_plot).cols(1)
        return train_plot

    def gains_chart(self, baseline=True, **kwargs):
        """Interactive Gains chart from `gains_table`.

        Parameters
        ----------
        baseline: bool, default True
            To include baseline
        kwargs: key, value mappings
            Other keyword arguments are passed down to hvPlot().

        Returns
        -------
        plot: holoview plot object
        """
        if self.multi_class:
            print("Not applicable for multi-class classification")
            return None

        return self.evaluator.gains_chart(baseline=baseline, **kwargs)

    def lift_chart(self, baseline=True, **kwargs):
        """Interactive Lift chart from `gains_table`.

        Parameters
        ----------
        baseline: bool, default True
            To include baseline
        kwargs: key, value mappings
            Other keyword arguments are passed down to hvPlot().

        Returns
        -------
        plot: holoview plot object
        """
        if self.multi_class:
            print("Not applicable for multi-class classification")
            return None

        return self.evaluator.lift_chart(baseline=baseline, **kwargs)

    def roc_curve(self, **kwargs):
        """Interactive roc plot using holoviews.

        Parameters
        ----------
        kwargs: key, value mappings
            Other keyword arguments are passed down to hvPlot().

        Returns
        -------
        plot: holoview plot object
        """
        if self.multi_class:
            print("Not applicable for multi-class classification")
            return None

        return self.evaluator.roc_curve(**kwargs)

    def precision_recall_curve(self):
        """Returns an interactive plot with a PR curve with average precision horizontal line.

        `Precision-Recall` curves are a metric used to evaluate a classifier's quality,
        particularly when classes are very imbalanced. The precision-recall curve
        shows the tradeoff between precision, a measure of result relevancy, and
        recall, a measure of how many relevant results are returned. A large area
        under the curve represents both high recall and precision, the best case
        scenario for a classifier, showing a model that returns accurate results
        for the majority of classes it selects.

        Returns
        -------
        plot: holoview plot object
        """
        if self.multi_class:
            print("Not applicable for multi-class classification")
            return None

        return self.evaluator.precision_recall_curve()

    def threshold_curve(self, **kwargs):
        """Returns line plot with `precision recall`, `f1 score` and `prevalence` as `threshold` is varied.

        Visualizes how `precision`, `recall`, `f1 score`, and `prevalence` change as the
        `discrimination threshold` increases. For probabilistic, binary classifiers,
        the discrimination threshold is the probability at which you choose the
        positive class over the negative. Generally this is set to 50%, but
        adjusting the `discrimination threshold` will adjust sensitivity to false
        positives which is described by the inverse relationship of `precision` and
        `recall` with respect to the threshold.

        Parameters
        ----------
        kwargs: key, value mappings
            Other keyword arguments are passed down to hvPlot().

        Returns
        -------
        plot: holoview plot object
        """
        if self.multi_class:
            print("Not applicable for multi-class classification")
            return None

        return self.evaluator.threshold_curve(**kwargs)

    def get_report(
        self,
        format=".html",
        file_path="",
        include_shap=False,
        errorbuckets_spec=None,
        n_features=20,
        cutoff_value=0.5,
        **kwargs,
    ):
        """
        Generate a consolidate report of model performance and interpretation.

        The generated report would have matrices values and prediction results under model performance section and
        feature importance under model interpretation section. If shap support is included (include_shap=True) then it
        will include shap values based feature importance, dependence plots and prediction error plots in the report. Refer `SHAP Interpretation <https://tigeranalytics-code-templates.readthedocs-hosted.com/projects/tigerml/en/latest/tigerml/model_eval.html#shap-baseline-ref>`__ for more information.

        It save the final report in the format and at location as specified by the user.

        Parameters
        ----------
        format : str, default='.html'
            format of report to be generated
        file_path : str, default=''
            location with filename where report to be saved. By default name is auto generated from system timestamp and saved in working directory.
        include_shap : bool, default=True
            flag whether shap support need to include or not for additional model interpretation. Note: this takes very long to run for tree models.
        errorbuckets_spec : dict, default={"cutoff": 0.5, "top_n_cols": 10}
            specifications for prediction error definition and error analysis. It would have following information:
                cutoff - float
                    probability cutoff value for class prediction (0 and 1)
                top_n_cols - int
                    number of columns to be selected for error analysis
        n_features : int, default=20
        No. of top features used for interpretation. Applies to dependency plots and
        feature importance.
        cutoff_value: float, default=0.5
            Probability cutoff_value for class prediction.
        excel_params : dict
            Dictionary containing the following keys if the format is ".xlsx".
            If a key is not provided, it will take the default values.

                - have_plot : boolean; default False.
                    If True, keep the plots in image format in excel report.
                - n_rows : int; default 100.
                    Number of sample rows to keep for plot types containing all the records in data (for example, density plot, scatter plot etc.)
        """
        if not file_path:
            file_path = "classification_report_at_{}".format(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )

        super()._get_report(
            format=format,
            file_path=file_path,
            include_shap=include_shap,
            errorbuckets_spec=errorbuckets_spec,
            n_features=n_features,
            include_shap_test_error_analysis=False,
            cutoff_value=cutoff_value, **kwargs,
        )

    def get_metrics(self, vary_thresholds=True, cutoff_value=0.5):
        """Model evaluation metrics as a dataframe.

        Parameters
        ----------
        vary_thresholds: bool, default=True
        cutoff_value: float, default=0.5

        Returns
        -------
        metrics: holoview object
        """
        return self.evaluator.get_metrics(
            vary_thresholds=vary_thresholds, cutoff_value=cutoff_value
        )

    def get_evaluation_plots(self, cutoff_value=0.5):
        """Model evaluation plots as a dictionary.

        Parameters
        ----------
        cutoff_value: float, default=0.5

        Retruns
        -------
        plot_dict: dict,
            all plots as key values pairs
        """
        return self.evaluator.get_plots(cutoff_value=cutoff_value)
