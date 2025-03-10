import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from tigerml.core.reports import create_report
from tigerml.core.utils import fail_gracefully

from .base import (
    ModelReport,
    print_dict_tree,
    set_x_type,
    set_y_type,
    verify_lengths,
    verify_x_type,
    verify_y_type,
)
from .plotters.evaluation import (
    ClassificationComparisonMixin,
    RegressionComparisonMixin,
)
from .plotters.interpretation import Algo, get_shap_summary_plot

algo_object = Algo()


class MultiModelReport:
    """Fits and scores the models given in the list format to ClassificationComparison and.

    RegressionComaparison classes.
    """

    def __init__(
        self,
        y,
        models,
        algo,
        x,
        yhats,
        refit=False,
        display_labels=None,
    ):
        self.y = y
        self.models = models
        self.algo = algo
        self.x = x
        self.yhats = yhats
        self.refit = refit
        self.display_labels = display_labels
        self.multi_class = type_of_target(self.y) == "multiclass"
        self._validate_inputs()
        self.reporters = self._create_standardized_report_dict()
        self._build_full_comp_element_tree()

    def _validate_inputs(self):
        # Validate data types
        verify_y_type(self.y)
        verify_x_type(self.x)

        # Set datatypes
        self.x = set_x_type(self.x)
        self.y = set_y_type(self.y, multi_class=self.multi_class)
        verify_lengths(self.y, self.x)
        if self.models is not None:
            if len(self.models) <= 1:
                raise ValueError(
                    "MultiModelReport requires more than 1 model to be passed."
                )
            if not (isinstance(self.models, dict) or isinstance(self.models, list)):
                raise ValueError(
                    "MultiModelReport requires more than 1 model to be passed."
                )
            self._interpertation = True

        if self.yhats is not None:
            if len(self.yhats) <= 1:
                raise ValueError(
                    "MultiModelReport requires more than 1 yhat to be passed."
                )
            if not isinstance(self.yhats, dict):
                raise ValueError(
                    "MultiModelReport requires more than 1 key value pair yhat to be passed as {'model_name': yhat}"
                )
            for model_name in self.yhats:
                verify_y_type(self.yhats[model_name])
                verify_lengths(self.yhats[model_name], self.y)
                self.yhats[model_name] = set_y_type(
                    self.yhats[model_name], multi_class=self.multi_class
                )
            self._interpertation = False

        if self.yhats is None:
            if self.models is None:
                raise ValueError("Either yhats or models required")
            elif self.x is None:
                raise ValueError(
                    "Requires x when models are provided and yhats is None"
                )

    def _create_standardized_report_dict(self):
        """Returns a dict of models with model details which is used in creating the comparison reports."""
        if isinstance(self.models, dict):
            self.model_names = list(self.models.keys())
            return dict(
                zip(
                    self.model_names,
                    [
                        ModelReport(
                            model=self.models[model],
                            x_train=self.x,
                            y_train=self.y,
                            algo=self.algo,
                            refit=self.refit,
                            display_labels=self.display_labels,
                        )
                        for model in self.models
                    ],
                )
            )
        elif isinstance(self.models, list):
            self.model_names = []
            for model in self.models:
                if isinstance(model, Pipeline):
                    model_name = ", ".join([step[0] for step in model.steps])
                else:
                    model_name = str(model).split("(")[0]
                self.model_names.append(model_name)
            # Ensure there are do duplicate names - will happen if self.models belong to the same class
            if len(set(self.model_names)) != len(self.model_names):
                duplicates = list(
                    set(
                        [
                            x
                            for idx, x in enumerate(self.model_names)
                            if x in self.model_names[idx + 1 :]
                        ]
                    )
                )
                for duplicate in duplicates:
                    for index, idx in enumerate(
                        [
                            idx
                            for idx, name in enumerate(self.model_names)
                            if name == duplicate
                        ]
                    ):
                        self.model_names[idx] = f"{duplicate} ({index})"
            return dict(
                zip(
                    self.model_names,
                    [
                        ModelReport(
                            model=model,
                            x_train=self.x,
                            y_train=self.y,
                            algo=self.algo,
                            refit=self.refit,
                            display_labels=self.display_labels,
                        )
                        for model in self.models
                    ],
                )
            )
        else:
            self.model_names = list(self.yhats.keys())
            return dict(
                zip(
                    self.model_names,
                    [
                        ModelReport(
                            x_train=self.x,
                            y_train=self.y,
                            yhat_train=self.yhats[model_name],
                            algo=self.algo,
                            refit=self.refit,
                            display_labels=self.display_labels,
                        )
                        for model_name in self.yhats
                    ],
                )
            )

    def feature_importances(self):
        """Returns a bar chart of feature importance of given models for comparison.

        Returns
        -------
        feature_importances: hvplot object
        """
        feature_importances = {}
        for model_name in self.reporters:
            feature_importances[model_name] = self.reporters[
                model_name
            ].explainer.get_feature_importances(
                self.reporters[model_name].explainer.x_train
            )
        return feature_importances

    def shap_summary_plots(self):
        """Returns stacked bar chart of mean absolute SHAP values.

        Returns
        -------
        shap_summary_plots : dict of hvplot objects
        """
        shap_summ_plots = {}
        for model_name in self.model_names:
            model = self.reporters[model_name].model
            X = self.reporters[model_name].x_train
            shap_fe = get_shap_summary_plot(model, X, native_plot=False)
            shap_summ_plots[model_name] = [shap_fe]
        return shap_summ_plots

    def _get_report(self, file_path="", cutoff_value=0.5, format=".html", **kwargs):
        """Generate a consolidate report having performance comparison of multiple models.

        Parameters
        ----------
        file_path : str, default=''
            location with filename where report to be saved. By default name is auto generated from system timestamp
            and saved in working directory.
        cutoff_value : float, default=0.5
            Probability cutoff_value for class prediction.
        """
        if algo_object.is_regression(self.algo):
            perf_dict = self.get_performance_report()
            self.element_tree["performance"]["performance_metrics"] = perf_dict[
                "performance_metrics"
            ]
            self.element_tree["performance"]["prediction_error"] = perf_dict[
                "prediction_error"
            ]["best_fit"]
            self.element_tree["residual_analysis"]["residual_distribution"] = perf_dict[
                "residual_distribution"
            ]
            self.element_tree["residual_analysis"]["actual_vs_predicted"] = perf_dict[
                "prediction_error"
            ]["actual_vs_predicted"]
            self.element_tree["residual_analysis"][
                "predicted_vs_residuals"
            ] = perf_dict["residual_plot"]["predicted_vs_residuals"]
        elif algo_object.is_classification(self.algo):
            perf_dict = self.get_performance_report(cutoff_value=cutoff_value)
            self.element_tree["performance"] = perf_dict

        if self._interpertation:
            if self.multi_class:
                interpret_dict = {}
                interpret_dict["feature_importances"] = self.shap_summary_plots()
                self.element_tree["interpretation"] = interpret_dict
            else:
                interpret_dict = {}
                interpret_dict["feature_importances"] = self.feature_importances()
                self.element_tree["interpretation"] = interpret_dict

        create_report(self.element_tree, name=file_path, format=format, **kwargs)

    def _build_full_comp_element_tree(self):
        self.element_tree = {}
        if algo_object.is_regression(self.algo):
            self.element_tree["performance"] = {
                "performance_metrics": {},
                "prediction_error": {},
            }
            self.element_tree["residual_analysis"] = {
                "residual_distribution": {},
                "actual_vs_predicted": {},
                "predicted_vs_residuals": {},
            }
        elif algo_object.is_classification(self.algo):
            self.element_tree["performance"] = {
                "performance_metrics": {},
                "confusion_matrices": {},
                "gains_charts": {},
                "lift_charts": {},
                "roc_curves": {},
                "precision_recall_curves": {},
                "threshold_analysis": {},
            }
        if self._interpertation:
            self.element_tree["interpretation"] = {"feature_importances": {}}

    def show_element_tree(self, expand_def=False):
        """Print element tree."""
        if self._interpertation:
            self.element_tree["interpretation"]["feature_importances"] = {}
        if algo_object.is_classification(self.algo):
            self.element_tree["performance"]["threshold_analysis"] = {}
        print_dict_tree(self.element_tree, 0, expand_def=expand_def)

    def add_report_element(self, element_name, element_func):
        """Add custom element (plot/table) to the report.

        Parameters
        ----------
        element_name : str
            report element (plot/table) name
        element_func : func
            function to get the element. hv plots are recommended for interactions.

        Examples
        --------
        >>> from tigerml.model_eval.plotters.evaluation.regression import create_scatter
        >>> import numpy as np
        >>> def plot_func():
        >>>     N = 50
        >>>     x = np.random.rand(N)
        >>>     y = np.random.rand(N)
        >>>     plot = create_scatter(x, y)
        >>>     return plot
        >>> self.add_report_element("x_y_scatter", plot_func())
        """
        self.element_tree[element_name] = element_func

    def remove_report_element(self, element_name):
        """Remove custom element (plot/table) from the report.

        Parameters
        ----------
        element_name : str
            report element (plot/table) name
        """
        self.element_tree.pop(element_name)


class ClassificationComparison(MultiModelReport, ClassificationComparisonMixin):
    """Model comparison toolkit for classification models.

    Generate report for model performance, interpretation & diagnostics.

    There are two options:

    * Option 1: Using list/dict model objects.
        - Must have list/dict models, x, y.
        - Capable to generate full fledged diagnostics listed below.
    * Option 2: Using predictors
        - Must have y, yhats (dict as {"model_name1": yhat1, "model_name2": yhat2 ..})
        - generates only model performance

    Parameters
    ----------
    models : list/dict models or pipeline object, default=None
        Each model object should have api similar to sklearn with methods `fit`, `predict`, and `get_model_type`
        If pipeline is passed the last stage of the pipeline is assumed as model and rest of the pipeline is applied on x before running the report.

    y : pd.DataFrame, array-like of shape (n_samples,)
        The target values. This is mandatory. When more than one column is sent, the first column is treated as y.

    x : {pd.DataFrame, array-like} of shape (n_samples, n_features), default=None
        The input samples. If available model diagnostics are reported.

    yhats : dict of yhats model names as keys, default=None
        Must be probabilities similar to output from `sklearn.predict_proba`.
        The prediction samples. When models is None, this is mandatory.
        If models & x are passed, this is not mandatory.

    refit : bool, default=False
        If True, model is retrained with x_train & y_train

    Examples
    --------
    >>> from tigerml.model_eval import ClassificationComparison
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.33, random_state=42)
    >>> # Model 1 - Logistic Regression
    >>> lr = LogisticRegression().fit(X_train, y_train)
    >>> yhat_test_lr = lr.predict_proba(X_test)
    >>> # Model 2 - Random Forest
    >>> rf = RandomForestClassifier().fit(X_train, y_train)
    >>> yhat_test_rf = rf.predict_proba(X_test)
    >>> # Option 1 - with model
    >>> clsOpt1 = ClassificationComparison(y=y_test, models=[lr, rf], x=X_test)
    >>> clsOpt1.get_report()
    >>> # Option 2 - without model
    >>> clsOpt2 = ClassificationComparison(y=y_test, yhats={"Logistic Regression":yhat_test_lr, "Random Forest":yhat_test_rf})
    >>> clsOpt2.get_report()
    """

    def __init__(
        self,
        y: pd.Series,
        models=None,
        x: pd.DataFrame = None,
        yhats=None,
        refit=False,
        display_labels: dict = None,
    ):

        super().__init__(
            y=y,
            models=models,
            algo=algo_object.classification,
            x=x,
            yhats=yhats,
            refit=refit,
            display_labels=display_labels,
        )

    def get_report(self, file_path="", cutoff_value=0.5, format=".html", **kwargs):
        """Generate a consolidate report of classification comparison for multiple models.

        Parameters
        ----------
        file_path : str, default=''
            location with filename where report to be saved. By default name is auto generated from system timestamp
            and saved in working directory.
        cutoff_value : float, default=0.5
            Probability cutoff_value for class prediction.
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
        if not file_path:
            file_path = f'classification_comparison_report_at_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        super()._get_report(
            file_path=file_path, cutoff_value=cutoff_value, format=format, **kwargs
        )  # noqa


class RegressionComparison(MultiModelReport, RegressionComparisonMixin):
    """Model comparison toolkit for regression models.

    Generate report for model performance, interpretation & diagnostics.

    There are two options:

    * Option 1: Using list/dict model objects.
        - Must have list/dict models, x, y.
        - Capable to generate full fledged diagnostics listed below.
    * Option 2: Using predictors
        - Must have y, yhats (dict as {"model_name1": yhat1, "model_name2": yhat2 ..})
        - generates only model performance

    Parameters
    ----------
    models : list/dict models or pipeline object, default=None
        Each model object should have api similar to sklearn with methods `fit`, `predict`, and `get_model_type`
        If pipeline is passed the last stage of the pipeline is assumed as model and rest of the pipeline is applied on x before running the report.

    y : pd.DataFrame, array-like of shape (n_samples,)
        The target values. This is mandatory. When more than one column is sent, the first column is treated as y.

    x : {pd.DataFrame, array-like} of shape (n_samples, n_features), default=None
        The input samples. If available model diagnostics are reported.

    yhats : dict of yhats with Similar to y with model names as keys, default=None
        The prediction samples. When models is None, this is mandatory.
        If models & x are passed, this is not mandatory.

    refit : bool, default=False
        If True, model is retrained with x_train & y_train

    Examples
    --------
    >>> from tigerml.model_eval import RegressionComparison
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = fetch_california_housing(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.33, random_state=42)
    >>> # Model 1 - Linear Regression
    >>> lr = LinearRegression().fit(X_train, y_train)
    >>> yhat_test_lr = lr.predict(X_test)
    >>> # Model 2 - Random Forest
    >>> rf = RandomForestRegressor().fit(X_train, y_train)
    >>> yhat_test_rf = rf.predict(X_test)
    >>> # Option 1 - with model
    >>> regOpt1 = RegressionComparison(y=y_test, models=[lr, rf], x=X_test)
    >>> regOpt1.get_report()
    >>> # Option 2 - without model
    >>> regOpt2 = RegressionComparison(y=y_test, yhats={"Linear Regression":yhat_test_lr, "Random Forest":yhat_test_rf})
    >>> regOpt2.get_report()
    """

    def __init__(
        self, y: pd.Series, models=None, x: pd.DataFrame = None, yhats=None, refit=False
    ):

        super().__init__(
            y=y,
            models=models,
            algo=algo_object.regression,
            x=x,
            yhats=yhats,
            refit=refit,
        )

    def get_report(self, file_path="", format=".html", **kwargs):
        """Generate a consolidate report of regression comparison for multiple models.

        Parameters
        ----------
        file_path : str, default=''
            location with filename where report to be saved. By default name is auto generated from system timestamp
            and saved in working directory.
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
        if not file_path:
            file_path = f'regression_comparison_report_at_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        super()._get_report(file_path=file_path, format=format, **kwargs)
