"""Description: Model Interpretation Module."""

import holoviews as hv
import hvplot.pandas
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
import tigerml.core.dataframe as td
from copy import deepcopy
from hvplot import hvPlot
from matplotlib import rcParams
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tigerml.core.common import ModelFeatureImportance
from tigerml.core.dataframe.base import TAPipeline
from tigerml.core.dataframe.helpers import detigerify
from tigerml.core.reports import format_tables_in_report
from tigerml.core.utils import DictObject, fail_gracefully, measure_time
from tigerml.core.utils.modeling import Algo
from tigerml.core.utils.pandas import get_bool_cols, get_cat_cols, get_num_cols

hv.extension("bokeh", "matplotlib")
hv.output(widget_location="bottom")
rcParams.update({"figure.autolayout": True})

_LOGGER = logging.getLogger(__name__)

MODEL_TYPES = DictObject(
    {
        "tree": "tree",
        "linear": "linear",
        "kernel": "kernel",
        "neuralnetwork": "neuralnetwork",
    }
)

SHAP_EXPLAINERS = DictObject(
    {
        "tree": shap.TreeExplainer,
        "linear": shap.LinearExplainer,
        "kernel": shap.KernelExplainer,
        "neuralnetwork": shap.DeepExplainer,
    }
)


def set_x_type(obj, cols=None):
    if isinstance(obj, np.ndarray):
        if cols is None:
            cols = ["Feature_" + str(i) for i in range(obj.shape[1])]
        obj = pd.DataFrame(obj, columns=cols)
    return obj


def dict_key_hier(t, s=[]):
    for key in t:
        s = s + [key]
        if type(t[key]) is dict:
            s = s + dict_key_hier(t[key])
    return list(set(s))


def _clear_shap(dict_to_clear):
    dict_shap = dict_to_clear.copy()
    dict_shap.pop("shap_interpretation", None)
    if "error_analysis" in dict_shap.keys():
        dict_shap["error_analysis"].pop("from_shap", None)
    if "feature_importance" in dict_shap.keys():
        dict_shap["feature_importance"].pop("from_shap", None)
    return dict_shap


def delete_keys_from_dict(dict_del, lst_keys):
    for k in lst_keys:
        try:
            del dict_del[k]
        except KeyError:
            pass
    for v in dict_del.values():
        if isinstance(v, dict):
            delete_keys_from_dict(v, lst_keys)
    return dict_del


def _format_error_bucket_label(error_edges):
    error_labels = []
    for i, val_ in enumerate(error_edges[1:]):
        if (error_edges[i] < 0) and (error_edges[i + 1] > 0):
            des = "Correct prediction"
        elif val_ < 0:
            des = "Under prediction"
        else:
            des = "Over prediction"
        error_labels.append(f"{des} ({error_edges[i]}, {val_}]")
    return error_labels


def regression_error_bucket(
    residual, y, error_edges, error_labels=None, threshold_type="perc"
):
    """Residual error buckets for regression models.

    Parameters
    ----------
    residual : pd.Series,
       Regression residuals equivalent to `y - yhat`
    y : pd.Series,
        actual values of y, optional when threshold_type is "abs"
    error_edges : pd.Series
        Threshold cut points for cutting residuals. Similar to edges in pd.cut
    error_labels : pd.Series, default=None
        Bucket labels for cutting residuals. Similar to labels in pd.cut.
        When None, labels are derived from error edges.
        Length must be 1 greater than that of bucket error edges.
    threshold_type: str, possible: ["perc", "abs"], default='perc'
        If percent, error_edges are used on as percent errors : `residual / y`
        If absolute, error_edges are used on residuals

    Returns
    -------
        pd.Series with error bucket labels
    """
    error_edges = [-np.inf] + error_edges + [np.inf]
    if error_labels is not None:
        if len(error_labels) + 1 != len(error_edges):
            raise ValueError(
                "Error edges must be one fewer than the number of error labels"
            )
    else:
        error_labels = _format_error_bucket_label(error_edges)

    residual_per = residual / y  # some inf needs to handled

    if threshold_type == "perc":
        error_bucket = pd.cut(residual_per, error_edges, labels=error_labels)
    elif threshold_type == "abs":
        error_bucket = pd.cut(residual, error_edges, labels=error_labels)
    return error_bucket


def classification_error_bucket(y, yhat, cutoff_value=0.5):
    """Confusion matrix based error buckets for classification models.

    Parameters
    ----------
    y : pd.Series,
        actual values of y, must be a series of 0 & 1
    y_hat : pd.Series,
        predicted probabilities of y, must be a series of float between 0 & 1
    cutoff_value : float, default=0.5
        cut_off value to derive predicted class from probability

    Returns
    -------
        pd.Series with error bucket labels
    """
    yhat_bin = (yhat > cutoff_value).astype(int)
    error_bucket = pd.Series(np.where(y == yhat_bin, "T", "F")).str.cat(
        pd.Series(np.where(yhat_bin == 1, "P", "N"))
    )
    return error_bucket


def get_multiclass_residual_plot(x, y_true, yhat, features=None, display_labels=None):
    if display_labels is None:
        display_labels = display_labels = dict(
            zip(set(y_true), [str(i) for i in set(y_true)])
        )
    if features is None:
        features = list(x.columns)
    y_pred = pd.Series(yhat.argmax(axis=1)).map(display_labels).values
    x_ = x.copy()
    x_["Actual"] = pd.Series(y_true).map(display_labels).values
    x_["Predicted"] = y_pred
    x_["Actual-Predicted"] = x_["Actual"] + "-" + x_["Predicted"]
    residual_plots = {}
    for class_ in x_["Actual"].unique():
        residual_plots_class = {}
        for feature in features:
            residual_plots_class[feature] = x_.loc[
                x_["Actual"] == class_
            ].hvplot.density(y=feature, by=["Actual-Predicted"])
        residual_plots[class_] = residual_plots_class
    return residual_plots


def sample_data(X, n=100):
    nrows = X.shape[0]
    X = X.sample(min([nrows, n]), random_state=42)
    return X


def calc_vif(X):
    """Calculate the Variance Inflation Factor (VIF) for the given data.

    VIF is a measure of multicollinearity between X variables. In general, a value of 10 or above indicated high degree of collinearity.

    Parameters
    ----------
    X: pd.Dataframe
        Input data frame with the features which VIF is to be computed.

    Returns
    -------
    pd.DataFrame
        Computed VIFs with each row for each feature.
    """
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


# def overlapping_histograms(dataset, col_idv, col_dv):
#     """Generate overlapping histogram.
#
#         For bivariate analysis between binary dependent variable
#         and continuous independent variable.
#
#         Input: dataset : Input dataframe.
#                col_idv : Continuous independent variable.
#                col_dv  : Binary target variable.
#
#         Output: The function generates an overlapping histogram
#                 over the dependent variable.
#
#         """
#
#     plt.clf()
#
#     majority = dataset[col_dv].value_counts(ascending=False).index[0]
#     minority = dataset[col_dv].value_counts(ascending=True).index[0]
#
#     sns.distplot(
#         dataset[col_idv][dataset[col_dv] == majority].values, hist=True, kde=False, color="red", label=majority,
#     )
#     sns.distplot(
#         dataset[col_idv][dataset[col_dv] == minority].values, hist=True, kde=False, color="blue", label=minority,
#     )
#     plt.grid(False)
#     plt.legend(loc="best")
#     plt.title("Overlapping histogram of " + col_idv + " over " + col_dv, fontsize=15)
#     plt.xlabel(col_idv, fontsize=13)
#     plt.ylabel("Distribution", fontsize=13)
#     plt.show()


def get_model_type(model):
    if isinstance(model, Pipeline):
        return get_model_type(model.steps[-1][1])

    model_module = str(model.__module__).lower()
    model_name = model.__class__.__name__.lower()
    if model_module.startswith("sklearn"):
        if (
            ".tree." in model_module
            or "forest" in model_module
            or "trees" in model_module
            or ("boost" in model_module and "adaboost" not in model_module)
            or (
                "ensemble" in model_module
                and ("boost" in model_name or "forest" in model_name)
            )
        ):
            return MODEL_TYPES.tree
        elif ".kernel." in model_module or ".svm." in model_module:
            return MODEL_TYPES.kernel
        elif ".linear_model." in model_module:
            return MODEL_TYPES.linear
        elif ".neural_network" in model_module:
            return MODEL_TYPES.neuralnetwork
    elif model_module.startswith("xgboost"):
        return MODEL_TYPES.tree
    elif model_module.__contains__("lightgbm"):
        return MODEL_TYPES.tree
    elif model_module.startswith("keras"):
        return MODEL_TYPES.neuralnerwork
    else:
        if "stats" in model_name:
            return MODEL_TYPES.linear
    return None


def _get_shap_explainer(model, X, model_type=None):
    model_str = str(type(model))
    model_type = model_type or get_model_type(model)
    if model_type is None:
        model_type = MODEL_TYPES.kernel
        _LOGGER.warning(
            "Could not infer the model_type from the model - {}. "
            "KernelExplainer is used for SHAP in such cases "
            "which could take a lot of time (hours, sometimes). "
            "Please pass model_type as one of these - {}.".format(
                model.__class__, MODEL_TYPES.keys()
            )
        )
    if model_type == MODEL_TYPES.kernel:
        model = deepcopy(model.predict)
    else:
        model = deepcopy(
            model
        )  # deepcopy is required to avoid overwriting model object
    if "XGB" in model_str:
        mybooster = model.get_booster()
        model_bytearray = mybooster.save_raw()[4:]

        def myfun(self=None):
            return model_bytearray

        mybooster.save_raw = myfun
        model = mybooster
        shap_explainer = SHAP_EXPLAINERS[model_type](model)
        _LOGGER.info(
            "\nData is not passed to ShapExplainer since it is an XGBoost model object.\n"
        )
    else:
        shap_explainer = SHAP_EXPLAINERS[model_type](model, sample_data(X, 100))
    return shap_explainer


def get_shap_summary_plot(
    model, X, model_type=None, native_plot=True, feature_names=None, class_names=None
):
    shap_explainer = _get_shap_explainer(model, X, model_type)
    try:
        shap_values = shap_explainer.shap_values(X)
    except Exception as e:  # TODO: Kiran please verify - Tamal
        error_msg = str(e)
        if "Additivity check failed" in error_msg:
            shap_values = shap_explainer.shap_values(
                X, approximate=True, check_additivity=False
            )
        else:
            raise Exception(error_msg)

    if not feature_names and isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    if class_names is None:
        class_names = ["Class {}".format(i) for i in range(len(shap_values))]

    if native_plot:
        shap_summary_plot = shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            feature_names=feature_names,
            class_names=class_names,
        )
    else:
        shap_values_matrix = np.array(shap_values)
        mean_abs_shap_values = np.mean(np.abs(shap_values_matrix), axis=1)
        plot_df = pd.DataFrame(
            mean_abs_shap_values, index=class_names, columns=feature_names
        )
        plot_df = plot_df.T
        plot_df["_row_sum"] = plot_df.sum(axis=1)
        plot_df.sort_values("_row_sum", inplace=True)
        plot_df.drop("_row_sum", axis=1, inplace=True)
        plot_df.index.name = "Features"
        plot_df.columns.name = "Class"
        shap_summary_plot = hvPlot(plot_df).barh(
            stacked=True,
            title="Average impact on model output magnitude",
            value_label="mean(|SHAP value|)",
        )
    return shap_summary_plot


algo_object = Algo()


class ModelInterpretation:
    """Model interpretation class declaration."""

    def __init__(
        self,
        model=None,
        model_type=None,
        algo=None,
        process_train=True,
        process_test=False,
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        yhat_train=None,
        yhat_test=None,
        residual_train=None,
        residual_test=None,
        multi_class=None,
        display_labels=None,
    ):
        if not (process_train or process_test):
            self.process_train = True
            self.process_test = False
        else:
            self.process_train = process_train
            self.process_test = process_test

        self.pipeline = None
        self.error = None
        self.model = model
        self.algo = algo
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.yhat_train = yhat_train
        self.yhat_test = yhat_test
        self.residual_train = residual_train
        self.residual_test = residual_test
        self.errorbucket_train = None
        self.errorbucket_test = None
        self.erroranalysis_cols = None
        self.feature_importance = None
        self.raw_x = None
        self.multi_class = multi_class
        self.display_labels = display_labels
        self.interpretations = self._get_interpretation_elements()
        self.shap_explainer = None
        if self.multi_class:
            self.interpretations = _clear_shap(self.interpretations)
            self.interpretations.pop("error_analysis", None)
            self.interpretations["error_analysis"] = {}
            if hasattr(self.model, "coef_"):
                # Coeff based feature importance is complex for multi-class models.
                self.interpretations.pop("feature_importance", None)
        if model is None:
            # Option only has residual_analysis
            self.report_option = 2
            self.interpretations.pop("coeff_table", None)
            self.interpretations.pop("feature_importance", None)
            self.interpretations = _clear_shap(self.interpretations)
            self.interpretations = delete_keys_from_dict(
                self.interpretations, ["error_analysis"]
            )
        else:
            self.report_option = 1
            # Extract model from pipeline
            if isinstance(model, Pipeline):
                model = TAPipeline(model)
            if isinstance(model, TAPipeline):
                if len(model.steps) > 1:
                    self.pipeline = model
                model = model.get_step(-1)
            if not model_type:
                model_type = get_model_type(model)
            self.model_type = model_type
            self.model = model
            if not (hasattr(self.model, "predict")):
                self.report_option = 2
                self.interpretations.pop("coeff_table", None)
                self.interpretations.pop("feature_importance", None)
                self.interpretations = _clear_shap(self.interpretations)
            else:
                # Algo Reference
                if algo is None and hasattr(self.model, "predict_proba"):
                    self.algo = algo_object.classification

                if hasattr(self.model, "predict_proba"):
                    self.interpretations = delete_keys_from_dict(
                        self.interpretations, ["errorbucket_drivers"]
                    )

                if not (hasattr(self.model, "summary") or hasattr(self.model, "coef_")):
                    self.interpretations.pop("coeff_table", None)

                if self.x_train is None:
                    self.interpretations.pop("coeff_table", None)
                    self.interpretations.pop("error_analysis", None)
                    self.interpretations = _clear_shap(self.interpretations)

                if self.pipeline:
                    # in case if ModelInterpretation used independently and Pipeline is passed for model
                    # TODO test if data is copied:
                    self.x_train = set_x_type(
                        self.pipeline.get_data_at_step(-1, self.x_train)
                    )
                    if self.has_test:
                        self.x_test = set_x_type(
                            self.pipeline.get_data_at_step(-1, self.x_test)
                        )

        if self.y_test is None:
            self.has_test = False
            self.interpret_test = False
        else:
            self.has_test = True

    def _get_interpretation_elements(self):
        interpretation_dict = {
            "coeff_table": {},
            "error_analysis": {
                "residual_analysis": {
                    "errorbucket_profiles": {},
                    # "errorbucket_drivers": {},
                },
                "from_shap": {"errorbucket": {}},
            },
            "feature_importance": {
                "from_model": {},
                "from_shap": {"shap_value_distribution": {}},
            },
            "shap_interpretation": {
                "shap_values": {},
                "dependence_plots_interpretation": {},
                "dependence_plots": {},
            },
        }
        return interpretation_dict

    def set_errorbuckets_spec(self, errorbuckets_spec=None):
        """Sets error buckets spec for Model interpretation class."""

        if self.x_train is not None:
            if algo_object.is_regression(self.algo):
                if errorbuckets_spec is not None:
                    if (
                        str(errorbuckets_spec["type"]) != "abs"
                        and str(errorbuckets_spec["type"]) != "perc"
                    ):
                        raise ValueError(
                            "Error thresholds type should be one of 'ABS' (for absolute value) or 'PERC' (for "
                            "percentage value)"
                        )
                    if sorted(errorbuckets_spec["edges"]) != errorbuckets_spec["edges"]:
                        raise ValueError(
                            "Error thresholds edges should be in ascending order."
                        )
                    if len(errorbuckets_spec["edges"]) < 2:
                        raise ValueError(
                            "Error thresholds edges should have at least 2 edges, one each for defining "
                            "under-prediction and over-prediction."
                        )
                    if ("labels" not in errorbuckets_spec.keys()) or len(
                        errorbuckets_spec["edges"]
                    ) + 1 != len(errorbuckets_spec["labels"]):
                        # Plot labels
                        errorbuckets_spec[
                            "labels"
                        ] = None  # create_labels(errorbuckets_spec["edges"])
                else:
                    errorbuckets_spec = {
                        "type": "perc",
                        "edges": [-0.2, 0.2],
                        "labels": [
                            "under predictions (-inf,-0.2]",
                            "correct predictions",
                            "over predictions [0.2, inf)",
                        ],
                        "top_n_cols": 10,
                    }
            else:
                if errorbuckets_spec is not None:
                    if "cutoff" in errorbuckets_spec.keys():
                        if not (
                            errorbuckets_spec["cutoff"] >= 0
                            and errorbuckets_spec["cutoff"] <= 1
                        ):
                            raise ValueError("Cutoff should be between 0 and 1")
                    else:
                        errorbuckets_spec["cutoff"] = 0.5

                    if "top_n_cols" not in errorbuckets_spec.keys():
                        if not ("cols" in errorbuckets_spec.keys()):
                            errorbuckets_spec["top_n_cols"] = 10
                else:
                    errorbuckets_spec = {"cutoff": 0.5, "top_n_cols": 10}

            self.errorbuckets_spec = errorbuckets_spec
            self._compute_errorbuckets()
            self._set_erroranalysis_cols()

    def _compute_errorbuckets(self):
        if self.errorbucket_train is None:
            self.error_buckets = None
            if algo_object.is_regression(self.algo):
                self.errorbucket_train = regression_error_bucket(
                    self.residual_train,
                    self.y_train,
                    error_edges=self.errorbuckets_spec["edges"],
                    error_labels=self.errorbuckets_spec["labels"],
                    threshold_type=self.errorbuckets_spec["type"],
                )
                if self.has_test and self.x_test is not None:
                    self.errorbucket_test = regression_error_bucket(
                        self.residual_test,
                        self.y_test,
                        error_edges=self.errorbuckets_spec["edges"],
                        error_labels=self.errorbuckets_spec["labels"],
                        threshold_type=self.errorbuckets_spec["type"],
                    )
            if algo_object.is_classification(self.algo):
                self.errorbucket_train = classification_error_bucket(
                    self.y_train,
                    self.yhat_train,
                    cutoff_value=self.errorbuckets_spec["cutoff"],
                )
                if self.has_test and self.x_test is not None:
                    self.errorbucket_test = classification_error_bucket(
                        self.y_test,
                        self.yhat_test,
                        cutoff_value=self.errorbuckets_spec["cutoff"],
                    )
            self.errorbucket_train = self.errorbucket_train.tolist()
            self.errorbucket_train = pd.Series(
                self.errorbucket_train, index=self.x_train.index
            )
            self.error_buckets = list(self.errorbucket_train.unique())
            if self.errorbucket_test is not None:
                self.errorbucket_test = self.errorbucket_test.tolist()
                self.errorbucket_test = pd.Series(
                    self.errorbucket_test, index=self.x_test.index
                )
                self.error_buckets = list(
                    np.unique(self.error_buckets + list(self.errorbucket_test.unique()))
                )
            self.error_buckets = [
                error_bucket
                for error_bucket in self.error_buckets
                if not pd.isnull(error_bucket)
            ]
            self.error_buckets = [
                error_bucket
                for error_bucket in self.error_buckets
                if error_bucket != "nan"
            ]

    def _set_erroranalysis_cols(self):
        if "cols" in self.errorbuckets_spec.keys():
            if len(self.errorbuckets_spec["cols"]) > 0:
                self.erroranalysis_cols = self.errorbuckets_spec["cols"]
        else:
            N = min(self.x_train.shape[1], self.errorbuckets_spec["top_n_cols"])
            if self.feature_importance is None:
                try:
                    df_feat = self.get_feature_importances(self.x_train, plot=False)
                    self.erroranalysis_cols = list(
                        df_feat.abs().sort_values("importance", ascending=False).index
                    )[:N]
                except Exception:
                    self.erroranalysis_cols = list(self.x_train.columns)[:N]
            else:
                self.erroranalysis_cols = list(
                    self.feature_importance.abs()
                    .sort_values("importance", ascending=False)
                    .index
                )[:N]
        self.erroranalysis_cols = list(
            set(self.erroranalysis_cols).intersection(self.x_train.columns)
        )

    def _set_erroranalysis_cols_shap(self, shap_values):
        if "cols" in self.errorbuckets_spec.keys():
            if len(self.errorbuckets_spec["cols"]) > 0:
                erroranalysis_cols_shap = self.errorbuckets_spec["cols"]
        else:
            N = min(self.x_train.shape[1], self.errorbuckets_spec["top_n_cols"])
            try:
                df_feat = pd.DataFrame(
                    {
                        "importance": abs(shap_values).mean(axis=0),
                        "features": list(self.x_train.columns),
                    }
                ).sort_values("importance", ascending=False)
                erroranalysis_cols_shap = list(df_feat.features)[:N]
            except Exception:
                erroranalysis_cols_shap = self.erroranalysis_cols
        return erroranalysis_cols_shap

    def get_coefs_table(self, vif=False):
        """Gets coefficients table."""
        if hasattr(self.model, "summary"):
            results_summary = self.model.summary()
            results_as_html = results_summary.tables[1].as_html()
            coeffs_table = pd.read_html(results_as_html, header=0, index_col=0)[0]
            coeffs_table.reset_index(inplace=True)
            coeffs_table.rename(columns={"index": "variables"}, inplace=True)
        elif hasattr(self.model, "coef_"):
            if self.model.coef_.ndim == 1:
                coeffs_table = pd.DataFrame(
                    {
                        "variables": self.x_train.columns.tolist(),
                        "coefficients": self.model.coef_,  # TODO, Raj verify this change.
                    }
                )
            else:
                coeffs_table = pd.DataFrame(self.model.coef_)
                coeffs_table.columns = self.x_train.columns.tolist()
                coeffs_table["class"] = self.display_labels
                coeffs_table = coeffs_table.melt(
                    id_vars="class", value_name="coefficients"
                )
                vif = False
        else:
            coeffs_table = None

        if coeffs_table is not None:
            if vif:
                vif = calc_vif(self.x_train)
                coeffs_table = coeffs_table.merge(vif, how="left", on=["variables"])
            if not ("class" in coeffs_table.columns):
                coeffs_table.set_index("variables", inplace=True)
                coeffs_table.index.name = None
                coeffs_table.columns.name = None
                coeffs_table.rename(columns={"coef": "coefficients"}, inplace=True)
            else:
                if all(coeffs_table["class"].isnull()):
                    coeffs_table = coeffs_table.drop(columns=["class"])
        return coeffs_table

    def _residual_analysis_by_feature(
        self, X, col_idv, categorical=True, train="train"
    ):
        """Generate distributions plots for elements by flag variable.

        Input: col_idv : Independent variable for which plot is to be generated.
               categorical : Independent variable type. (True - categorical, False - Continuous)
        Output: The function generates confusion matrix components as
                joint plot, distributions for categorical and continuous
                respectively.
        """
        # yet to test
        dataset = X.copy()
        if categorical:
            df_confusion = (
                dataset.groupby([col_idv, "errorbucket_label"]).size().reset_index()
            )
            df_confusion.columns = [col_idv, "errorbucket_label"] + ["values"]
            df_confusion[col_idv] = df_confusion[col_idv].str.wrap(50)
            df_confusion["values"].fillna(0, inplace=True)
            heatmap = df_confusion.hvplot.heatmap(
                x="errorbucket_label",
                y=col_idv,
                C="values",
                title="Index heatmap of "
                + col_idv
                + " over Confusion Matrix components",
                width=450,
                height=400,
            )
            # heatmap = heatmap * hv.Labels(heatmap)
            heatmap.opts(
                fontsize={"title": 15},
                xlabel=f"Confusion Matrix components{train}",
                ylabel=col_idv,
                width=450,
                height=400,
            )
            return heatmap
        else:
            return dataset.hvplot.kde(
                y=col_idv,
                by="errorbucket_label",  # Grouping by Predictions
                width=450,
                height=400,
                alpha=0.7,
                ylabel="density",
                xlabel=col_idv,
                title=f"{col_idv}(density){train}",
                legend="top_right",
            )

    def _get_multiclass_residual_plots(self, features=None):
        # features = list(self.x_train.columns)[:10]
        train_plot = get_multiclass_residual_plot(
            self.x_train,
            self.y_train,
            self.yhat_train,
            features,
            display_labels=self.display_labels,
        )
        if self.has_test:
            test_plot = get_multiclass_residual_plot(
                self.x_test,
                self.y_test,
                self.yhat_test,
                features,
                display_labels=self.display_labels,
            )
            cm_dict = {}
            cm_dict["Train Data"] = train_plot
            cm_dict["Test Data"] = test_plot
            train_plot = cm_dict
        return train_plot

    # def _get_X_and_shap_values(self, X):
    #     if self.shap_explainer is None:
    #         self._shap_fit()
    #     if len(X) > 1000:
    #         _LOGGER.warning(
    #             "Considering only a sample of 1000 data points, as SHAP interpretation could be slow on full data.")
    #         X = sample_data(X, 1000)
    #     shap_values = self._shap_score(X, get_expected_value=False)
    #     return X, shap_values

    # @fail_gracefully(_LOGGER)
    def shap_distribution(self, X=None, shap_values=None):
        """Gets the shap distributions."""
        # if X is None or shap_values is None:  # for user api
        #     X, shap_values = self._get_X_and_shap_values(self.x_train)
        if len(X) == 0:
            return "No of instances is 0."
        plt.close("all")
        plt.figure()
        if isinstance(X, td.DataFrame):
            X = detigerify(X)
        shap.summary_plot(shap_values, X, show=False)
        plot = plt.gcf()
        plt.close("all")
        return plot

    # @fail_gracefully(_LOGGER)
    def shap_feature_contributions(self, X=None, shap_values=None):
        """Gets the shap distributions."""
        # if X is None or shap_values is None:  # for user api
        #     X, shap_values = self._get_X_and_shap_values(self.x_train)
        plt.close("all")
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plot = plt.gcf()
        # t = plt.gca()
        # print(t.lines[0].get_data())
        plt.close("all")
        return plot

    # @fail_gracefully(_LOGGER)
    def get_errorbucket_profiles(self):
        """Gets the error bucket distributions."""
        # if self.errorbucket_train is None:
        #     self.set_errorbuckets_spec()
        x_cols = self.erroranalysis_cols
        X = self.x_train.copy()
        X = X[x_cols].reset_index(drop=True)
        num_cols = get_num_cols(X)
        cat_cols = get_cat_cols(X)
        bool_cols = get_bool_cols(X)
        num_cols = list(set(num_cols) - set(bool_cols))
        if len(bool_cols) > 0:
            df = X[bool_cols].reset_index()
            df = df.set_index("index")
            df_bool = df[df == 1].stack().reset_index().drop(0, axis=1)
        else:
            df_bool = None
        X["errorbucket_label"] = self.errorbucket_train.values
        if X["errorbucket_label"].isnull().sum() > 0:
            _LOGGER.warning(
                "Residual Analysis: Ignoring {} observations in train data for error analysis where y=0".format(
                    X["errorbucket_label"].isnull().sum()
                )
            )
            X = X.loc[
                X["errorbucket_label"].isnull() == False,  # noqa: E712
            ]
        if self.has_test:
            X_test = self.x_test.copy()
            X_test = X_test[x_cols].reset_index(drop=True)
            X_test["errorbucket_label"] = self.errorbucket_test.values
            if len(bool_cols) > 0:
                df_test = X_test[bool_cols].reset_index()
                df_test = df_test.set_index("index")
                df_test_bool = (
                    df_test[df_test == 1].stack().reset_index().drop(0, axis=1)
                )
            if X_test["errorbucket_label"].isnull().sum() > 0:
                _LOGGER.warning(
                    "Residual Analysis: Ignoring {} observations in test data for error analysis where y=0".format(
                        X_test["errorbucket_label"].isnull().sum()
                    )
                )
                X_test = X_test.loc[
                    X_test["errorbucket_label"].isnull() == False,  # noqa: E712
                ]

        plots_dict = {}
        for i in num_cols:
            train_plot = self._residual_analysis_by_feature(X, i, False)
            if self.has_test:
                train_plot = (
                    train_plot
                    + self._residual_analysis_by_feature(X_test, i, False, "test")
                ).cols(2)
                plots_dict[i] = [train_plot]
            else:
                plots_dict[i] = train_plot
        for i in cat_cols:
            train_plot = self._residual_analysis_by_feature(X, i, True)
            if self.has_test and self.x_test is not None:
                train_plot = (
                    train_plot
                    + self._residual_analysis_by_feature(X_test, i, True, "test")
                ).cols(2)
                plots_dict[i] = [train_plot]
            else:
                plots_dict[i] = train_plot
        if df_bool is not None:
            df_bool.rename(columns={"level_1": "categorical"}, inplace=True)
            df_bool = df_bool.merge(X["errorbucket_label"].reset_index(), on="index")
            df_bool.drop(["index"], axis=1, inplace=True)
            train_plot = self._residual_analysis_by_feature(
                df_bool, "categorical", True
            )
            if self.has_test and self.x_test is not None:
                df_test_bool.rename(columns={"level_1": "categorical"}, inplace=True)
                df_test_bool = df_test_bool.merge(
                    X_test["errorbucket_label"].reset_index(), on="index"
                )
                df_test_bool.drop(["index"], axis=1, inplace=True)
                train_plot = (
                    train_plot
                    + self._residual_analysis_by_feature(
                        df_test_bool, "categorical", True, "test"
                    )
                ).cols(2)
                plots_dict["categorical"] = [train_plot]
            else:
                plots_dict["categorical"] = train_plot
        return plots_dict

    # @fail_gracefully(_LOGGER)
    def get_error_drivers(self):
        """Gets the error drivers."""
        from tigerml.eda import Analyser

        # if self.erroranalysis_cols is None:
        #     self.set_errorbuckets_spec()
        x_cols = self.erroranalysis_cols
        df = self.x_train[x_cols]
        df["residuals"] = self.residual_train
        an_train = Analyser(df, y="residuals")
        if self.has_test and self.x_test is not None:
            df_test = self.x_test[x_cols]
            df_test["residuals"] = self.residual_test
            an_test = Analyser(df_test, y="residuals")
        bivariate_plot = {}
        for i in range(len(x_cols)):
            train_plot = an_train.bivariate_plots(
                x_vars=x_cols[i], y_vars="residuals"
            ).opts(xlabel=f"{x_cols[i]}(train)", width=500)

            if self.has_test and self.x_test is not None:
                train_plot = (
                    train_plot
                    + an_test.bivariate_plots(
                        x_vars=x_cols[i], y_vars="residuals"
                    ).opts(xlabel=f"{x_cols[i]}(test)", width=500)
                ).cols(2)
                bivariate_plot[x_cols[i]] = [train_plot]
            else:
                bivariate_plot[x_cols[i]] = train_plot
        return bivariate_plot

    # @fail_gracefully(_LOGGER)
    def _get_shap_intro(self):
        """
        # To be used in error analysis section.

            plots_dict["error_analysis"]["from_shap"] = {"shap_introduction": [self.get_shap_intro()]}
        """
        shap_intro = """
                    <div class="left_align">
                    <ul>
                        <li>
                            <b class="margin_top">Shapley value:</b>
                            <ul>
                                <li>It is the average of the marginal contributions across variable combinations.</li>
                            </ul>
                        </li>
                        <li>
                            <b class="margin_top">Feature importance Plot Interpretation:</b>
                            <ul>
                                <li>Features sorted by the average of Shapley values across data points.</li>
                            </ul>
                        </li>
                        <li>
                            <b class="margin_top">Shapley Value Distribution Plot Interpretation:</b>
                            <ul>
                                <li>Indicates effect of the variable associated with a higher or lower prediction.</li>
                                <li>Blues on right of "0" indicates negative correlation (lower variable values lead to higher prediction).</li>
                                <li>Reds on right of "0" indicates positive correlation (higher variable values lead to higher prediction).</li>
                            </ul>
                        </li>
                    </ul>
                    </div>
                    """
        return shap_intro

    # @fail_gracefully(_LOGGER)
    def get_shap_error_analysis(self, test_plot=True):
        """Gets the shap error analysis."""
        # if self.errorbucket_train is None:
        #     self.set_errorbuckets_spec(None)
        # if self.shap_explainer is None:
        #     self._shap_fit()
        plots_dict = dict()
        X_train = sample_data(self.x_train, 1000)
        indices = X_train.index
        if self.errorbucket_train.isnull().sum() > 0:
            _LOGGER.warning(
                "SHAP Error Analysis: Ignoring {} observations in train data for error analysis where y=0".format(
                    self.errorbucket_train.isnull().sum()
                )
            )
        if self.has_test and test_plot:
            X_test = sample_data(self.x_test, 1000)
            test_indices = X_test.index
            if self.errorbucket_test.isnull().sum() > 0:
                _LOGGER.warning(
                    "SHAP Error Analysis: Ignoring {} observations in test data for error analysis where y=0".format(
                        self.errorbucket_test.isnull().sum()
                    )
                )
        for error_bucket in self.error_buckets:
            key_ = "shap_distribution_for_{}".format(error_bucket)
            plots_dict[key_] = {}
            fil_ = np.array(self.errorbucket_train.loc[indices] == error_bucket)
            train_key = "({} out of {})(train)".format(fil_.sum(), len(X_train))
            if fil_.sum() > 0:
                shap_values = self._shap_score(
                    X_train.loc[fil_], get_expected_value=False
                )
                plots_dict[key_][train_key] = self.shap_distribution(
                    X_train.loc[fil_], shap_values
                )
            else:
                plots_dict[key_][train_key] = "No {}".format(error_bucket)

            if self.has_test and test_plot:
                fil_ = np.array(self.errorbucket_test.loc[test_indices] == error_bucket)
                test_key = "({} out of {})(test)".format(fil_.sum(), len(X_test))
                if fil_.sum() > 0:
                    shap_values = self._shap_score(
                        X_test.loc[fil_], get_expected_value=False
                    )
                    plots_dict[key_][test_key] = self.shap_distribution(
                        X_test.loc[fil_], shap_values
                    )
                else:
                    plots_dict[key_][test_key] = "No {}".format(error_bucket)

            else:
                key_new = key_ + train_key
                val_ = plots_dict[key_][train_key]
                plots_dict[key_new] = plots_dict.pop(key_)
                plots_dict[key_new] = val_

        return plots_dict

    # @fail_gracefully(_LOGGER)
    def _get_shap_values_plot(self, X, features):
        """Gets the shap values plot."""
        # if self.shap_explainer is None:
        #     self._shap_fit()
        shap_values, expected_value = self._shap_score(X, get_expected_value=True)
        if features is not None:
            cols_index = [X.columns.get_loc(c) for c in features]
            X = X[features]
            shap_values = shap_values[:, cols_index]
        force_plot_div_data = shap.force_plot(
            expected_value, shap_values, X, show=False
        )
        force_plot_div = force_plot_div_data.html()
        bundle_path = os.path.join(
            os.path.abspath(shap.__file__).rsplit("__init__.py", maxsplit=1)[0],
            "plots",
            "resources",
            "bundle.js",
        )
        with open(bundle_path, "r", encoding="utf-8") as f:
            bundle_data = f.read()
        force_plot_script = "<script>{}</script>".format(bundle_data)
        return force_plot_script + force_plot_div

    # @fail_gracefully(_LOGGER)
    def _get_dependence_plots_intro(self):
        """Gets the dependence plots."""
        shap_interpretation = """
            <div class="left_align">
            <ul>
              <li>The partial dependence plot shows the marginal effect a features has on the predicted outcome.</li>
              <li>It tells if the relationship between the target and a feature is linear, non-linear or complex.</li>
            </ul>
            </div>
        """
        return shap_interpretation

    # @fail_gracefully(_LOGGER)
    def get_dependence_plots(self, X=None, shap_values=None, features=None):
        """Gets the shap dependence plots."""
        # if X is None or shap_values is None:  # for user api
        #     X, shap_values = self._get_X_and_shap_values(self.x_train)
        if features is None:
            features = X.columns
        plt.close("all")
        dependence_plots = {}
        for col in features:
            shap.dependence_plot(
                col, shap_values, X, show=False, interaction_index=None
            )
            dependence_plots[col] = plt.gcf()
            plt.close("all")
        return dependence_plots

    @fail_gracefully(_LOGGER)
    def get_feature_importances(self, X, plot=True, n=20):
        """Gets the shap feature importance."""
        vizer = ModelFeatureImportance(model=deepcopy(self.model))
        feature_importances_, features_ = vizer._get_feature_importance(
            labels=X.columns
        )
        data = (
            td.DataFrame(feature_importances_)
            .set_index(features_)
            .merge(
                td.DataFrame(X.abs().mean().rename("mean")),
                left_index=True,
                right_index=True,
            )
        )
        if hasattr(self.model, "coef_"):
            feature_importances_ = data.iloc[:, 0].mul(data.iloc[:, 1]).to_numpy()
        elif hasattr(self.model, "feature_importances_"):
            feature_importances_ = data.iloc[:, 0].to_numpy()
        sort_idx = np.argsort(feature_importances_)
        vizer.features_ = features_[sort_idx]
        vizer.feature_importances_ = feature_importances_[sort_idx]
        self.feature_importance = vizer.get_plot_data(n=n)
        # vizer.score(X, y)
        if plot:
            return vizer.get_plot(n=n)
        else:
            return self.feature_importance

    def _get_shap_plots(self, features, test_plot=False):
        """Gets the shap plots."""
        plots_dict = {}
        if test_plot & self.has_test:
            plots_dict["train"] = self._get_shap_values_plot(
                X=sample_data(self.x_train, 1000),
                features=features,
            )
            plots_dict["test"] = self._get_shap_values_plot(
                X=sample_data(self.x_test, 1000),
                features=features,
            )
            return plots_dict
        else:
            return self._get_shap_values_plot(
                X=sample_data(self.x_train, 1000), features=features
            )

    # @fail_gracefully(_LOGGER)
    def _shap_fit(self):
        """Fit shap explainer."""
        # defining 'SHAP' kernel based on model type
        if not self.model_type:
            self.model_type = MODEL_TYPES.kernel
            _LOGGER.warning(
                "Could not infer the model_type from the model - {}. KernelExplainer is used for SHAP in such cases "
                "which could take a lot of time (hours, sometimes). Please pass model_type as one of these - {}.".format(
                    self.model.__class__, MODEL_TYPES.keys()
                )
            )
        if self.model_type == MODEL_TYPES.kernel:
            model = deepcopy(self.model.predict)
        else:
            model = deepcopy(
                self.model
            )  # deepcopy is required to avoid overwriting model object

        if "XGB" in str(type(self.model)):
            mybooster = model.get_booster()
            model_bytearray = mybooster.save_raw()[4:]

            def myfun(self=None):
                return model_bytearray

            mybooster.save_raw = myfun
            model = mybooster
            self.shap_explainer = SHAP_EXPLAINERS[self.model_type](model)
            _LOGGER.info(
                "\nData is not passed to ShapExplainer since it is an XGBoost model object.\n"
            )
        else:
            self.shap_explainer = SHAP_EXPLAINERS[self.model_type](
                model, sample_data(self.x_train, 100)
            )

        # except Exception as e:
        #     self.error = e
        #     return None
        #     # TODO: Raise warning and clear shap

    def _shap_score(self, X, get_expected_value=False):
        """Computing shap values and expected values."""
        # shap_values = self.shap_explainer.shap_values(X)
        try:  # using try catch to handle few common SHAPError
            shap_values = self.shap_explainer.shap_values(X)
        except Exception as e:  # TODO: Kiran please verify - Tamal
            error_msg = str(e)
            if "Additivity check failed" in error_msg:
                shap_values = self.shap_explainer.shap_values(
                    X, approximate=True, check_additivity=False
                )
            else:
                raise Exception(error_msg)
        if len(shap_values) == 2 and self.algo == "classification":
            shap_values = shap_values[1]
        elif len(shap_values) > 1 and len(shap_values) != len(X):
            _LOGGER.info("Shap values have a shape of {}".format(len(shap_values)))
            shap_values = shap_values[1]

        if get_expected_value:
            expected_value = self.shap_explainer.expected_value
            from collections.abc import Iterable

            if isinstance(expected_value, Iterable) and len(expected_value) > 1:
                expected_value = expected_value[1]
            return shap_values, expected_value

        return shap_values

    # @fail_gracefully(_LOGGER)
    def get_plots(
        self,
        include_shap=False,
        errorbuckets_spec=None,
        include_shap_test_error_analysis=False,
        n_features=20,
    ):
        """Gets the shap plots."""
        plots_dict = self.interpretations.copy()
        if not (include_shap) or (self.report_option == 2):
            plots_dict = _clear_shap(plots_dict)
        elif self.report_option == 1:
            self._shap_fit()
            X = sample_data(self.x_train, 1000)
            shap_values, expected_value = self._shap_score(X, get_expected_value=True)

        if "coeff_table" in plots_dict.keys():
            # plots_dict["coeff_table"] = self.get_coefs_table()
            coeff_table = self.get_coefs_table()
            coeff_table = format_tables_in_report(coeff_table, title="coeff_table")
            plots_dict["coeff_table"] = coeff_table

        if "feature_importance" in plots_dict.keys():
            if "from_model" in plots_dict["feature_importance"].keys():
                plots_dict["feature_importance"][
                    "from_model"
                ] = self.get_feature_importances(self.x_train, n=n_features)
            if "from_shap" in plots_dict["feature_importance"].keys():
                plots_dict["feature_importance"]["from_shap"] = {
                    "feature_contributions": self.shap_feature_contributions(
                        X, shap_values=shap_values
                    ),
                    "shap_value_distribution": self.shap_distribution(
                        X, shap_values=shap_values
                    ),
                }

        if not (self.multi_class):
            self.set_errorbuckets_spec(errorbuckets_spec)
        if "error_analysis" in plots_dict.keys():
            if "residual_analysis" in plots_dict["error_analysis"].keys():
                if (
                    "errorbucket_profiles"
                    in plots_dict["error_analysis"]["residual_analysis"].keys()
                ):
                    plots_dict["error_analysis"]["residual_analysis"][
                        "errorbucket_profiles"
                    ] = self.get_errorbucket_profiles()
                if (
                    "errorbucket_drivers"
                    in plots_dict["error_analysis"]["residual_analysis"].keys()
                ):
                    plots_dict["error_analysis"]["residual_analysis"][
                        "errorbucket_drivers"
                    ] = self.get_error_drivers()
            if "from_shap" in plots_dict["error_analysis"].keys():
                plots_dict["error_analysis"][
                    "from_shap"
                ] = self.get_shap_error_analysis(
                    test_plot=include_shap_test_error_analysis
                )
            if self.multi_class:
                plots_dict["error_analysis"] = self._get_multiclass_residual_plots()

        if "shap_interpretation" in plots_dict.keys():
            cols = self._set_erroranalysis_cols_shap(shap_values)
            plots_dict["shap_interpretation"]["shap_values"] = self._get_shap_plots(
                features=cols, test_plot=include_shap_test_error_analysis
            )
            plots_dict["shap_interpretation"]["dependence_plots_interpretation"] = [
                self._get_dependence_plots_intro()
            ]
            plots_dict["shap_interpretation"][
                "dependence_plots"
            ] = self.get_dependence_plots(X, shap_values=shap_values, features=cols)

        return plots_dict
