"""Description: Classification Evaluation Child class."""

import gc
import holoviews as hv
import logging
import numpy as np
import pandas as pd
from functools import reduce
from hvplot import hvPlot
from sklearn.metrics import auc, average_precision_score, classification_report
from sklearn.metrics import confusion_matrix as sklearn_cm
from sklearn.metrics import precision_recall_curve, roc_curve
from tigerml.core.reports import format_tables_in_report
from tigerml.core.scoring import SCORING_OPTIONS, TEST_PREFIX, TRAIN_PREFIX
from tigerml.core.utils._lib import fail_gracefully, flatten_list

from .base import Evaluator

_LOGGER = logging.getLogger(__name__)

hv.extension("bokeh", "matplotlib")
hv.output(widget_location="bottom")


def verify_x_type(obj):
    if obj is None:
        pass
    elif not (
        isinstance(obj, pd.DataFrame)
        or isinstance(obj, pd.Series)
        or isinstance(obj, np.ndarray)
    ):
        raise TypeError("Data should be of type pd.DataFrame / pd.Series / np.ndarray")


def set_x_type(obj, cols=None):
    if isinstance(obj, np.ndarray):
        if cols is None:
            obj_shape = 1 if obj.ndim == 1 else obj.shape[1]
            cols = ["Column_" + str(i + 1) for i in range(obj_shape)]
        obj = pd.DataFrame(obj, columns=cols)
    elif isinstance(obj, pd.Series):
        obj = pd.DataFrame(obj, columns=[obj.name])
    return obj


def _get_metric_elements(multi_class=False):
    if multi_class:
        metrics_dict = SCORING_OPTIONS.multi_class.copy()
    else:
        metrics_dict = SCORING_OPTIONS.classification.copy()

    return metrics_dict


def compute_pr_curve(y, yhat, multi_class=False, classes=[]):
    """Dataframe with precision-recall curve data and the average precision.

    Parameters
    ----------
        y : nd.array of size (n, ).
            Acutal data of prediction class. Should be 0 or 1.
        yhat : nd.array of size (n, ).
            Predicted probability of the class. Should be a float between 0 or 1.

    Returns
    -------
        pd.DataFrame
    """
    if not multi_class:
        if yhat.ndim == 2:
            yhat = yhat
        precision_, recall_, _ = precision_recall_curve(y, yhat)
        score_ = average_precision_score(y, yhat)
    else:
        from sklearn.preprocessing import label_binarize

        # Use label_binarize to be multi-label like settings
        y = label_binarize(y, classes=classes)
        precision_, recall_, score_ = {}, {}, {}
        # Compute PRCurve for all classes
        for i in range(len(classes)):
            precision_[i], recall_[i], _ = precision_recall_curve(y[:, i], yhat[:, i])
            score_[i] = average_precision_score(y[:, i], yhat[:, i])
        # Compute micro average PR curve
        precision_["micro"], recall_["micro"], _ = precision_recall_curve(
            y.ravel(), yhat.ravel()
        )
        score_["micro"] = average_precision_score(y, yhat, average="micro")
    if type(precision_) == dict:
        plot_data = []
        for i in range(len(classes)):
            plot_data += [
                pd.DataFrame({"precision": precision_[i], "recall": recall_[i]})
            ]
        avg_precision = score_["micro"]
    else:
        plot_data = pd.DataFrame({"precision": precision_, "recall": recall_})
        avg_precision = score_
    return plot_data, avg_precision


def create_pr_curve(y, yhat, multi_class=False, baseline=True, classes=[], **kwargs):
    """Interactive precision-recall plot using holoviews.

    Parameters
    ----------
        y : nd.array of size (n, 1).
            Acutal data of prediction class. Should be 0 or 1.
        yhat : nd.array of size (n, 1).
            Predicted probability of the class. Should be a float between 0 or 1.

    Returns
    -------
        holoview plot
    """
    plot_data, avg_precision = compute_pr_curve(y, yhat, multi_class, classes)
    if type(plot_data) == list:
        pr_curve_list = []
        for i in range(len(plot_data)):
            pr_curve_list += [
                hvPlot(plot_data[i], **kwargs).line(
                    x="recall",
                    y="precision",
                    label="PRCurve for class " + str(i),
                    ylim=(0, 1),
                )
            ]
        pr_curve = reduce((lambda x, y: x * y), pr_curve_list)
    else:
        label = "Binary PR Curve"
        if "label" in kwargs:
            label = kwargs["label"]
        pr_curve = hvPlot(plot_data, **kwargs).line(
            x="recall",
            y="precision",
            label=label,
            ylim=(0, 1),
        )
    if baseline:
        avg_p = hv.Curve(
            pd.DataFrame.from_dict({"x": [0, 1], "y": [avg_precision] * 2}),
            label="avg precision",
        )
        avg_p.opts(line_dash="dotted")
        final_plot = pr_curve * avg_p
    else:
        final_plot = pr_curve
    return final_plot


ventiles_ = list(np.arange(5, 105, 5) / 100)
deciles_ = list(np.arange(10, 110, 10) / 100)


def gains_table(
    y,
    yhat,
    cut_off_points=deciles_,
    custom_lift_data=None,
    show_lift=True,
    show_mean=False,
):
    """Gains table from predicted probabilies.

    This function gives a dataframe with columns as  no of True_positives, false_positive etc under each provided quantile which will be helpful to make lift charts
    Parameters
    ----------
        y : nd.array of size (n, 1).
            Acutal data of prediction class. Should be 0 or 1.
        yhat : nd.array of size (n, 1).
            Predicted probability of the positive class (i.e. model.predict_proba(X)[:, 1]). Should be a float between 0 or 1.
        cut_off_points: list of floats, default deciles - `list(np.arange(10, 110, 10) / 100)`
            Threshold cutoff points. Each cutoff point will be a row in Gains table.
        custom_lift_data : pd.Series of size (n, 1) OR nd.array/pd.DataFrame of size (n, k) where k represents number of features
            Additional custom data for lift calculation. For example, revenue.
        show_lift: bool, default=True
            whether to compute and show lift values for custom data
        show_mean: bool, default=False
            whether to keep mean values for custom data in the output
    Returns
    -------
        pd.DataFrame
    """
    cut_off_points = sorted(cut_off_points)
    y_pred = yhat.copy()
    df = pd.DataFrame()
    df["target"] = y.copy()
    df["probs"] = y_pred
    if custom_lift_data is not None:
        verify_x_type(custom_lift_data)
        custom_lift_data = set_x_type(custom_lift_data)
        for col in custom_lift_data.columns:
            df["lift_metric_" + col] = custom_lift_data[col]
    df.sort_values("probs", ascending=False, inplace=True)
    labels = [str(round(d * 100)) + "%" for d in cut_off_points]
    thr = df["probs"].quantile([(1 - d) for d in cut_off_points])
    # quant_label_mapping = dict(zip(quants, labels))
    out_df = pd.DataFrame(
        {
            "Bucket": labels,
            "Thresholds": thr.values,
            "bucket_size": 0,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
        }
    )
    if custom_lift_data is not None:
        for col in custom_lift_data.columns:
            out_df["mean_" + col] = 0
    out_df.sort_values("Thresholds", ascending=False, inplace=True)
    prevalence = float(df.target.mean())
    pop_events = float(df.target.sum())
    for i, decile in enumerate(cut_off_points):
        # import pdb
        # pdb.set_trace()
        top_df = df.iloc[: int(decile * len(df))]
        bottom_df = df.iloc[int(decile * len(df)) :]
        out_df.loc[i, "bucket_size"] = len(top_df)
        out_df.loc[i, "true_positive"] = (top_df.target == 1).sum()
        out_df.loc[i, "false_positive"] = (top_df.target == 0).sum()
        out_df.loc[i, "true_negative"] = (bottom_df.target == 0).sum()
        out_df.loc[i, "false_negative"] = (bottom_df.target == 1).sum()
        if custom_lift_data is not None:
            for col in custom_lift_data.columns:
                out_df.loc[i, "mean_" + col] = top_df["lift_metric_" + col].mean()
    out_df["targets_in_bucket"] = out_df["true_positive"] / (
        out_df["true_positive"] + out_df["false_positive"]
    )
    out_df["percent_targets_captured"] = out_df["true_positive"] / pop_events
    out_df["lift"] = out_df["targets_in_bucket"] / prevalence
    if custom_lift_data is not None:
        for col in custom_lift_data.columns:
            if show_lift:
                out_df["custom_lift_" + col] = out_df["mean_" + col] / float(
                    df["lift_metric_" + col].mean()
                )
            if not show_mean:
                out_df.drop("mean_" + col, axis=1, inplace=True)
    out_df.set_index("Bucket", inplace=True)
    return out_df


def gains_chart(df_gains_table, baseline=True, **kwargs):
    """Interactive Gains chart from `gains_table`.

    Parameters
    ----------
        df_gains_table : Gains table dataframe
            Should be a identical to return from `gains_table`
        baseline: bool, default True
            To include baseline
        kwargs:
            Holoviews plotting options.

    Returns
    -------
        holoview plot
    """
    deciles = [float(i.replace("%", "")) / 100 for i in df_gains_table.index]
    gains_data = (
        df_gains_table["percent_targets_captured"].reset_index(drop=True).to_frame()
    )
    gains_data["deciles"] = deciles
    gains_curve = hvPlot(gains_data).line(
        y="percent_targets_captured",
        x="deciles",
        xlim=(0, 1),
        ylim=(0, max(gains_data["percent_targets_captured"])),
        **kwargs,
    )
    if baseline:
        df = pd.Series(deciles, name="deciles").to_frame()
        gains_baseline = hvPlot(df).line(y="deciles", x="deciles", label="Baseline")

        gains_chart = gains_curve * gains_baseline
        gains_chart.opts(legend_position="right")
    else:
        gains_chart = gains_curve
    return gains_chart


def lift_chart(df_gains_table, baseline=True, **kwargs):
    """Interactive Lift chart from `gains_table`.

    Parameters
    ----------
        df_gains_table : Gains table dataframe
            Should be a identical to return from `gains_table`
        baseline: bool, default True
            To include baseline
        kwargs:
            Holoviews plotting options.

    Returns
    -------
        holoview plot
    """
    deciles = [float(i.replace("%", "")) / 100 for i in df_gains_table.index]
    lift_data = df_gains_table["lift"].reset_index(drop=True).to_frame()
    lift_data["deciles"] = deciles
    lift_curve = hvPlot(lift_data).line(
        y="lift", x="deciles", ylim=(1, max(lift_data["lift"])), **kwargs
    )
    if baseline:
        hline = hv.HLine(1, label="Baseline")
        lift_chart = lift_curve * hline
        lift_chart.opts(legend_position="right")
    else:
        lift_chart = lift_curve
    return lift_chart


def compute_roc_curve(y, yhat, multi_class=False, classes=None):
    """Computes data for roc_curve.

    Parameters
    ----------
        y : nd.array of size (n, 1).
            Acutal data of prediction class. Should be 0 or 1.
        yhat : nd.array of size (n, 1).
            Predicted probability of the class. Should be a float between 0 or 1.

    Returns
    -------
        pd.DataFrame
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if not (multi_class):
        fpr["macro"], tpr["macro"], _ = roc_curve(y, yhat)
        roc_auc_ = auc(fpr["macro"], tpr["macro"])
    else:
        n_classes = yhat.shape[1]
        # Get the curve for +ve class
        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y, yhat[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        from scipy import interp
        from sklearn.preprocessing import label_binarize

        y_test = label_binarize(y, classes=classes)
        if n_classes == 2:
            y_test = np.hstack((1 - y_test, y_test))
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yhat.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        roc_auc_ = roc_auc["macro"]

    plot_data = pd.DataFrame(
        {"True Positive Rate": tpr["macro"], "False Positive Rate": fpr["macro"]}
    )
    return plot_data, roc_auc_


def create_roc_curve(y, yhat, label="LogisticRegression", **kwargs):
    """Interactive roc plot using holoviews.

    Parameters
    ----------
        y : nd.array of size (n, 1).
            Acutal data of prediction class. Should be 0 or 1.
        yhat : nd.array of size (n, 1).
            Predicted probability of the class. Should be a float between 0 or 1.
        label: string, default "1"
            class label useful when plotting mutliple models or comparing mulitple binary-model experiments.
        kwargs:
            Holoviews plotting options.

    Returns
    -------
        holoview plot
    """
    plot_data, roc_auc = compute_roc_curve(y, yhat)
    plot = hvPlot(plot_data).line(
        x="False Positive Rate",
        y="True Positive Rate",
        xlim=(0, 1),
        label="ROC for {}, AUC: {}".format(label, round(roc_auc, 3)),
        **kwargs,
    )
    return plot


def compute_threshold_data(y, yhat, metrics_dict=None):
    """Dataframe with threshold curve data.

    Parameters
    ----------
        y : nd.array of size (n, ).
            Acutal data of prediction class. Should be 0 or 1.
        yhat : nd.array of size (n, ).
            Predicted probability of the class. Should be a float between 0 or 1.
    """
    if metrics_dict is None:
        metrics_dict = SCORING_OPTIONS.classification.copy()
    thresholds_ = np.arange(0.01, 1, 0.01)
    plot_data = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "% of Class 1": [],
        "threshold": thresholds_,
    }

    for threshold in thresholds_:
        predict_class = (yhat > threshold).astype(int)
        plot_data["% of Class 1"].append(predict_class.mean())
        for key_ in ["precision", "recall", "f1_score"]:
            func = metrics_dict[key_]["func"]
            default_params = metrics_dict[key_].get("default_params", {})
            train_params = [y, predict_class]
            plot_data[key_].append(func(*train_params, **default_params))

    plot_data = pd.DataFrame(plot_data)
    return plot_data


def create_threshold_chart(y, yhat, **kwargs):
    """Interactive threshold plot using holoviews.

    Returns an interactive line plot with fl_score, precision, recall & prevelance with thresholds as x-axis.

    Parameters
    ----------
    y : nd.array of size (n, 1).
        Acutal data of prediction class. Should be 0 or 1.
    yhat : nd.array of size (n, 1).
        Predicted probability of the class. Should be a float between 0 or 1.
    kwargs: key, value mappings
        Other keyword arguments are passed down to hvPlot().
    """
    plot_data = compute_threshold_data(y, yhat)

    plot = None
    x = "threshold"
    if plot_data is None:
        return None
    for col in plot_data.columns:
        if x == col:  # Do not plot threshold
            continue
        current_plot = hvPlot(plot_data).line(
            x=x,
            y=col,
            xlim=(0, 1),
            ylim=(0, 1),
            label=col,
            **kwargs,
        )
        if plot is None:
            plot = current_plot
        else:
            plot = plot * current_plot
    plot.opts(legend_position="right")
    return plot


class ClassificationEvaluation(Evaluator):
    """Classification evaluation class."""

    def __init__(
        self,
        model=None,
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        yhat_train=None,
        yhat_test=None,
        multi_class=False,
        display_labels=None,
    ):
        """
        Classification evaluation class.

        Parameters
        ----------
        model : a `Scikit-Learn` Classifier
            Should be an instance of a `classifier`.
        """

        super().__init__(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            yhat_train,
            yhat_test,
            display_labels,
        )
        self.multi_class = multi_class
        self.metrics = _get_metric_elements(multi_class=self.multi_class)
        # self.plots = _get_plot_elements(multi_class=self.multi_class)
        if not self.multi_class:
            self.plots = {
                "gains_chart": self.gains_chart,
                "lift_chart": self.lift_chart,
                "confusion_matrix": self.get_confusion_matrix,
                "roc_curve": self.roc_curve,
                "precision_recall": self.precision_recall_curve,
                "threshold_curve": self.threshold_curve,
            }
        else:
            self.plots = {
                "classification_report": self.get_classification_report,
                "confusion_matrix": self.get_confusion_matrix_multiclass,
                "class_distribution": self.get_class_distributions,
            }
            # probability_distribution plot is shown only if predicted values are given in the form of probability
            if any(pd.DataFrame(self.yhat_train).nunique() > 2):
                self.plots[
                    "probability_distribution"
                ] = self.get_probability_distribution
        self.gains_table_result_train = None
        self.gains_table_result_test = None

    @property
    def gives_probs(self):
        """Returns if attribute predict probabilies is available in model object."""
        return hasattr(self.model, "predict_proba")

    @fail_gracefully(_LOGGER)
    def precision_recall_curve(self, baseline=True, **kwargs):
        """Returns an interactive plot with a PR curve with average precision horizontal line.

        `Precision-Recall` curves are a metric used to evaluate a classifier's quality,
        particularly when classes are very imbalanced. The precision-recall curve
        shows the tradeoff between precision, a measure of result relevancy, and
        recall, a measure of how many relevant results are returned. A large area
        under the curve represents both high recall and precision, the best case
        scenario for a classifier, showing a model that returns accurate results
        for the majority of classes it selects.
        """
        train_plot = create_pr_curve(
            self.y_train, self.yhat_train, baseline=baseline, **kwargs
        ).opts(title="Train Data", frame_width=350, frame_height=300)
        if self.has_test:
            test_plot = create_pr_curve(
                self.y_test, self.yhat_test, baseline=baseline, **kwargs
            ).opts(title="Test Data", frame_width=350, frame_height=300)
            return (train_plot + test_plot).cols(2)
        return train_plot

    def _get_classification_report(self, y_true, yhat):
        # FIXME: This chart takes a lot of white space.
        y_pred = pd.Series(yhat.argmax(axis=1)).values
        cr = classification_report(y_true, y_pred, output_dict=True)
        cr = pd.DataFrame(cr).T
        cr = cr.reset_index().rename(columns={"index": "#"})
        cr = cr.loc[~cr["#"].isin(["accuracy", "macro avg", "weighted avg"])]
        cr.reset_index(drop=True, inplace=True)
        cr = hvPlot(cr).table()
        return cr

    def get_classification_report(self):
        """Gets report for classification."""
        train_report = self._get_classification_report(
            self.y_train, self.yhat_train
        ).opts(title="Train Data")
        if self.has_test:
            test_report = self._get_classification_report(
                self.y_test, self.yhat_test
            ).opts(title="Test Data")
            # cr_dict = {}
            # cr_dict["Train Data"] = [train_report]
            # cr_dict["Test Data"] = [test_report]
            # train_report = cr_dict
            train_report = (train_report + test_report).cols(1)
        return train_report

    def _get_class_distributions(self, y_true, yhat):
        y_pred = pd.Series(yhat.argmax(axis=1)).map(self.display_labels).value_counts()
        y_true = pd.Series(y_true).map(self.display_labels).value_counts()
        df_cd = pd.concat([y_true, y_pred], axis=1)
        df_cd.sort_index(inplace=True)
        df_cd.columns = ["Actual", "Predicted"]
        plot_true = df_cd["Actual"].hvplot.bar(line_color="k", color=None)
        plot_pred = df_cd["Predicted"].hvplot.bar(color="b", alpha=0.5, line_color=None)
        cd_plot = plot_true * plot_pred
        cd_plot = cd_plot.opts(
            frame_width=350, frame_height=300, legend_position="top_right"
        )
        return cd_plot

    # @fail_gracefully
    def get_class_distributions(self):
        """Gets the class distributions."""
        train_plot = self._get_class_distributions(self.y_train, self.yhat_train).opts(
            title="Train Data", legend_position="right"
        )
        if self.has_test:
            test_plot = self._get_class_distributions(self.y_test, self.yhat_test).opts(
                title="Test Data", legend_position="right"
            )
            # cd_dict = {}
            # cd_dict["Train Data"] = [train_plot]
            # cd_dict["Test Data"] = [test_plot]
            # train_plot = cd_dict
            train_plot = (train_plot + test_plot).cols(2)
        return train_plot

    def _compute_cm(self, y_true, y_pred, return_df=True):
        cm = sklearn_cm(y_true, y_pred)
        if not return_df:
            return cm
        cm_df = pd.DataFrame(cm)
        if self.multi_class:
            cm_df.rename(
                columns={
                    i: "Predicted_" + self.display_labels[i]
                    for i in self.display_labels
                },
                inplace=True,
            )
            cm_df.rename(
                index={
                    i: "Actual_" + self.display_labels[i] for i in self.display_labels
                },
                inplace=True,
            )
        else:
            cm_df.rename(
                columns={c: "Predicted_" + str(c) for c in cm_df.columns}, inplace=True
            )
            cm_df.rename(
                index={i: "Actual_" + str(i) for i in cm_df.index}, inplace=True
            )
        cm_df = cm_df.reset_index().rename(columns={"index": "#"})
        return cm_df

    def confusion_matrix(self, normalized=False, flattened=False):
        """Returns the confusion matrix for the model as dataframe."""
        cms = {}
        cm_dfs = {}
        if self.has_train:
            y_pred_train = self.yhat_train.copy()
            y_pred_train[y_pred_train >= 0.5] = 1
            y_pred_train[y_pred_train < 1] = 0
            cms[TRAIN_PREFIX] = self._compute_cm(
                self.y_train, y_pred_train, return_df=False
            )
        if self.has_test:
            y_pred_test = self.yhat_test.copy()
            y_pred_test[y_pred_test >= 0.5] = 1
            y_pred_test[y_pred_test < 1] = 0
            cms[TEST_PREFIX] = self._compute_cm(
                self.y_test, y_pred_test, return_df=False
            )
        for dataset in cms:
            if normalized:
                cms[dataset] = (
                    cms[dataset].astype("float")
                    / cms[dataset].sum(axis=1)[:, np.newaxis]
                )
            cm_dfs[dataset] = pd.DataFrame(cms[dataset])
            cm_dfs[dataset] = cm_dfs[dataset].rename(
                columns=dict(
                    zip(
                        [x for x in cm_dfs[dataset].columns],
                        [
                            "predicted_{}{}".format(
                                x, "_normalized" if normalized else ""
                            )
                            for x in cm_dfs[dataset].columns
                        ],
                    )
                )
            )
        if flattened:
            classes = cm_dfs[TRAIN_PREFIX].index.values
            labels = [
                [
                    "predicted_{}_for_{}".format(pred_class, true_class)
                    for pred_class in classes
                ]
                for true_class in classes
            ]
            labels = flatten_list(labels)
            labels = [
                label
                + (
                    " (False Positives)"
                    if "1_for_0" in label
                    else " (False Negatives)"
                    if "0_for_1" in label
                    else ""
                )
                for label in labels
            ]
            flat_cm_dict = {}
            for index, label in enumerate(labels):
                flat_cm_dict[label] = {}
                for dataset in self.datasets:
                    flat_cm_dict[label][dataset] = cm_dfs[dataset].values.ravel()[index]
            # values = np.append(cm_df_train.values.ravel(), [cm_df_test.values.ravel()])
            dict_of_df = {k: pd.DataFrame([v]) for k, v in flat_cm_dict.items()}
            flat_cm_df = pd.concat(dict_of_df, axis=1)
            # flat_cm_df = pd.DataFrame(columns=pd.MultiIndex.from_product([labels, self.datasets]))
            # flat_cm_df.loc[0] = values
            return flat_cm_df
        cm_df = pd.concat(
            {dataset: cm_dfs[dataset] for dataset in self.datasets}, axis=1
        )
        cm_df.index = cm_df.index.rename("true_label")
        cm_df.columns.set_names(["dataset", "metric"], inplace=True)
        if normalized:
            cm_df = cm_df.rename(
                columns=dict(
                    zip(
                        [x for x in cm_df.columns],
                        [str(x) + "_normalized" for x in cm_df.columns],
                    )
                )
            )
        return cm_df

    @fail_gracefully(_LOGGER)
    def _get_conf_matrix(self, y_true, yhat, vary_thresholds=True):
        """Returns confusion matrix as a HoloMap or Table object."""
        if not self.multi_class and vary_thresholds:
            cm_dict = {}
            for thresh in range(5, 100, 5):
                y_pred = yhat.copy()
                y_pred[y_pred >= (thresh / 100)] = 1
                y_pred[y_pred < 1] = 0
                cm_df = self._compute_cm(y_true, y_pred)
                cm_dict.update({thresh / 100: hvPlot(cm_df).table()})
            plot = hv.HoloMap(cm_dict, kdims=["decision_threshold"])
            # .opts(frame_width=350, frame_height=300)
        else:
            if not self.multi_class:
                y_pred = yhat.copy()
                y_pred[y_pred >= 0.5] = 1
                y_pred[y_pred < 1] = 0
                cm_df = self._compute_cm(y_true, y_pred)
            else:
                y_pred = pd.Series(yhat.argmax(axis=1)).values
                cm_df = self._compute_cm(y_true, y_pred)
                cm_df = hvPlot(cm_df).table()
            plot = cm_df
        return plot

    def get_confusion_matrix(self, cutoff_value=0.5):
        """Gets confusion matrix."""
        train_plot = self._get_conf_matrix(self.y_train, self.yhat_train).opts(
            title="Train Data", width=350, height=100
        )
        if self.has_test:
            test_plot = self._get_conf_matrix(self.y_test, self.yhat_test).opts(
                title="Test Data", width=350, height=100
            )
            cm_dict = {}
            for thresh in range(5, 100, 5):
                cm_dict.update(
                    {
                        thresh
                        / 100: hv.Layout(
                            [train_plot[thresh / 100], test_plot[thresh / 100]]
                        )
                    }
                )
            train_plot = hv.HoloMap(cm_dict, kdims=["decision_threshold"])
            train_plot.kdims[0].default = cutoff_value
            # train_plot = (train_plot + test_plot).cols(1)
        return train_plot

    def get_confusion_matrix_multiclass(self, cutoff_value=0.5):
        """Gets confusion matrix for multi_class."""
        train_plot = self._get_conf_matrix(self.y_train, self.yhat_train).opts(
            title="Train Data", width=350, height=100
        )
        if self.has_test:
            test_plot = self._get_conf_matrix(self.y_test, self.yhat_test).opts(
                title="Test Data", width=350, height=100
            )
            train_plot = (train_plot + test_plot).cols(2)
        return train_plot

    def _get_probability_distribution(self, y_true, yhat):
        """Gets probability distributions."""
        y_pred = pd.Series(yhat.argmax(axis=1)).map(self.display_labels).values
        cols = [self.display_labels[i] for i in range(yhat.shape[1])]
        cm_df = pd.DataFrame(yhat, columns=cols)
        cm_df["Actual"] = pd.Series(y_true).map(self.display_labels).values
        cm_df["Predicted"] = y_pred
        cm_df["Actual-Predicted"] = cm_df["Actual"] + "-" + cm_df["Predicted"]
        cm_plot = {}
        for key in cm_df["Actual-Predicted"].unique():
            cm_plot[key] = cm_df.loc[cm_df["Actual-Predicted"] == key].hvplot.box(
                y=cols, groupby=["Actual-Predicted"], ylabel="Probability"
            )
        # cm_plot = cm_df.hvplot.box(y=cols, groupby=["Actual-Predicted"], ylabel='Probability')
        return cm_plot

    def get_probability_distribution(self):
        """Gets probability distributions."""
        train_plot = self._get_probability_distribution(self.y_train, self.yhat_train)
        if self.has_test:
            test_plot = self._get_probability_distribution(self.y_test, self.yhat_test)
            cm_dict = {}
            cm_dict["Train Data"] = train_plot
            cm_dict["Test Data"] = test_plot
            train_plot = cm_dict
        return train_plot

    @fail_gracefully(_LOGGER)
    def roc_curve(self, **kwargs):
        """An interactive ROC curve for `class 1`."""
        train_plot = create_roc_curve(self.y_train, self.yhat_train, **kwargs).opts(
            title="Train Data", frame_width=350, frame_height=300
        )

        if self.has_test:
            test_plot = create_roc_curve(self.y_test, self.yhat_test, **kwargs).opts(
                title="Test Data", frame_width=350, frame_height=300
            )
            return (train_plot + test_plot).cols(2)
        return train_plot

    @fail_gracefully(_LOGGER)
    def threshold_curve(self, **kwargs):
        """Returns line plot with `precision recall`, `f1 score` and `prevalence` as `threshold` is varied.

        Visualizes how `precision`, `recall`, `f1 score`, and `prevalence` change as the
        `discrimination threshold` increases. For probabilistic, binary classifiers,
        the discrimination threshold is the probability at which you choose the
        positive class over the negative. Generally this is set to 50%, but
        adjusting the `discrimination threshold` will adjust sensitivity to false
        positives which is described by the inverse relationship of `precision` and
        `recall` with respect to the threshold.
        """
        train_plot = create_threshold_chart(
            self.y_train, self.yhat_train, **kwargs
        ).opts(title="Train Data", frame_width=350, frame_height=300)

        if self.has_test:
            test_plot = create_threshold_chart(
                self.y_test, self.yhat_test, **kwargs
            ).opts(title="Test Data", frame_width=350, frame_height=300)
            return (train_plot + test_plot).cols(2)
        return train_plot

    def _get_metrics_for_decision_threshold(self, threshold=0.5):
        metrics_dict = {}
        for metric in self.metrics:
            metric_details = self.metrics[metric]
            func = metric_details["func"]
            default_params = {}
            metrics_dict[metric] = {}
            if "default_params" in metric_details:
                default_params = metric_details["default_params"]

            dict_train_test = {}
            if self.has_train:
                dict_train_test["train"] = (self.y_train, self.yhat_train)
            if self.has_test:
                dict_train_test["test"] = (self.y_test, self.yhat_test)

            for key_, value_ in dict_train_test.items():
                actual_, predict_ = value_
                if self.multi_class:
                    predict_class = pd.Series(predict_.argmax(axis=1)).values
                else:
                    predict_class = (predict_ > threshold).astype(int)

                train_params = []
                if metric in ["log_loss", "roc_auc"]:
                    train_params.append(actual_)
                    train_params.append(predict_)
                else:
                    train_params.append(actual_)
                    train_params.append(predict_class)
                value = func(*train_params, **default_params)
                try:
                    value = round(value, metric_details["round"])
                except KeyError:
                    pass
                metrics_dict[metric][key_] = value
        # reform = {(outerKey, innerKey): values for outerKey, innerDict in metrics_dict.items() for innerKey, values in
        #           innerDict.items()}
        dict_of_df = {k: pd.DataFrame([v]) for k, v in metrics_dict.items()}
        metrics_df = pd.concat(dict_of_df, axis=1)
        metrics_df.columns.set_names(["metric", "dataset"], inplace=True)
        # metrics_df = pd.DataFrame([reform])
        return metrics_df

    @fail_gracefully(_LOGGER)
    def get_metrics(self, vary_thresholds=True, cutoff_value=0.5):
        """Returns a dataframe containing all the metrics used in classification modelling in tigerml."""
        if (
            hasattr(self.model, "predict_proba")
            and not self.multi_class
            and vary_thresholds
        ):
            metrics_dict = {}
            from tigerml.core.plots import hvPlot

            for thresh in range(5, 100, 5):
                metrics = self._get_metrics_for_decision_threshold(
                    threshold=thresh / 100
                )
                try:
                    metrics = (
                        metrics.transpose()
                        .reset_index(level=[1])
                        .pivot(columns="dataset")[0]
                        .reset_index()
                    )
                except Exception:
                    import pdb

                    pdb.set_trace()
                metrics_dict.update({thresh / 100: hvPlot(metrics).table()})
            plot = hv.HoloMap(metrics_dict, kdims=["decision_threshold"])
            plot.kdims[0].default = cutoff_value
            return plot
        else:
            return self._get_metrics_for_decision_threshold(threshold=cutoff_value)

    def _init_gains(self):
        if self.gains_table_result_train is None:
            self.gains_table_result_train = gains_table(self.y_train, self.yhat_train)
            if self.has_test:
                self.gains_table_result_test = gains_table(self.y_test, self.yhat_test)

    def gains_table(self):
        """Computes the gains table."""
        self._init_gains()
        plots_dict = {}
        plots_dict["train"] = [self.gains_table_result_train]
        if self.has_test:
            plots_dict["test"] = [self.gains_table_result_test]
        else:
            return [self.gains_table_result_train]
        return plots_dict

    def gains_chart(self, baseline=True, **kwargs):
        """Computes the gains chart."""
        self._init_gains()
        train_plot = gains_chart(
            self.gains_table_result_train, baseline=baseline, **kwargs
        ).opts(title="Train Data", frame_width=350, frame_height=300)
        if self.has_test:
            test_plot = gains_chart(
                self.gains_table_result_test, baseline=baseline, **kwargs
            ).opts(title="Test Data", frame_width=350, frame_height=300)
            return (train_plot + test_plot).cols(2)
        return train_plot

    @fail_gracefully(_LOGGER)
    def lift_chart(self, baseline=True, **kwargs):
        """Computes the lift chart."""
        self._init_gains()
        train_plot = lift_chart(
            self.gains_table_result_train, baseline=baseline, **kwargs
        ).opts(title="Train Data", frame_width=350, frame_height=300)
        if self.has_test:
            test_plot = lift_chart(
                self.gains_table_result_test, baseline=baseline, **kwargs
            ).opts(title="Test Data", frame_width=350, frame_height=300)
            return (train_plot + test_plot).cols(2)
        return train_plot

    @fail_gracefully(_LOGGER)
    def get_plots(self, cutoff_value=0.5):
        """Returns a dictionary of plots to be used for classification report."""
        plots_dict = {}
        if not self.multi_class:
            plots_dict["gains_table"] = format_tables_in_report(self.gains_table())
            plots_dict["confusion_matrix"] = [
                self.get_confusion_matrix(cutoff_value=cutoff_value)
            ]
        else:
            plots_dict["confusion_matrix"] = self.get_confusion_matrix_multiclass(
                cutoff_value=cutoff_value
            )
        for plot_ in self.plots.keys():
            if plot_ not in ["gains_table", "confusion_matrix"]:
                func = self.plots[plot_]
                if self.has_test:
                    plots_dict[plot_] = [func()]
                else:
                    plots_dict[plot_] = func()
        return plots_dict


class ClassificationComparisonMixin:
    """Classification comparison mixin class."""

    def perf_metrics(self, cutoff_value=0.5):
        """Returns a HTML table for the classification metrics for all the models given as list input.

        Parameters
        ----------
        cutoff_value : float, default=0.5
            Probability cutoff_value for class prediction.

        Returns
        -------
        Performance metrices table: HTMLTable
        """
        perf_metrics = pd.DataFrame()
        for model_name in self.reporters:
            current_metrics = self.reporters[model_name].evaluator.get_metrics(
                vary_thresholds=False, cutoff_value=cutoff_value
            )
            current_metrics.index = [model_name]
            perf_metrics = pd.concat([perf_metrics, current_metrics], axis=0)
        perf_metrics.columns = perf_metrics.columns.droplevel(level=1)  # no train test
        from tigerml.core.reports.html import HTMLTable, preset_styles

        # more_is_better_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "green"])
        # perf_metrics.style.background_gradient(cmap=more_is_better_cmap, axis=1, subset=['accuracy'])
        table = HTMLTable(perf_metrics)
        bad_metrics = ["log_loss"]
        table.apply_conditional_format(
            cols=[
                x
                for x in perf_metrics.columns
                if all([col not in x for col in bad_metrics])
            ],
            style=preset_styles.more_is_good_2colors,
        )
        table.apply_conditional_format(
            cols=[
                x
                for x in perf_metrics.columns
                if any([col in x for col in bad_metrics])
            ],
            style=preset_styles.less_is_good_2colors,
        )
        return table

    def roc_curves(self):
        """Returns roc_curves plot which contains the roc_curves associated with multiple models for comparison.

        Returns
        -------
        ROC curves plot: hvPlot
        """
        roc_curves = None
        for model_name in self.reporters:
            current_roc = self.reporters[model_name].evaluator.roc_curve(
                label=model_name
            )
            if roc_curves is None:
                roc_curves = current_roc
            else:
                roc_curves *= current_roc
        roc_curves.opts(legend_position="right")
        return roc_curves

    def confusion_matrices(self):
        """Returns dynamic confusion matrix as an interactive plot for all models for comparison.

        Returns
        -------
        Confusion matrix: hv.HoloMap
        """
        cm_matrices = {}
        for model_name in self.reporters:
            cm_matrix = self.reporters[model_name].evaluator.get_confusion_matrix()
            cm_matrix.opts(title=model_name)
            cm_matrices[model_name] = cm_matrix
        cm_dict = {}
        for thresh in range(5, 100, 5):
            cm_dict.update(
                {
                    thresh
                    / 100: hv.Layout(
                        [cm_matrices[model][thresh / 100] for model in cm_matrices]
                    )
                }
            )
        mat = hv.HoloMap(cm_dict, kdims=["decision_threshold"])
        mat.kdims[0].default = 0.5
        return mat

    def pr_curves(self):
        """Returns precision-recall curve plot associated with multiple models for comparison.

        Returns
        -------
        Precision-recall curves: hvPlot
        """
        pr_curves = None
        baseline = False
        for model_name in self.reporters:
            if model_name == list(self.reporters.keys())[-1]:
                baseline = True
            current_pr = self.reporters[model_name].evaluator.precision_recall_curve(
                label=model_name, baseline=baseline
            )
            if pr_curves is None:
                pr_curves = current_pr
            else:
                pr_curves *= current_pr
        pr_curves.opts(legend_position="right")
        return pr_curves

    def threshold_analysis(self):
        """Returns line plot with `precision recall`, `f1 score` and `prevalence` as `threshold` is varied for all the input models.

        Returns
        -------
        Threshold plot: hvPlot
        """
        th_plots = {}
        for model_name in self.reporters:
            th_plot = self.reporters[model_name].evaluator.threshold_curve()
            th_plot.opts(title=model_name)
            th_plots[model_name] = th_plot
        return th_plots

    def gains_charts(self):
        """Interactive Gains chart for all the input models.

        Returns
        -------
        Gains chart plot: hvPlot
        """
        gains_curves = None
        baseline = False
        for model_name in self.reporters:
            evaluator = self.reporters[model_name].evaluator
            if model_name == list(self.reporters.keys())[-1]:
                baseline = True
            current_gains_curve = evaluator.gains_chart(
                baseline=baseline, label=model_name
            )
            if gains_curves is None:
                gains_curves = current_gains_curve
            else:
                gains_curves *= current_gains_curve
        gains_curves.opts(legend_position="right")
        gains_curves.opts(title="")
        return gains_curves

    def lift_charts(self):
        """Interactive Lift chart for all the input models.

        Returns
        -------
        Lift chart plot: hvPlot
        """
        lift_curves = None
        baseline = False
        for model_name in self.reporters:
            evaluator = self.reporters[model_name].evaluator
            if model_name == list(self.reporters.keys())[-1]:
                baseline = True
            current_lift_curve = evaluator.lift_chart(
                baseline=baseline, label=model_name
            )
            if lift_curves is None:
                lift_curves = current_lift_curve
            else:
                lift_curves *= current_lift_curve
        lift_curves.opts(legend_position="right")
        lift_curves.opts(title="")
        return lift_curves

    def multi_classification_reports(self):
        """Computes the classification report table."""
        multi_cls_reports = {}
        for model_name in self.reporters:
            multi_cls_reports[model_name] = self.reporters[
                model_name
            ].evaluator.get_classification_report()
        return multi_cls_reports

    def multi_classification_distributions(self):
        """Computes the classification distributions table."""
        multi_cls_dists = {}
        for model_name in self.reporters:
            multi_cls_dists[model_name] = self.reporters[
                model_name
            ].evaluator.get_class_distributions()
        return multi_cls_dists

    def multi_classification_cm(self):
        """Computes the classification confusion_matrix table."""
        cm_matrices = {}
        for model_name in self.reporters:
            cm_matrices[model_name] = self.reporters[
                model_name
            ].evaluator.get_confusion_matrix_multiclass()
        return cm_matrices

    def get_performance_report(self, cutoff_value=0.5):
        """Return a consolidate dictionary contains classification specific comparative matrices values and different.

        performance plots for all the input models.

        Parameters
        ----------
        cutoff_value : float, default=0.5
            Probability cutoff_value for class prediction.

        Returns
        -------
        Models performance plots: dict
        """
        perf_dict = {}
        if not self.multi_class:
            perf_dict["performance_metrics"] = [
                self.perf_metrics(cutoff_value=cutoff_value)
            ]
            mat = self.confusion_matrices()
            perf_dict["confusion_matrices"] = [mat]
            perf_dict["gains_charts"] = self.gains_charts()
            perf_dict["lift_charts"] = self.lift_charts()
            perf_dict["roc_curves"] = self.roc_curves()
            perf_dict["precision_recall_curves"] = self.pr_curves()
            perf_dict["threshold_analysis"] = self.threshold_analysis()
        else:
            perf_dict["performance_metrics"] = [
                self.perf_metrics(cutoff_value=cutoff_value)
            ]
            perf_dict["confusion_matrices"] = [self.multi_classification_cm()]
            perf_dict["classification_reports"] = [self.multi_classification_reports()]
            perf_dict["class_distributions"] = [
                self.multi_classification_distributions()
            ]
        return perf_dict
