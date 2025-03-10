import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error
from tigerml.core.utils import flatten_list


def compute_residual(y, yhat):
    return y - yhat


def mape(y_true, y_pred):
    """Computes Mean Absolute Percent error.

    Ignore values with y_true=0

    Parameters
    ----------
    y_true: actual values
    y_hat: predicted values
    """
    yp = y_pred[y_true != 0]
    yt = y_true[y_true != 0]
    return np.mean(np.abs((yt - yp) / yt))


def wmape(y_true, y_pred):
    """Computes Weighted Mean Absolute Deviation.

    Similar to mape but weighted by y_true.
    This ignores y_true=0 and tends to be higher
    if the errors are higher for higher y_true values.

    Parameters
    ----------
    y_true: actual values
    y_hat: predicted values
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def root_mean_squared_error(*args, **kwargs):
    """Computes Root Mean Squared Error.

    Parameters
    ----------
    y_true: actual values
    y_hat: predicted values
    """
    return pow(mean_squared_error(*args, **kwargs), 0.5)


def confusion_matrix_df(
    y_train, yhat_train, y_test=None, yhat_test=None, normalized=False, flattened=False
):
    """Returns the confusion matrix as dataframe.

    Parameters
    ----------
    y_train: pd.DataFrame, array-like of shape (n_samples,)
        The training target values.
    yhat_train: pd.DataFrame, array-like of shape (n_samples,)
        The training predicted target values.
        Note: for classification, it needs to be the probabilities of positive class
    y_test: pd.DataFrame, array-like of shape (n_samples,), default=None
        The testing target values.
    yhat_test: pd.DataFrame, array-like of shape (n_samples,), default=None
        The testing predicted target values.
        Note: for classification, it needs to be the probabilities of positive class
    normalized: bool, default=False
        whether to normalize the confusion matrix values
    flattened: bool, default=False
        if the confusion matrix needs to be flattened out

    Returns
    -------
    confusion_matrix: pd.DataFrame
        confusion matrix as a dataframe
    """
    cms = {}
    cm_dfs = {}
    yhat_train = (yhat_train > 0.5).astype(int)
    cms["train"] = confusion_matrix(y_train, yhat_train)
    if y_test is not None and yhat_test is not None:
        yhat_test = (yhat_test > 0.5).astype(int)
        cms["test"] = confusion_matrix(y_test, yhat_test)
    for dataset in cms:
        if normalized:
            cms[dataset] = (
                cms[dataset].astype("float") / cms[dataset].sum(axis=1)[:, np.newaxis]
            )
        cm_dfs[dataset] = pd.DataFrame(cms[dataset])
        cm_dfs[dataset] = cm_dfs[dataset].rename(
            columns=dict(
                zip(
                    [x for x in cm_dfs[dataset].columns],
                    [
                        "predicted_{}{}".format(x, "_normalized" if normalized else "")
                        for x in cm_dfs[dataset].columns
                    ],
                )
            )
        )
    if flattened:
        classes = cm_dfs["train"].index.values
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
            for dataset in cms:
                flat_cm_dict[label][dataset] = cm_dfs[dataset].values.ravel()[index]
        dict_of_df = {k: pd.DataFrame([v]) for k, v in flat_cm_dict.items()}
        flat_cm_df = pd.concat(dict_of_df, axis=1)
        return flat_cm_df
    cm_df = pd.concat({dataset: cm_dfs[dataset] for dataset in cms}, axis=1)
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
