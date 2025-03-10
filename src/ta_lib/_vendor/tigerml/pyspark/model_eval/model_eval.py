"""Module for Model Evaluation and Interpretation."""

import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from functools import reduce
from matplotlib import pyplot as plt
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import (
    BinaryClassificationMetrics,
    MulticlassMetrics,
)
from pyspark.sql import DataFrame as PySparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as DT
from pyspark_dist_explore import hist
from tigerml.core.reports import create_report

from ..core.dp import custom_column_name, identify_col_data_type
from .handy_spark_cd import (
    BinaryClassificationMetrics as CustomBinaryClassificationMetrics,
)

sns.set()

_VALID_REGRESSION_METRICS_ = {
    "Explained Variance": "exp_var",
    "RMSE": "rmse",
    "MAE": "mae",
    "MSE": "mse",
    "MAPE": "mape",
    "WMAPE": "wmape",
    "R.Sq": "r2",
}


def get_regression_metrics(spark, data, y_col, y_pred_col, sig=2, data_type="Train"):
    """Generate the regression metrics for a model trained.

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data: pyspark.sql.DataFrame
        data with predictions for computing metrices
    y_col: str
        target column name in the data
    y_pred_col: str
        prediction column name in the data
    sig: float
        significance in terms of decimals for metrics as well as threshold
    data_type: str
        name of data type, e.g. 'train', 'test' etc.

    Returns
    -------
    df: pd.DataFrame
        dataframe containing metrics values for the data

    """
    df = defaultdict(list)
    metrics = _VALID_REGRESSION_METRICS_
    for metric in metrics.keys():
        df["Metric"].append(metric)
        if metrics[metric] == "wmape":
            val = wmape(spark, data, y_col, y_pred_col)
        elif metrics[metric] == "mape":
            val = mape(spark, data, y_col, y_pred_col)
        elif metrics[metric] == "exp_var":
            val = exp_var(spark, data, y_col, y_pred_col)
        else:
            e = RegressionEvaluator(labelCol=y_col, predictionCol=y_pred_col)
            val = e.evaluate(data, {e.metricName: metrics[metric]})
        df[data_type].append(round(val, sig))
    df = dict(df)
    df = pd.DataFrame(df)
    return df


def wmape(spark, data, y_col, y_hat_col):
    """Calculate the WMAPE in a data.

    Parameters
    ----------
        spark:
        data - spark.DataFrame
        y_col - str
        y_hat_col - str

    Returns
    -------
        wmape - numeric
    """
    wmape = (
        data.groupBy()
        .agg((F.sum(F.abs(F.col(y_hat_col) - F.col(y_col))) / F.sum(F.col(y_col))))
        .collect()[0][0]
    )
    return wmape


def mape(spark, data, y_col, y_hat_col):
    """Calculate the MAPE in a data.

    Parameters
    ----------
            spark: SparkSession
            data:  pyspark.sql.DataFrame
            y_col: str
            y_hat_col: str

    Returns
    -------
            mape: numeric
    """
    mape = (
        data.groupBy()
        .agg((F.mean(F.abs(F.col(y_hat_col) - F.col(y_col)) / F.col(y_col))))
        .collect()[0][0]
    )
    return mape


def exp_var(spark, data, y_col, y_hat_col):
    """Calculate the Explained Variance in a data.

    Parameters
    ----------
    spark: SparkSession
        Spark session object
    data: pyspark.sql.DataFrame
        Spark dataframe containing the data of actual and predicted target values
    y_col: str
        Actual target column name
    y_hat_col: str
        Predicted target column name

    Returns
    -------
    exp_var: float
        A scalar numeric value calculated as
        Explained Variance = 1 - (variance(y_col - y_hat_col) / variance(y_col))
    """
    un_exp_var = (
        data.groupby()
        .agg(
            F.pow(F.stddev(F.col(y_col) - F.col(y_hat_col)), 2)
            / F.pow(F.stddev(F.col(y_col)), 2)
        )
        .collect()[0][0]
    )
    if un_exp_var is None:
        un_exp_var = 0
    exp_var = 1 - un_exp_var
    return exp_var


def get_regression_plots(
    spark, data, y_col, y_pred_col, threshold=0.5, feature_cols=[]
):
    """Generate the regression plots for a model trained.

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data: pyspark.sql.DataFrame
        data with predictions
    y_col: str
        Column name of the target in the data
    y_pred_col: str
        Column name of the predictions in the data
    threshold: float, default=0.5
        Threshold in range [0,1] for identifying over-predictions/under-predictions
    feature_cols: list (str)
        feature columns to consider for feature level comparison of model performance

    Returns
    -------
    plots_dict: dict
        dictionary with plot names as key and plots as values
    """
    plots_dict = {}
    custom_residual = custom_column_name("residual", data.columns)
    custom_forecast_flag = custom_column_name("forecast_flag", data.columns)
    data = data.withColumn(custom_residual, F.col(y_pred_col) - F.col(y_col))
    data = data.withColumn(
        custom_forecast_flag,
        F.when((F.col(y_pred_col) > (1 + threshold) * F.col(y_col)), "Above threshold")
        .when((F.col(y_pred_col) < (1 - threshold) * F.col(y_col)), "Below threshold")
        .otherwise("Within threshold"),
    )

    # Residual Distribution Plot
    fig, ax = plt.subplots()
    hist(ax, [data.select(custom_residual)], bins=20)
    ax.set_title("Residual Histogram")
    plots_dict["Residual Histogram"] = plt.gcf()

    # Actual vs Predicted Scatter
    fig, ax = plt.subplots()
    for k, v in {
        "Above threshold": "royalblue",
        "Below threshold": "darkorange",
    }.items():
        plot_df = data.filter((F.col(custom_forecast_flag) == k))
        x = plot_df.select(y_col).collect()
        y = plot_df.select(y_pred_col).collect()
        ax.scatter(x=x, y=y, label=k, color=v, s=10)
    ax.set_xlabel(y_col)
    ax.set_ylabel(y_pred_col)
    ax.legend()
    ax.set_title(f"Actual vs Predicted with threshold={threshold}")
    plots_dict["Actual vs Predicted"] = plt.gcf()

    # Residual vs Predicted Scatter
    fig, ax = plt.subplots()
    for k, v in {
        "Above threshold": "royalblue",
        "Below threshold": "darkorange",
    }.items():
        plot_df = data.filter((F.col(custom_forecast_flag) == k))
        x = plot_df.select(y_pred_col).collect()
        y = plot_df.select(custom_residual).collect()
        ax.scatter(x=x, y=y, label=k, color=v, s=10)
    ax.set_xlabel(y_pred_col)
    ax.set_ylabel("Residual")
    ax.legend()
    ax.set_title(f"Residual vs Predicted with threshold={threshold}")
    plots_dict["Residual vs Predicted"] = plt.gcf()

    if len(feature_cols) != 0:
        # Plotting interaction plots
        fig, axs = plt.subplots(
            nrows=len(feature_cols), ncols=1, figsize=(18, 6 * len(feature_cols))
        )
        for idx, feature in enumerate(feature_cols):
            axs[idx] = plot_interaction(
                spark, data, feature, custom_residual, ax=axs[idx]
            )
            axs[idx].set_title(f"Interaction of {feature} with Residual")
        plots_dict["Feature Plots"] = [plt.gcf()]

    return plots_dict


# FIX ME
# Interactive y,yhat plot
# How can we get this in pyspark

# -----------------------------------------------------------------------
# Classification - Individual Model (WIP)
# -----------------------------------------------------------------------
_BINARY_CLASSIFICATION_METRICS_ = {
    "Accuracy": "accuracy",
    "F1 Score": "f1",
    "TPR": "tpr",
    "FPR": "fpr",
    "Precision": "precision",
    "Recall": "recall",
    "AuROC": "auROC",
    "AuPR": "auPR",
}


def get_classification_scores(
    data, y_col, probability_col=None, y_pred_col=None, threshold=0.5
):
    """Getting confusion matrix and area under RoC & PR curves for a data with predictions.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
        data having actual and predicted values
    y_col: str
        actual values column name
    probability_col: str
        column name which have predicted probabilities
    threshold: float
        threshold for computing classes from probability of positive class

    Returns
    -------
    conf_matrix, auROC, auPR
        confusion matrix, area under RoC curve and area under Precision-Recall curve
    """
    if probability_col is None and y_pred_col is None:
        raise ValueError(
            "For computing classification scores, pass a column name having either predicted "
            "probabilities (probability_col) or classes (y_pred_col)."
        )
    if probability_col is not None:
        proboScoreAndLabels = data.select(probability_col, y_col).rdd.map(
            lambda row: (float(row[probability_col][1]), float(row[y_col]))
        )
        scoreAndLabels = proboScoreAndLabels.map(
            lambda t: (float(t[0] > threshold), t[1])
        )
    else:
        scoreAndLabels = data.select(y_pred_col, y_col).rdd.map(
            lambda row: (float(row[y_pred_col]), float(row[y_col]))
        )
        proboScoreAndLabels = scoreAndLabels
    conf_matrix = MulticlassMetrics(scoreAndLabels).confusionMatrix().toArray()
    binary_cls_obj = BinaryClassificationMetrics(proboScoreAndLabels)
    auROC = binary_cls_obj.areaUnderROC
    auPR = binary_cls_obj.areaUnderPR
    return conf_matrix, auROC, auPR


def get_binary_classification_metrics(
    spark,
    data,
    y_col,
    probability_col=None,
    y_pred_col=None,
    threshold=0.50,
    sig=2,
    data_type="Train",
):
    """Get the classification metrics for a model trained.

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data: pyspark.sql.DataFrame
        data with target and predicted probabilities
    y_col: str
            column name of actuals/target in the data
    probability_col: str
            column name contain predicted probabilities
    threshold: float, default=0.5
            threshold in range of [0, 1] to consider for calculation of metrics
    sig: int
            significance in terms of decimals for metrics as well as threshold

    Returns
    -------
    metrics: pd.DataFrame
        dataframe having different classification metric values for given data
    """
    relevant_cols = [
        col for col in [y_col, y_pred_col, probability_col] if col is not None
    ]
    data = data.select(*relevant_cols)
    df = defaultdict(list)
    metrics = _BINARY_CLASSIFICATION_METRICS_
    for metric in metrics.keys():
        df["Metric"].append(metric)

    conf_matrix, auROC, auPR = get_classification_scores(
        data, y_col, probability_col, y_pred_col, threshold=threshold
    )
    tp = conf_matrix[1][1]
    fp = conf_matrix[0][1]
    tn = conf_matrix[0][0]
    fn = conf_matrix[1][0]

    accuracy = (tp + tn) / (tp + fp + tn + fn)  # noqa
    tpr = tp / (tp + fn)  # noqa
    fpr = fp / (fp + tn)  # noqa
    precision = tp / (tp + fp)  # noqa
    recall = tp / (tp + fn)  # noqa
    f1 = 2 * precision * recall / (precision + recall)  # noqa

    for metric in df["Metric"]:
        val = round(eval(metrics[metric]), sig)  # noqa
        df[data_type].append(val)
    df = pd.DataFrame(df)
    return df


def get_binary_classification_plots(
    spark,
    data,
    y_col,
    y_pred_col,
    probability_col,
    threshold=0.5,
    feature_cols=[],
):
    """Get the binary classification plots for a model trained.

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data: pyspark.sql.DataFrame
        data having actual and predicted probabilities and/or predicted classes
    y_col: str
            column name of actuals in the data
    y_pred_col: str
            column of prediction label in the data
    probability_cols: str
            column name of probability column in the data
    feature_cols: list(str)
            columns we consider for feature level comparison of the performance, the same must be present in the data
    threshold: float, default=0.5
            threshold to define the class for calculation of metrics
    sig: int
            significance in terms of decimals for metrics as well as threshold

    Returns
    -------
    plots_dict: dict
        dictionary which having different classification plots
    """
    conf_matrix, auROC, auPR = get_classification_scores(
        data, y_col, probability_col, y_pred_col, threshold=threshold
    )

    plots_dict = {}
    if probability_col is not None:
        # ROC and PR Curve
        bcm = CustomBinaryClassificationMetrics(
            data, scoreCol=probability_col, labelCol=y_col
        )
        fig, ax = plt.subplots()
        bcm.plot_roc_curve(ax=ax)
        plots_dict["ROC Curve"] = plt.gcf()
        fig, ax = plt.subplots()
        bcm.plot_pr_curve(ax=ax)
        plots_dict["Precision-Recall Curve"] = plt.gcf()

    # Confusion Matrix
    cm_df = pd.DataFrame(conf_matrix)
    cm_df.rename(
        columns={c: "Predicted_" + str(c) for c in cm_df.columns}, inplace=True
    )
    cm_df.rename(index={i: "Actual_" + str(i) for i in cm_df.index}, inplace=True)
    cm_df = cm_df.reset_index().rename(columns={"index": "#"})
    plots_dict["Confusion Matrix"] = cm_df

    # Feature Plots
    if len(feature_cols) != 0:
        # Plotting Interaction Plots with Confusion Cell
        data = generate_confusion_cell_col(
            spark,
            data,
            y_col,
            probability_col,
            y_pred_col,
            confusion_cell_col="confusion_matrix_cell",
            threshold=threshold,
        )
        fig, axs = plt.subplots(
            nrows=len(feature_cols), ncols=1, figsize=(18, 6 * len(feature_cols))
        )
        for idx, feature in enumerate(feature_cols):
            axs[idx] = plot_interaction(
                spark, data, feature, "confusion_matrix_cell", ax=axs[idx]
            )
        plots_dict["Feature Plots"] = [plt.gcf()]
    return plots_dict


def generate_confusion_cell_col(
    spark,
    data,
    y_col,
    probability_col,
    y_pred_col,
    confusion_cell_col="confusion_matrix_cell",
    threshold=0.5,
):
    """Column to Generate the column to define the cell of confusion matrix the row belongs to.

    Paramters:
    ----------
    spark: SparkSession
        spark session object
    data: pyspark.sql.DataFrame
        data having actual and predicted probabilities column
    y_col: str
        actuals column name
    probability_col: str
        column name having predicted probabilities
    y_pred_col: str
        column name having predicted classes
    threshold: float, default=0.5
        probability threshold for calculating the prediction labels.

    Returns
    -------
    df - pyspark.sql.DataFrame
        input data with a new column defined for classification type value (TP, TN, FP or FN)
    """

    def _get_label(probabilities):
        if probabilities[1] > threshold:
            return 1.0
        else:
            return 0.0

    def _get_conf_cell(pred_label, actual_label):
        if pred_label == 1:
            if actual_label == 1:
                return "TP"
            else:
                return "FP"
        else:
            if actual_label == 0:
                return "TN"
            else:
                return "FN"

    _get_label = F.udf(_get_label, DT.DoubleType())
    _get_conf_cell = F.udf(_get_conf_cell, DT.StringType())

    custom_pred_label = custom_column_name("pred_label", data.columns)
    if probability_col is not None:
        data = data.withColumn(custom_pred_label, _get_label(F.col(probability_col)))
        data = data.withColumn(
            confusion_cell_col, _get_conf_cell(F.col(custom_pred_label), F.col(y_col))
        )
    else:
        data = data.withColumn(
            confusion_cell_col, _get_conf_cell(F.col(y_pred_col), F.col(y_col))
        )

    return data


def plot_interaction(spark, data, col1, col2, ax):
    """Plot the interaction b/w 2 columns in a dataframe.

    Plots based on column types:
            continuous vs continuous - scatter plot
            continuous vs categorical - distribution plot
            categorical vs categorical - stacked bar plot

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data - pyspark.sql.DataFrame
        data having col1 and col2
    col1 - str
        a feature/performance column in the data
    col2 - str
        a feature/performance column in the data
    ax - matplotlib axis
        axis to draw plot

    Returns
    -------
    ax - matplotlib axis
        axis with plot
    """
    col1_type = identify_col_data_type(data, col1)
    col2_type = identify_col_data_type(data, col2)

    if (col1_type == "date_like") | (col2_type == "date_like"):
        raise (
            NotImplementedError("Datelike column interactions not applicable as of now")
        )

    if (col1_type != "numerical") & (col2_type != "numerical"):
        ax = stacked_bar_plot(spark, data, col1, col2, ax)
    elif col1_type != "numerical":
        ax = distribution_plot(spark, data, cat_col=col1, cont_col=col2, ax=ax)
    elif col2_type != "numerical":
        ax = distribution_plot(spark, data, cat_col=col2, cont_col=col1, ax=ax)
    else:
        ax = scatter_plot(spark, data, col1=col2, col2=col1, ax=ax)

    return ax


def scatter_plot(spark, data, col1, col2, ax):
    """Generate a scatter plot for interaction b.w 2 continuous columns.

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data - pyspark.sql.DataFrame
        data having col1 and col2
    col1 - str
        a feature/performance column in the data
    col2 - str
        a feature/performance column in the data
    ax - matplotlib axis
        axis to draw plot

    Returns
    -------
    ax - matplotlib axis
        axis with plot
    """
    x = data.select(col1).collect()
    y = data.select(col2).collect()
    ax.scatter(x=x, y=y, marker="o")  # , ax=ax)
    ax.set_title(f"Scatter Plot of {col2} vs {col1}")
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    return ax


def distribution_plot(spark, data, cat_col, cont_col, ax):
    """Generate a  distribution plot for interaction b.w a categorical and continuous variable.

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data - pyspark.sql.DataFrame
        data having col1 and col2
    col1 - str
        a feature/performance column in the data
    col2 - str
        a feature/performance column in the data
    ax - matplotlib axis
        axis to draw plot
    Returns
    -------
    ax - matplotlib axis
        axis with plot
    """
    df = data.select(cat_col, cont_col).toPandas()
    df.groupby(cat_col)[cont_col].plot(kind="density", legend=True, ax=ax)
    ax.set_title("Density Plot(" + cont_col + ")")
    return ax


def stacked_bar_plot(spark, data, col1, col2, ax):
    """Generate a scatter plot for interaction b.w a categorical and continuous variable.

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data - pyspark.sql.DataFrame
        data having col1 and col2
    col1 - str
        a feature/performance column in the data
    col2 - str
        a feature/performance column in the data
    ax - matplotlib axis
        axis to draw plot

    Returns
    -------
    ax - matplotlib axis
        axis with plot
    """
    df = data.groupBy(col1, col2).agg(F.count(F.col(col1)).alias("count")).toPandas()
    df = (
        (df.groupby([col1, col2])["count"].sum() / df.groupby(col1)["count"].sum())
        * 100
    ).reset_index()
    df = df.pivot(index=col1, columns=col2, values="count")
    df.plot(kind="bar", ax=ax)
    ax.set_title("Stacked Bar Plot")
    ax.set_xlabel(col1)
    return ax


def get_regression_report(
    spark,
    data_train,
    y_col,
    y_pred_col,
    data_test=None,
    sig=2,
    threshold=0.5,
    feature_cols=[],
    report_name=None,
    report_format=".html",
):
    """Generate a report for regression model with metrics and plots.

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data_train: pyspark.sql.DataFrame
        training data
    y_col: str
        name of target column in the data
    y_pred_col: str
        name of predicted values column in the data
    data_test: pyspark.sql.DataFrame, default=None
        testing data
    sig: int
        significance in terms of decimals for metrics as well as threshold
    threshold: float, default=0.5
        threshold in range of [0, 1] for defining under and over predictions
    feature_cols: list (str)
        name of feature columns for comparing with residuals
    report_name: str
        report name (with path) without type e.g. "dir_path/regression_report"
    report_format: str
        report format, one of from ".html", ".xlsx" and ".pdf"

    Returns
    -------
    generate and save the report at specified location
    """
    report_dict = {}
    metrics_df = get_regression_metrics(
        spark=spark,
        data=data_train,
        y_col=y_col,
        y_pred_col=y_pred_col,
        sig=sig,
        data_type="Train",
    )
    if data_test is not None:
        test_metrics_df = get_regression_metrics(
            spark=spark,
            data=data_test,
            y_col=y_col,
            y_pred_col=y_pred_col,
            sig=sig,
            data_type="Test",
        )
        metrics_df = pd.merge(metrics_df, test_metrics_df, how="outer", on="Metric")
    report_dict["Metrices"] = metrics_df
    plots_dict = get_regression_plots(
        spark=spark,
        data=data_train,
        y_col=y_col,
        y_pred_col=y_pred_col,
        threshold=threshold,
        feature_cols=feature_cols,
    )
    if data_test is not None:
        test_plots_dict = get_regression_plots(
            spark=spark,
            data=data_test,
            y_col=y_col,
            y_pred_col=y_pred_col,
            threshold=threshold,
            feature_cols=feature_cols,
        )
        for key, value in test_plots_dict.items():
            plots_dict[key] = {"Train": plots_dict[key], "Test": value}
    report_dict["Plots"] = plots_dict
    if report_name is None:
        report_name = "pyspark_regression_report_at_{}".format(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    create_report(report_dict, name=report_name, format=report_format)


def get_binary_classification_report(
    spark,
    data_train,
    y_col,
    y_pred_col,
    probability_col,
    data_test=None,
    threshold=0.50,
    sig=2,
    feature_cols=[],
    report_name=None,
    report_format=".html",
):
    """Generate a report for binary classification model with metrics and plots.

    Parameters
    ----------
    spark: SparkSession
        spark session object
    data_train: pyspark.sql.DataFrame
        training data
    y_col: str
        name of target column in the data
    y_pred_col: str
        name of predicted class column in the data
    probability_col: str
        name of predicted probabilities column in the data
    data_test: pyspark.sql.DataFrame, default=None
        testing data
    threshold: float, default=0.5
        threshold in range of [0, 1] for defining class from probability of positive class
        Note: For option 2, this will applicable only when probabilities are passed in the data.
    sig: int
        significance in terms of decimals for metrics as well as threshold
    feature_cols: list (str)
        name of feature columns for comparing with residuals
    report_name: str
        report name (with path) without type e.g. "dir_path/regression_report"
    report_format: str
        report format, one of from ".html", ".xlsx" and ".pdf"

    Returns
    -------
    generate and save the report at specified location
    """
    report_dict = {}
    metrics_df = get_binary_classification_metrics(
        spark=spark,
        data=data_train,
        y_col=y_col,
        y_pred_col=y_pred_col,
        probability_col=probability_col,
        threshold=threshold,
        sig=sig,
        data_type="Train",
    )
    if data_test is not None:
        test_metrics_df = get_binary_classification_metrics(
            spark=spark,
            data=data_test,
            y_col=y_col,
            y_pred_col=y_pred_col,
            probability_col=probability_col,
            threshold=threshold,
            sig=sig,
            data_type="Test",
        )
        metrics_df = pd.merge(metrics_df, test_metrics_df, how="outer", on="Metric")
    report_dict["Metrices"] = metrics_df
    plots_dict = get_binary_classification_plots(
        spark=spark,
        data=data_train,
        y_col=y_col,
        y_pred_col=y_pred_col,
        probability_col=probability_col,
        threshold=threshold,
        feature_cols=feature_cols,
    )
    if data_test is not None:
        test_plots_dict = get_binary_classification_plots(
            spark=spark,
            data=data_test,
            y_col=y_col,
            y_pred_col=y_pred_col,
            probability_col=probability_col,
            threshold=threshold,
            feature_cols=feature_cols,
        )
        for key, value in test_plots_dict.items():
            plots_dict[key] = {"Train": plots_dict[key], "Test": value}
    report_dict["Plots"] = plots_dict
    if report_name is None:
        report_name = "pyspark_binary_classification_report_at_{}".format(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    create_report(report_dict, name=report_name, format=report_format)


class PySparkReport:
    """Base class for pyspark report utilities."""

    def __init__(
        self,
        is_classification,
        spark,
        train_df,
        label_Col,
        model=None,
        features_Col="features",
        prediction_Col="prediction",
        ignore_cols=[],
        test_df=None,
    ):
        self.is_classification = is_classification
        self.spark = spark
        self.train_df = train_df
        self.label_Col = label_Col
        self.model = model
        self.features_Col = features_Col
        self.prediction_Col = prediction_Col
        self.ignore_cols = ignore_cols
        self.test_df = test_df
        self._validate_inputs()
        self._set_inputs()
        if self.model is not None:
            self._compute_predictions()

    def _validate_inputs(self):
        if (
            not isinstance(self.train_df, PySparkDataFrame)
            and not isinstance(self.train_df, pd.DataFrame)
        ) or (
            self.test_df is not None
            and not isinstance(self.test_df, PySparkDataFrame)
            and not isinstance(self.test_df, pd.DataFrame)
        ):
            raise ValueError(
                "Input data should be either an instance of pandas.DataFrame or pyspark.sql.DataFrame."
            )
        if self.model is None:
            if not self.is_classification:
                if (
                    self.features_Col not in self.train_df.columns
                    or self.prediction_Col not in self.train_df.columns
                ):
                    raise ValueError(
                        "In regression, for option 2, train/test data must contains features column (features_Col) "
                        "and prediction column (prediction_Col)."
                    )
            elif self.features_Col not in self.train_df.columns or (
                self.prediction_Col not in self.train_df.columns
                and self.probability_col not in self.train_df.columns
            ):
                raise ValueError(
                    "In classification, for option 2, train/test data must contains features column (features_Col) "
                    "and one of predicted class column (prediction_Col) or predicted probabilities column (probability_col)."
                )
        elif "pyspark" not in str(type(self.model)):
            raise ValueError("Please pass a valid instance of pyspark model!")

    def _set_inputs(self):
        if isinstance(self.train_df, pd.DataFrame):
            self.train_df = self.spark.createDataFrame(self.train_df)
        if isinstance(self.test_df, pd.DataFrame):
            self.test_df = self.spark.createDataFrame(self.test_df)

    def _compute_predictions(self):
        feature_columns = [
            col
            for col in self.train_df.columns
            if col not in [self.label_Col] + self.ignore_cols
        ]
        assembler = VectorAssembler(
            inputCols=feature_columns, outputCol=self.features_Col
        )
        self.train_df = assembler.transform(self.train_df)
        if self.test_df is not None:
            self.test_df = assembler.transform(self.test_df)
        col_names = {
            self.features_Col: "features",
            self.label_Col: "label",
            self.prediction_Col: "prediction",
        }
        self.train_df = reduce(
            lambda temp, idx: temp.withColumnRenamed(
                list(col_names.keys())[idx], list(col_names.values())[idx]
            ),
            range(len(list(col_names.keys()))),
            self.train_df,
        )
        if self.test_df is not None:
            self.test_df = reduce(
                lambda temp, idx: temp.withColumnRenamed(
                    list(col_names.keys())[idx], list(col_names.values())[idx]
                ),
                range(len(list(col_names.keys()))),
                self.test_df,
            )
        self.model = self.model.fit(self.train_df)
        self.train_df = self.model.transform(self.train_df)
        if self.test_df is not None:
            self.test_df = self.model.transform(self.test_df)
        self.features_Col = "features"
        self.label_Col = "label"
        self.prediction_Col = "prediction"
        if self.is_classification:
            self.probability_col = "probability"

    def _get_report(
        self,
        threshold=0.50,
        sig=2,
        feature_cols=[],
        report_name=None,
        report_format=".html",
    ):
        if self.is_classification:
            get_binary_classification_report(
                spark=self.spark,
                data_train=self.train_df,
                y_col=self.label_Col,
                y_pred_col=self.prediction_Col,
                probability_col=self.probability_col,
                data_test=self.test_df,
                threshold=threshold,
                sig=sig,
                feature_cols=feature_cols,
                report_name=report_name,
                report_format=report_format,
            )
        else:
            get_regression_report(
                spark=self.spark,
                data_train=self.train_df,
                y_col=self.label_Col,
                y_pred_col=self.prediction_Col,
                data_test=self.test_df,
                threshold=threshold,
                sig=sig,
                feature_cols=feature_cols,
                report_name=report_name,
                report_format=report_format,
            )
        print("Report generated and saved successfully!")


class RegressionReport(PySparkReport):
    """PySpark model evaluation toolkit for regression problems.

    Generate report for model performance.

    There are two options available:

        * Option 1: Using model object.
            - Must have pyspark model object (model), train data (train_df) and target column name (label_Col).
        * Option 2: Using predictors
            - Must have train data (train_df) which should have target column (label_Col), features column (features_Col)
              and prediction column (prediction_Col)

    Parameters
    ----------
    spark: SparkSession
        spark session object

    train_df: pyspark.sql.DataFrame or pandas.DataFrame
        training data

    label_Col: str
        name of target column in the data

    model : pyspark model object, default=None
        Model object from pyspark.ml with methods `fit` and `transform` (applicable for option 1)

    features_Col: str, default='features'
        name of column which define the features (applicable for option 2)

    prediction_Col: str, default='prediction',
        name of column having predicted values (applicable for option 2)

    ignore_cols: list,
        list of columns to be ignored while creating features from the data (applicable for option 1)

    test_df: pyspark.sql.DataFrame or pandas.DataFrame, default=None
        testing data, if passed evaluation would perform for the same and will include in the report along with train data

    Examples
    --------
    >>> import pandas as pd
    >>> from pyspark.sql import SparkSession
    >>> from pyspark.ml.regression import RandomForestRegressor
    >>> from sklearn.datasets import fetch_california_housing
    >>> from tigerml.pyspark.model_eval import RegressionReport
    >>> from tigerml.pyspark.dp import test_train_split
    >>> spark = SparkSession.builder.appName('model_eval_regression').getOrCreate()
    >>> data = fetch_california_housing()
    >>> X = pd.DataFrame(data['data'], columns=data['feature_names'])
    >>> y = pd.DataFrame(data['target'], columns=['Target'])
    >>> df = spark.createDataFrame(pd.concat([X,y], axis=1))
    >>> df_train, df_test = test_train_split(spark, data=df, target_col="Target", train_prop=0.7, random_seed=1234, target_type='continuous')

    >>> # Option 1 - with model
    >>> reg_obj1 = RegressionReport(spark=spark, train_df=df_train, label_Col='Target', model=RandomForestRegressor(), test_df=df_test)
    >>> reg_obj1.get_report(feature_cols=['CRIM','ZN','AGE'])

    >>> # Option 2 - without model
    >>> feature_cols = [col for col in df.columns if col not in ['Target']]
    >>> assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    >>> train_df = assembler.transform(train_df)
    >>> test_df = assembler.transform(test_df)
    >>> m = RandomForestRegressor(featuresCol="features", labelCol="Target", predictionCol='yhat')
    >>> model = m.fit(df_train)
    >>> df_train = model.transform(df_train)
    >>> df_test = model.transform(df_test)
    >>> reg_obj2 = RegressionReport(spark=spark, train_df=df_train, label_Col='Target', features_Col='features', prediction_Col='yhat', test_df=df_test)
    >>> reg_obj2.get_report(feature_cols=['CRIM','ZN','AGE'])
    """

    def __init__(
        self,
        spark,
        train_df,
        label_Col,
        model=None,
        features_Col="features",
        prediction_Col="prediction",
        ignore_cols=[],
        test_df=None,
    ):
        super().__init__(
            is_classification=False,
            spark=spark,
            train_df=train_df,
            label_Col=label_Col,
            model=model,
            features_Col=features_Col,
            prediction_Col=prediction_Col,
            ignore_cols=ignore_cols,
            test_df=test_df,
        )

    def get_report(
        self,
        threshold=0.50,
        sig=2,
        feature_cols=[],
        report_name=None,
        report_format=".html",
    ):
        """Generate a report for regression model with metrics and plots.

        Parameters
        ----------
        threshold: float, default=0.5
            threshold in range of [0, 1] for defining under and over predictions
        sig: int
            significance in terms of decimals for metrics as well as threshold
        feature_cols: list (str)
            name of feature columns for comparing with residuals
        report_name: str
            report name (with path) without type e.g. "dir_path/regression_report"
        report_format: str
            report format, one of from ".html", ".xlsx" and ".pdf"

        Returns
        -------
        generate and save the report at specified location
        """
        super()._get_report(
            threshold=threshold,
            sig=sig,
            feature_cols=feature_cols,
            report_name=report_name,
            report_format=report_format,
        )


class ClassificationReport(PySparkReport):
    """PySpark model evaluation toolkit for classification problems.

    Generate report for model performance.

    There are two options available:

        * Option 1: Using model object.
            - Must have pyspark model object (model), train data (train_df) and target column name (label_Col).
        * Option 2: Using predictors
            - Must have train data (train_df) which should have target column (label_Col), features column (features_Col)
              and one of predicted class column (prediction_Col) or predicted probabilities column (probability_col)

    Parameters
    ----------
    spark: SparkSession
        spark session object

    train_df: pyspark.sql.DataFrame or pandas.DataFrame
        training data

    label_Col: str
        name of target column in the data

    model : pyspark model object, default=None
        Model object from pyspark.ml with methods `fit` and `transform` (applicable for option 1)

    features_Col: str, default='features'
        name of column which define the features (applicable for option 2)

    prediction_Col: str, default='prediction',
        name of column having predicted class values (applicable for option 2)

    probability_col: str, default='probability'
        name of column having predicted probability values (applicable for option 2)

    ignore_cols: list,
        list of columns to be ignored while creating features from the data (applicable for option 1)

    test_df: pyspark.sql.DataFrame or pandas.DataFrame, default=None
        testing data, if passed evaluation would perform for the same and will include in the report along with train data

    Examples
    --------
    >>> import pandas as pd
    >>> from pyspark.sql import SparkSession
    >>> from pyspark.ml.classification import LogisticRegression
    >>> from sklearn.datasets import load_breast_cancer
    >>> from tigerml.pyspark.model_eval import ClassificationReport
    >>> from tigerml.pyspark.dp import test_train_split
    >>> spark = SparkSession.builder.appName('model_eval_classification').getOrCreate()
    >>> data = load_breast_cancer()
    >>> X = pd.DataFrame(data['data'], columns=data['feature_names'])
    >>> y = pd.DataFrame(data['target'], columns=['Target'])
    >>> df = spark.createDataFrame(pd.concat([X,y], axis=1))
    >>> df_train, df_test = test_train_split(spark, data=df, target_col="Target", train_prop=0.7, random_seed=1234, target_type='categorical')

    >>> # Option 1 - with model
    >>> cls_obj1 = ClassificationReport(spark=spark, train_df=df_train, label_Col='Target', model=LogisticRegression(), test_df=df_test)
    >>> cls_obj1.get_report(feature_cols=['mean radius', 'mean texture'])

    >>> # Option 2 - without model
    >>> feature_cols = [col for col in df.columns if col not in ['Target']]
    >>> assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    >>> train_df = assembler.transform(train_df)
    >>> test_df = assembler.transform(test_df)
    >>> m = LogisticRegression(featuresCol="features", labelCol="Target", predictionCol='yhat')
    >>> model = m.fit(df_train)
    >>> df_train = model.transform(df_train)
    >>> df_test = model.transform(df_test)
    >>> cls_obj2 = ClassificationReport(spark=spark, train_df=df_train, label_Col='Target', features_Col='features', prediction_Col='yhat', probability_col='probability', test_df=df_test)
    >>> cls_obj2.get_report(feature_cols=['mean radius', 'mean texture'])

    """

    def __init__(
        self,
        spark,
        train_df,
        label_Col,
        model=None,
        features_Col="features",
        prediction_Col="prediction",
        probability_col="probability",
        ignore_cols=[],
        test_df=None,
    ):
        self.probability_col = probability_col
        super().__init__(
            is_classification=True,
            spark=spark,
            train_df=train_df,
            label_Col=label_Col,
            model=model,
            features_Col=features_Col,
            prediction_Col=prediction_Col,
            ignore_cols=ignore_cols,
            test_df=test_df,
        )

    def get_report(
        self,
        threshold=0.50,
        sig=2,
        feature_cols=[],
        report_name=None,
        report_format=".html",
    ):
        """Generate a report for binary classification model with metrics and plots.

        Parameters
        ----------
        threshold: float, default=0.5
            threshold in range of [0, 1] for defining class from probability of positive class
            Note: For option 2, this will applicable only when probabilities are passed in the data.
        sig: int
            significance in terms of decimals for metrics as well as threshold
        feature_cols: list (str)
            name of feature columns for comparing with residuals
        report_name: str
            report name (with path) without type e.g. "dir_path/regression_report"
        report_format: str
            report format, one of from ".html", ".xlsx" and ".pdf"

        Returns
        -------
        generate and save the report at specified location
        """
        super()._get_report(
            threshold=threshold,
            sig=sig,
            feature_cols=feature_cols,
            report_name=report_name,
            report_format=report_format,
        )
