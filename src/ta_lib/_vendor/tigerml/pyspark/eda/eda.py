"""Functions to carry out the EDA in spark framework."""

import gc
import holoviews as hv
import logging  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from hvplot import hvPlot
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import functions as F
from pyspark.sql.functions import col, desc, lit, percentile_approx
from pyspark_dist_explore import distplot
from tigerml.core.plots.bokeh import add_to_secondary, finalize_axes_right
from tigerml.core.reports import create_report
from tigerml.pyspark.core import dp
from tigerml.pyspark.core.dp import (
    custom_column_name,
    identify_col_data_type,
    list_boolean_columns,
    list_categorical_columns,
    list_datelike_columns,
    list_numerical_categorical_columns,
    list_numerical_columns,
)
from tigerml.pyspark.core.utils import (
    append_file_to_path,
    flatten_list,
    time_now_readable,
)

_LOGGER = logging.getLogger(__name__)


def setanalyse(df1, df2, col, simplify=True, exceptions_only=False):
    """
    Given two spark dataframes, returns a dictionary of set analysis.

    A-B: set(A) - set(B)
    B-A: set(B) - set(A)
    AuB: A union B
    A^B: A intersection B

    Parameters
    ----------
        df1, df2: spark dataframes to be evaluated
        exceptions_only: 'False'(default):
            if True, gives only A-B & B-A. False gives all 4.
            True is efficient while dealing with large sets or analyzing exceptions alone.

    """
    A = set(df1.select(col).rdd.map(lambda r: r).collect())
    B = set(df2.select(col).rdd.map(lambda r: r).collect())
    output = {"A-B": A - B, "B-A": B - A}
    if ~exceptions_only:
        output["AuB"] = A.union(B)
        output["A^B"] = A.intersection(B)
    if simplify:
        for key, value in output.items():
            output[key] = len(value)
    return output


# ---------------
# Health Report
# ---------------
def column_values_summary(data):
    """Summarise the column types, number of unique values, and percentage of unique values per column.

    Parameters
    ----------
        data - pyspark.sql.DataFrame

    Returns
    -------
        df - pd.core.frame.DataFrame
    """

    # datatypes
    a = pd.DataFrame({x.name: [x.dataType] for x in list(data.schema)})

    # countDistinct
    b = data.agg(*(F.countDistinct(F.col(c)).alias(c) for c in data.columns)).toPandas()

    # percent of countDistinct over the entire len
    c = round(b * 100 / data.count(), 2)

    df = a.append(b)
    df = df.append(c)

    df.index = ["Data type", "Distinct count", "Distinct count(%)"]
    return df


def get_datatypes(data):
    """List the numerical columns and non_numerical columns.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    numerical_columns: list(str)
        list of numerical columns
    non_numerical_columns: list(str)
        list of non numerical columns
    """
    numerical_columns = dp.list_numerical_columns(data)
    non_numerical_columns = set(data.columns) - set(numerical_columns)
    return numerical_columns, non_numerical_columns


def get_missing_values_summary(data):
    """Get a summary of the missing values.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    df_mising: pd.DataFrame
        pyspark dataframe that contains counts of missing values
    """
    df_missing = dp.identify_missing_values(data)
    df_missing = df_missing.toPandas().T
    return df_missing


def _missing_values(data):
    """Get a pandas dataframe with the information about missing values in the dataset.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    df_missing: pd.core.frame.DataFrame
        pandas dataframe that contains summary of missing values

    """
    no_of_rows = data.count()
    df = data
    df = get_missing_values_summary(data)
    df = df.reset_index()
    df = df.rename(
        columns=dict(zip(list(df.columns), ["Variable Name", "No of Missing"],))  # noqa
    )
    df["Percentage Missing"] = df["No of Missing"] / float(no_of_rows) * 100
    return df


def get_health_analysis(
    data, missing=True, data_types=True, duplicate_values=True, duplicate_columns=True
):
    """Get the summary of health analysis.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    missing: bool, default is True.
    data_types: bool, default is True.
    duplicate_values:bool, default is True.
    duplicate_columns:bool, default is True.

    Returns
    -------
    dict_summary: dict
        dictionary containing the summary of health analysis
    """
    dict_summary = {
        "data_types": {},
        "missing": {},
        "duplicate_values": {},
        "duplicate_columns": {},
    }
    row_count = data.count()  # gives only num of rows
    col_count = len(data.columns)  # gives num of cols
    numerical_columns, non_numerical_columns = get_datatypes(data)
    numeric_list = [
        len(numerical_columns) / col_count,
        len(non_numerical_columns) / col_count,
    ] * 100
    df_missing = get_missing_values_summary(data)
    per_ = df_missing.sum()[0] / (
        row_count * col_count
    )  # data is pyspark.sql.dataframe NOT pyspark.pandas.dataframe hence df.size is not available
    missing_values = [1 - per_, per_] * 100
    dict_summary["missing"] = {
        "Available": missing_values[0],
        "Missing": missing_values[1],
    }
    dict_summary["data_types"] = {"Numeric": numeric_list[0], "Others": numeric_list[1]}
    duplicate_rows = (row_count - data.dropDuplicates().count()) / row_count
    duplicate_rows_list = [(1 - duplicate_rows), duplicate_rows]
    dict_summary["duplicate_values"] = {
        "Unique": duplicate_rows_list[0],
        "Duplicated": duplicate_rows_list[1],
    }

    # TBD duplicate columns
    return dict_summary


def plot_health(
    data, missing=True, data_types=True, duplicate_values=True, duplicate_columns=True
):
    """Get the health analysis plots.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    missing: bool, default is True.
    data_types: bool, default is True.
    duplicate_values:bool, default is True.
    duplicate_columns:bool, default is True.

    Returns
    -------
    final_plot: HvPlot
            HvPlot containing all the health plots of variables
    """
    data_dict = get_health_analysis(
        data, missing, data_types, duplicate_values, duplicate_columns
    )

    df_dict = {
        "type": flatten_list([[x] * len(data_dict[x]) for x in data_dict.keys()]),
        "labels": list(data_dict["data_types"].keys())
        + list(data_dict["missing"].keys())
        + list(data_dict["duplicate_values"].keys()),
        "values": list(data_dict["data_types"].values())
        + list(data_dict["missing"].values())
        + list(data_dict["duplicate_values"].values()),
    }
    df = pd.DataFrame(df_dict)
    df = df.set_index(["type", "labels"])
    # this multipliers resolves the issue of duplicate columns as it's values are multiplied by 1 and others
    # with no_of_columns. which was needed for the correct metrics.
    final_plot = None
    for metric in df.index.get_level_values(0).unique():
        plot = (
            hvPlot(df.loc[metric].T)
            .bar(stacked=True, title=metric, height=100, invert=True)
            .opts(xticks=list([i for i in range(df.shape[1])]))
        )
        if final_plot:
            final_plot += plot
        else:
            final_plot = plot
    return final_plot.cols(1)


def missing_plot(data):
    """Get the Missing Variable plot.

    The function returns a bar plot mentioning the number of variables that are present in each missing value
    bucket(0%-100% in increments of 5%).

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    f: `hvplot`
        missing_plot returns a bar plot with the following axis:

        X.axis - % of missing observation bucket
        Y.axis - Number of variables
    """
    # plt.close('all')
    df = data
    missing_values = _missing_values(data)
    break_value = [0, 5, 10, 20, 30, 40, 50, 100]
    lab_value = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%"]
    cuts = pd.cut(
        missing_values["Percentage Missing"],
        bins=break_value,
        labels=lab_value,
        right=True,
    )
    cuts = cuts.value_counts().reindex(lab_value)
    remaining_cols = len(df.columns) - cuts.sum()
    cuts = pd.concat([pd.Series([remaining_cols], index=["No Missing"]), cuts])
    plot = hvPlot(cuts).bar(
        rot=0,
        title="Missing Variables",
        xlabel="# of missing observations",
        ylabel="# of variables",
    )
    return plot


def missing_value_summary(data):
    """Get the summary of missing values computed from `missing_values` function.

    This function describes the share of missing values for each variable
    in the dataset. If there are no missing values, "No Missing Values"
    message is displayed, else a table containing the percentage of
    missing for all variables with missing values are displayed

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    df: pandas.DataFrame

    """
    df = _missing_values(data)
    df = df.loc[df["No of Missing"] != 0].reset_index(drop=True)
    if df.empty:
        return "No Missing Values"
    else:
        return df


def get_outliers(data, cols=None, get_index=False):
    """To get the summary of outliers.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list(str)
        list of numerical columns
    get_index: bool, default=False
        By default False, make it True for getting a dictionary with the index of outliers present in the data for both iqr and std bounds for each column

    Returns
    -------
    outliers_dict_iqr: dict
        contains the details of iqr bounds
    outliers_dict_mean: dict
        contains the details of std bounds
    outliers_dict_index: dict
        returned when get_index is set to True, contains index of outliers of both iqr and std bounds
    """
    iqr_bounds = dp._calculate_outlier_bounds_iqr(data, cols)
    mean_bounds = dp._calculate_outlier_bounds_sdv(data, cols)
    outliers_dict_iqr = {}
    outliers_dict_mean = {}
    outliers_dict_index = {}
    for column in cols:

        col_in_process = [row[0] for row in data.select(column).collect()]

        # fetching the upper and lower range of columns from
        # iqr_bounds and mean_bounds (i.e iqr based and sdv based)
        iqr_min = iqr_bounds[column]["min_b"]
        iqr_max = iqr_bounds[column]["max_b"]
        mean_min = mean_bounds[column]["min_b"]
        mean_max = mean_bounds[column]["max_b"]

        # creating the outlier list based in iqr range, where
        # "upper" -> upper range outlier, "lower" -> lower range outlier
        # else it is 0
        map_iqr = list(
            map(
                lambda x: ("upper" if x > iqr_max else "lower" if x < iqr_min else 0)
                if x is not None
                else 0,
                col_in_process,
            )
        )
        # creating the outlier list based in standard deviation and mean, where
        # "upper" -> upper range outlier, "lower" -> lower range outlier
        # else it is 0
        map_mean = list(
            map(
                lambda x: ("upper" if x > mean_max else "lower" if x < mean_min else 0)
                if x is not None
                else 0,
                col_in_process,
            )
        )

        # iqr_based_outlier_count and mean_based_outlier_count
        # are the placeholder array for number of outliers after the count
        # from map_iqr and map_mean lists repectively.
        iqr_based_outlier_count = [0, 0]
        mean_based_outlier_count = [0, 0]

        # taking count of outliers from the map_iqr and map_mean list
        iqr_based_outlier_count[0] = map_iqr.count("lower")
        iqr_based_outlier_count[1] = map_iqr.count("upper")
        outliers_dict_iqr[column] = tuple(iqr_based_outlier_count)

        mean_based_outlier_count[0] = map_mean.count("lower")
        mean_based_outlier_count[1] = map_mean.count("upper")
        outliers_dict_mean[column] = tuple(mean_based_outlier_count)
        if get_index is True:
            index_dict = {}
            upper_mean = []
            lower_mean = []
            upper_iqr = []
            lower_iqr = []
            for idx, value in enumerate(map_iqr):
                if value == "upper":
                    upper_iqr.append(idx)
                elif value == "lower":
                    lower_iqr.append(idx)
            for idx, value in enumerate(map_mean):
                if value == "upper":
                    upper_mean.append(idx)
                elif value == "lower":
                    lower_mean.append(idx)
            index_dict["lower_mean"] = lower_mean
            index_dict["upper_mean"] = upper_mean
            index_dict["lower_iqr"] = lower_iqr
            index_dict["upper_iqr"] = upper_iqr
            outliers_dict_index[column] = index_dict
    if get_index is True:
        return outliers_dict_iqr, outliers_dict_mean, outliers_dict_index
    else:
        return outliers_dict_iqr, outliers_dict_mean


def get_outliers_table(data, get_index=False):
    """To get a dataframe with outlier analysis table.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    get_index: bool, default=False
        By default False, if True it returns the outliers_df with index of outliers

    Returns
    -------
    outliers_df: pandas.core.frame.DataFrame
        when get_index flag is False, it returns dataframe with count of outliers for both iqr and std bounds
        when get_index flag is True, it returns dataframe with count of outliers aswell as index of those outliers for both iqr and std bounds
    """
    if data is None:
        raise ValueError("Data is not provided")
    outlier_col_labels = [
        "< (mean-3*std)",
        "> (mean+3*std)",
        "< (1stQ - 1.5 * IQR)",
        "> (3rdQ + 1.5 * IQR)",
    ]
    numerical_columns = list(
        set(dp.list_numerical_columns(data))
        - set(dp.list_numerical_categorical_columns(data))
    )
    if len(numerical_columns) != 0:
        if get_index is True:
            outliers_dict_iqr, outliers_dict_mean, outliers_dict_index = get_outliers(
                data, numerical_columns, get_index=True
            )
        else:
            outliers_dict_iqr, outliers_dict_mean = get_outliers(
                data, numerical_columns, get_index=False
            )
        outliers_df = pd.DataFrame.from_dict(outliers_dict_mean)
        outliers_df = pd.concat(
            [outliers_df, pd.DataFrame.from_dict(outliers_dict_iqr)]
        )
        outliers_df = outliers_df.reset_index(drop=True).T
        outliers_df.rename(
            columns=dict(zip(list(outliers_df.columns), outlier_col_labels)),
            inplace=True,
        )
        outliers_sum = outliers_df.sum(axis=1)
        outliers_df = outliers_df[outliers_sum > 0]
        outliers_df.index.name = "feature"
        """ Creating a new outliers_index_df dataframe with index of outliers from
        outliers_dict_index dictionary and merged with the existing outliers_df with count of outliers """
        if get_index is True:
            outliers_index_df = pd.DataFrame.from_dict(
                outliers_dict_index
            ).T.rename_axis("feature")
            outlier_col_labels_with_index = [
                "< (mean-3*std) Index",
                "> (mean+3*std) Index",
                "< (1stQ - 1.5 * IQR) Index",
                "> (3rdQ + 1.5 * IQR) Index",
            ]
            outliers_index_df.rename(
                columns=dict(
                    zip(list(outliers_index_df.columns), outlier_col_labels_with_index)
                ),
                inplace=True,
            )
            outliers_df = pd.merge(
                outliers_df, outliers_index_df, left_index=True, right_index=True
            )
        return outliers_df
    else:
        outliers_df = pd.DataFrame([], columns=outlier_col_labels)
        outliers_df.index.name = "feature"
        return outliers_df


def data_health_recommendations(data):
    """
    Get health recommendations that can be applied on the data.

    The recommendations are made to improve the quality of the data,
    These are not Data Science recommendations.

    Parameters
    ----------
    data: pyspark.sql.dataframe.DataFrame
        Spark DataFrame for which you want to get the recommendations.

    Returns
    -------
    recommendation_df: pd.DataFrame
        The df containing 2 columns, Recommendations, Reason for Recommendation

    """
    recommendation_list = []
    reason_list = []
    # Recommendation 1: Drop duplicate columns if present(get from health plot)
    # pip install chispa for column equality

    # Recommendation 2: Drop duplicates rows if present
    duplicate_row_count = data.count() - data.dropDuplicates().count()
    if duplicate_row_count:
        recommendation_list.append("Drop duplicates rows")
        reason_list.append(f"There are {duplicate_row_count} duplicate rows")

    # Recommendation 3: Drop null cols if present(get from missing summary)
    df_missing = missing_value_summary(data)
    null_cols = []
    if not isinstance(df_missing, str):
        null_cols = df_missing[df_missing["Percentage Missing"] > 60][
            "Variable Name"
        ].to_list()
        if len(null_cols) > 0:
            recommendation_list.append(
                "Drop columns with high percentage of null values"
            )
            reason_list.append(f"Column(s) {null_cols} have more than 60% null values")

    # Recommendation 4: Check if no. of columns is >2 if more than proceed,
    # else display to user that EDA cannot be performed and get more data to continue
    if (len(data.columns) - len(null_cols)) < 2:
        recommendation_list.append(
            "Add more data to your data source to generate EDA charts"
        )
        reason_list.append(
            f"After dropping columns with high percentage of null values, you are left with <2 columns"
        )

        # Create a DataFrame
        if len(recommendation_list) == 0:
            recommendation_list.append("No Recommendations")
            reason_list.append("No Recommendations")

        recommendation_df = pd.DataFrame(
            {
                "Recommendations": recommendation_list,
                "Reason For Recommendation": reason_list,
            }
        )
        return recommendation_df

    # Recommendation 5: DataType check to typecast to common ones if present(get % of non-numeric from health plots and do dtype check on them)
    numerical_columns, non_numerical_columns = get_datatypes(data)
    non_string_columns = non_numerical_columns - set(list_categorical_columns(data))
    if len(non_string_columns) > 0:
        recommendation_list.append(
            "Convert dtype of columns to string(categorical columns) OR a numeric type"
        )
        reason_list.append(
            f"Columns {non_string_columns} are of dtypes {data.select(*non_string_columns).dtypes}"
        )

    # Recommendation 6: Imputation Columns
    # Can be recommended if the num of null rows is <5% if present (get from missing summary)
    imputation_cols = []
    if not isinstance(df_missing, str):
        imputation_cols = df_missing[df_missing["Percentage Missing"] <= 5][
            "Variable Name"
        ].to_list()
        if len(imputation_cols) > 0:
            recommendation_list.append("Impute columns")
            reason_list.append(
                f"Columns {imputation_cols} have less than 5% missing values."
            )

    data_without_nulls = data.drop(*null_cols)
    data_without_imputation = data_without_nulls.drop(*imputation_cols)
    # Recommendation 7: Drop rows where null present in more than 1 cols
    threshold = len(data_without_imputation.columns) - 1
    multi_null_rows = (
        data_without_imputation.count()
        - data_without_imputation
        .dropna(thresh=threshold)
        .count()
    )
    if multi_null_rows > 0:
        recommendation_list.append(
            "Drop rows where null is present in more than 1 columns"
        )
        reason_list.append(f"{multi_null_rows} rows have more than 1 null values.")

    # Recommendation 8: Imputation Columns
    # Can be recommended if the num of null rows is <5% if present (get from missing summary)
    df_missing_2 = missing_value_summary(
        data_without_imputation.dropna(thresh=threshold)
    )
    imputation_cols_2 = []
    if not isinstance(df_missing_2, str):
        imputation_cols_2 = df_missing_2[df_missing_2["Percentage Missing"] <= 5][
            "Variable Name"
        ].to_list()
        if len(imputation_cols_2) > 0:
            recommendation_list.append(
                "Impute columns after dropping rows where null is present in more than 1 columns"
            )
            reason_list.append(
                f"Columns {imputation_cols_2} have less than 5% missing values after dropping rows where null is present in more than 1 columns."
            )

    # Recommendation 9: Drop null rows
    imputation_cols_total = imputation_cols + imputation_cols_2
    data_without_null_imputation = data_without_nulls.drop(*imputation_cols_total).dropna()
    null_rows = (
        data_without_null_imputation.count()
        - data_without_null_imputation.dropna().count()
    )
    if null_rows > 0:
        recommendation_list.append("Drop null rows that cannot be properly imputed")
        reason_list.append(
            f"{null_rows} rows can be dropped as they cannot be properly imputed"
        )

    # Create a DataFrame
    if len(recommendation_list) == 0:
        recommendation_list.append("No Recommendations")
        reason_list.append("No Recommendations")
    recommendation_df = pd.DataFrame(
        {
            "Recommendations": recommendation_list,
            "Reason For Recommendation": reason_list,
        }
    )
    return recommendation_df


def health_analysis(data, save_as=None, save_path=""):
    """Data health report.

    Compiles outputs from data_health, missing_plot, missing_value_summary, get_outliers_df and data_health_recommendations as a report.

    Parameters
    ----------
    save_as : str, default=None
        give ".html" for saving the report
    save_path : str, default=''
        Location where report to be saved. By default report saved in working directory.

    Examples
    --------
    >>> from tigerml.pyspark import health_analysis
    >>> df = spark.read.parquet("train.parquet")
    >>> health_analysis(df, save_as=".html", save_path="PySpark_Reports/")
    """
    # data_shape = str(data.count())
    health_analysis_report = {}
    health_analysis_report.update({"health_plot": plot_health(data)})
    health_analysis_report.update({"missing_plot": missing_plot(data)})
    health_analysis_report.update(
        {"missing_value_summary": missing_value_summary(data)}
    )

    health_analysis_report.update({"outliers_in_features": get_outliers_table(data)})
    health_analysis_report.update({"data_health_recommendation": data_health_recommendations(data)})
    if save_as:
        default_report_name = "health_analysis_report_at_{}".format(time_now_readable())
        save_path = append_file_to_path(save_path, default_report_name + save_as)
        create_report(
            health_analysis_report, path=save_path, format=save_as,
        )
    return health_analysis_report


# ----------------
# Feature Analysis
# ----------------
def describe_data(data, columns):
    """Obtain the basic stats results and percentiles of numerical data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    columns: list(str)
            the column name list of the numerical variables

    Returns
    -------
    describe_df: pd.DataFrame
        the numerical describe info. of the input dataframe
    """
    numeric_df = data.select(columns)

    # Get Pyspark describe df,convert to pd.df to transpose
    describe_df = numeric_df.describe().toPandas().T
    # Make 1st row as column
    describe_df.columns = describe_df.iloc[0]
    # Drop 1st row
    describe_df.drop(index=describe_df.index[0], axis=0, inplace=True)
    # Create new cols
    describe_df["samples"] = None
    describe_df["nunique"] = None
    describe_df["25%"] = None
    describe_df["50%"] = None
    describe_df["75%"] = None

    for c in numeric_df.columns:
        numeric_series = numeric_df.select(c).dropna().distinct()
        nunique = numeric_series.count()
        samples = numeric_series.limit(5).rdd.flatMap(lambda x: x).collect()
        describe_df.at[c, "samples"] = samples
        describe_df.at[c, "nunique"] = nunique
        percentile_list = numeric_df.select(
            F.percentile_approx(
                col=c, percentage=[0.25, 0.5, 0.75], accuracy=10000
            ).alias("q")
        ).first()[0]
        describe_df.at[c, "25%"] = percentile_list[0]
        describe_df.at[c, "50%"] = percentile_list[1]
        describe_df.at[c, "75%"] = percentile_list[2]

    # Rename Index column
    describe_df.index.rename(None, inplace=True)

    return describe_df


def describe_categoricaldata(data, cat_cols):
    """Obtain basic stats results of categorical data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cat_cols: list(str)
            the column name list of the categorical variable

    Returns
    -------
    describe_df: pd.DataFrame
        the categorical describe info. of the input dataframe
    """
    cat_cols = sorted(cat_cols)
    categorical_df = data.select(cat_cols)
    # Create empty df
    describe_df = pd.DataFrame(
        index=cat_cols, columns=["nunique", "samples", "mode", "mode_freq"]
    )

    for c in cat_cols:
        categorical_series = categorical_df.select(c).dropna().distinct()
        nunique = categorical_series.count()
        samples = categorical_series.limit(5).rdd.flatMap(lambda x: x).collect()
        describe_df.at[c, "samples"] = samples
        describe_df.at[c, "nunique"] = nunique
        mode_modefreq_list = (
            categorical_df.select(c)
            .groupBy(c)
            .count()
            .sort("count", ascending=False)
            .limit(1)
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        describe_df.at[c, "mode"] = mode_modefreq_list[0]
        describe_df.at[c, "mode_freq"] = mode_modefreq_list[1]

    # Rename Index column
    describe_df.index.rename(None, inplace=True)

    return describe_df


def feature_analysis_table(data):
    """Get the descriptive statistics of the data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    numerical_description: pd.DataFrame
        contains the descriptive statistics of numerical variables.
    categorical_description: pd.DataFrame
        contains the descriptive statistics of categorical variables.
    """
    numerical_columns = list(
        set(dp.list_numerical_columns(data))
        - set(dp.list_numerical_categorical_columns(data))
    )
    if len(numerical_columns) == 0:
        columns = [
            "count",
            "mean",
            "stddev",
            "min",
            "max",
            "samples",
            "nunique",
            "25%",
            "50%",
            "75%",
        ]
        numerical_description = pd.DataFrame([], columns=columns)
        numerical_description = numerical_description.reset_index(level=0).drop(
            ["index"], axis=1
        )
    else:
        numerical_description = describe_data(data, numerical_columns)

    categorical_columns = dp.list_categorical_columns(
        data
    ) + dp.list_numerical_categorical_columns(data)
    if len(categorical_columns) == 0:
        columns = [
            "nunique",
            "samples",
            "mode",
            "mode_freq",
        ]
        categorical_description = pd.DataFrame([], columns=columns)
        categorical_description = categorical_description.reset_index(level=0).drop(
            ["index"], axis=1
        )
    else:
        categorical_description = describe_categoricaldata(data, categorical_columns)
    return numerical_description, categorical_description


def density_plots_numerical(data):
    """Get the densisty plots of the numerical variables.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    """
    num_cols = list(
        set(dp.list_numerical_columns(data))
        - set(dp.list_numerical_categorical_columns(data))
    )
    fig, axes = plt.subplots(nrows=int(np.ceil(len(num_cols) / 2)), ncols=2)
    fig.set_size_inches(20, 20)
    # Concatenate only if axes is multi-dimensional
    if axes.ndim > 1:
        axes = np.concatenate(axes)
    plots_dict = {}
    for index_, col_ in enumerate(num_cols):
        plot_ = distplot(axes[index_], [data.select(col_)], bins=40)  # noqa
        axes[index_].set_title(f"distribution of {col_}")
        axes[index_].legend()
        plots_dict.update({col_: plot_})
    return plots_dict


def non_numeric_frequency_plots(data, cols):
    """Get a dictionary of interactive frequency plots and summary table for non-numeric cols.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list(str)
        default: empty, takes the requested columns in the given dataframe.

    Returns
    -------
    plot: dict
        for all the non-numeric required columns in the list if it is not empty else all non-numeric from the given data.
    """
    plots_dict = {}
    for col_ in sorted(cols):
        series = data.select(col_)
        summary_df = series.describe().toPandas().T
        summary_df.columns = summary_df.iloc[0]
        summary_df.drop(index=summary_df.index[0], axis=0, inplace=True)
        summary_table = hvPlot(summary_df).table(
            columns=list(summary_df.columns), height=60, width=600
        )
        freq_plot = hvPlot(
            series.toPandas()[col_].value_counts().head(20).sort_values(ascending=True)
        ).bar(title="Frequency Plot for {}".format(col_), invert=True, width=600)
        plots_dict.update({col_: (freq_plot + summary_table).cols(1)})
    return plots_dict


def density_plots(data, cols):
    """Get a dict of interactive density plots and numeric summary.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list
        default: empty, takes the requested columns in the given dataframe.

    Returns
    -------
    plots_dict: dict
        for all the requested numeric columns defined in the list if it is not empty else all non-numeric from the given data.
    """
    plots_dict = {}
    for col_ in sorted(cols):
        series = data.select(col_)
        summary_df = series.describe().toPandas().T.round(2)

        summary_df.columns = list(
            np.concatenate(summary_df.loc[summary_df.index == "summary"].values)
        )
        summary_df.drop("summary", inplace=True)
        summary_table = hvPlot(summary_df).table(
            columns=list(summary_df.columns), height=60, width=600
        )
        try:
            hist_plot = hv.Histogram(np.histogram(series.toPandas(), bins=20))
            density_plot = hvPlot(series.toPandas()).kde(
                title="Density Plot for {}".format(col_), width=600
            )
            hooks = [add_to_secondary, finalize_axes_right]
            complete_plot = hist_plot.options(
                color="#00fff0", xlabel=col_
            ) * density_plot.options(hooks=hooks)
            plots_dict[col_] = (complete_plot + summary_table).cols(1)
        except Exception as e:
            plots_dict[col_] = f"Could not generate. Error - {e}"

    return plots_dict


def non_numeric_frequency_plots_v2(data, cols):
    """Get a holoviews layout of interactive frequency plots and summary table for non-numeric cols.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list(str)
        default: empty, takes the requested columns in the given dataframe.

    Returns
    -------
    plot: holoviews.core.layout.Layout
        for all the non-numeric required columns in the list if it is not empty else all non-numeric from the given data.
    """
    series = data.select(cols)
    summary_df = series.describe().toPandas().T.round(2)
    summary_df.columns = list(
        np.concatenate(summary_df.loc[summary_df.index == "summary"].values)
    )
    summary_df.drop("summary", inplace=True)
    summary_table = hvPlot(summary_df).table(
        columns=list(summary_df.columns), height=60, width=600
    )
    freq_plot = hvPlot(
        series.toPandas()
        .apply(lambda x: x.value_counts().head(20).sort_values(ascending=True))
        .T.reset_index()
        .melt(id_vars="index")
        .dropna()
        .reset_index()
        .drop(columns=["level_0"])
    ).bar(
        invert=True,
        width=600,
        x="variable",
        y=["value"],
        subplots=True,
        by="index",
        shared_axes=False,
    )
    plots = (freq_plot + summary_table).cols(1)
    return plots


def density_plots_v2(data, cols):
    """Get a holoviews layout of interactive density plots.

    A numeric summary for the given columns or all numeric columns for the given data is also provided.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list
        default: empty, takes the requested columns in the given dataframe.

    Returns
    -------
    plots: holoviews.core.layout.Layout
        for all the requested numeric columns defined in the list if it is not empty else all non-numeric from the given data.
    """
    series = data.select(cols)
    summary_df = series.describe().toPandas().T.round(2)
    summary_df.columns = list(
        np.concatenate(summary_df.loc[summary_df.index == "summary"].values)
    )
    summary_df.drop("summary", inplace=True)
    summary_table = hvPlot(summary_df.loc[cols]).table(
        columns=list(summary_df.columns),
        height=60,
        width=600,
        subplots=True,
        shared_axes=False,
    )

    density_plot = hvPlot(series.toPandas()[cols]).kde(
        width=600, subplots=True, shared_axes=False
    )
    plots = (density_plot + summary_table).cols(1)

    return plots


def feature_density_plots(data, num_cols=[], cat_cols=[]):
    """Get density plots and bar plots for numerical and categorical columns respectively.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    Returns
    -------
    numerical_plots: dict
        dict containing the density plots of numerical variables.
    categorical_plots:
        dict containing the bar plots of categorical variables

    """
    categorical_cols = dp.list_categorical_columns(
        data
    ) + dp.list_numerical_categorical_columns(data)
    if not cat_cols:
        cat_cols = categorical_cols
    else:
        for column in cat_cols:
            assert (  # noqa
                column in categorical_cols
            ), "{0} is not a valid categorical column in the input data"

    numerical_cols = list(
        set(dp.list_numerical_columns(data))
        - set(dp.list_numerical_categorical_columns(data))
    )
    if not num_cols:
        num_cols = numerical_cols
    else:
        for column in num_cols:
            assert (  # noqa
                column in numerical_cols
            ), "{0} is not a valid numerical column in the input data"
    numerical_plots = density_plots(data, numerical_cols)
    categorical_plots = non_numeric_frequency_plots(data, categorical_cols)
    return numerical_plots, categorical_plots


def feature_analysis(data, save_as=None, save_path=""):
    """Univariate analysis for the columns. Generates summary_stats and distributions plots for columns.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    save_as : str, default=None
        Name of the report. By default name is auto generated from system timestamp.
    save_path : str, default=''
        Location where report to be saved. By default report saved in working directory.

    Examples
    --------
    >>> from tigerml.pyspark.eda import feature_analysis
    >>> df = spark.read.parquet("train.parquet")
    >>> feature_analysis(df, save_as=".html", save_path="PySpark_Reports/")
    """
    report = {}
    numeric_variables, non_numeric_summary = feature_analysis_table(data)
    report["summary_stats"] = {}
    # report['summary_stats']['variable_summary'] = self.variable_summary()
    report["summary_stats"]["numeric_variables"] = [numeric_variables]
    report["summary_stats"]["non_numeric_variables"] = [non_numeric_summary]

    report["distributions"] = {}
    numeric_density, non_numeric_frequency = feature_density_plots(data)
    report["distributions"]["numeric_variables"] = numeric_density
    report["distributions"]["non_numeric_variables"] = non_numeric_frequency

    if save_as:
        default_report_name = "feature_analysis_report_at_{}".format(
            time_now_readable()
        )
        save_path = append_file_to_path(save_path, default_report_name + save_as)
        create_report(
            report, path=save_path, format=save_as,
        )
    return report


# Feature Interactions
def correlation_table(data, plot="table"):
    """Get feature interaction plot or table.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    plot: str
        if table, then correlation table is obtained else correlation plot

    Returns
    -------
    c_df: pd.DataFrame
        correlation table
    heatmap: hvplot
        correlation plot
    """

    cat_cols = dp.list_categorical_columns(data)
    if len(cat_cols):
        data = label_encode(data, cat_cols)
    cols = dp.list_numerical_columns(data)
    outputCol_name = custom_column_name("features", data.columns)
    assembler = VectorAssembler(
        inputCols=cols, outputCol=outputCol_name, handleInvalid="skip"
    )
    df_vector = assembler.transform(data).select(outputCol_name)
    corr_mat = Correlation.corr(df_vector, outputCol_name, method="pearson")
    corr_mat = corr_mat.collect()[0].asDict()[f"pearson({outputCol_name})"]
    corr_df = pd.DataFrame(corr_mat.toArray())
    corr_df.index, corr_df.columns = cols, cols

    if plot == "table":
        corr_df = corr_df.where(np.triu(np.ones(corr_df.shape)).astype(np.bool))
        c_df = corr_df.stack().reset_index()
        c_df = c_df.rename(
            columns=dict(zip(list(c_df.columns), ["var1", "var2", "corr_coef"]))
        )
        return c_df
    else:
        heatmap = hvPlot(corr_df).heatmap(rot=45, height=450)

        return heatmap


def num_vs_num_bivariate(data, num_col1: str, num_col2: str, return_plots=False):
    """Get bivaraite data for numerical vs numerical features.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    num_col1: str
        first numerical column
    num_col2: str
        second numerical column
    return_plots:  bool, default=False
        If false return bivariate data dataframe, else return both bivariate data and bivariate plot data

    Returns
    -------
    num_vs_num_bivariate: pyspark.sql.DataFrame
        contains bivariate data
    chart: holoviews.element.chart.Scatter
        return when return_plots is True, contains hv scatter plot
    """
    num_vs_num_df = data.select(col(num_col1), col(num_col2)).dropna()
    if return_plots:
        df = num_vs_num_df.toPandas()
        chart = hvPlot(df).scatter(x=num_col1, y=num_col2, height=400, width=400)
        return num_vs_num_df, chart
    return num_vs_num_df


def cat_vs_cat_bivariate(data, cat_col1: str, cat_col2: str, return_plots=False):
    """Get bivaraite data for categorical vs categorical features. Limited to top 10 most frequent categorical values.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cat_col1: str
        first categorical column
    cat_col2: str
        second categorical column
    return_plots:  bool, default=False
        If false return bivariate data dataframe, else return both bivariate data and bivariate plot data

    Returns
    -------
    processed_df: pyspark.sql.DataFrame
        contains bivariate data
    chart: holoviews.element.chart.Bar
        return when return_plots is True, contains hv bar plot
    """
    cat_vs_cat_df = data.select(col(cat_col1), col(cat_col2)).dropna()
    processed_df = (
        cat_vs_cat_df.groupBy([cat_col1, cat_col2])
        .agg({cat_col2: "count"})
        .withColumnRenamed(f"count({cat_col2})", "Total_count")
        .replace(float("nan"), None)
        .sort(col("Total_count").desc())
    )
    if return_plots:
        df = processed_df.limit(10).toPandas()
        chart = hvPlot(df).bar(
            ylabel="Number of " "Observations",
            x=cat_col2,
            y="Total_count",
            by=cat_col1,
            stacked=True,
            rot=45,
        )
        return processed_df, chart
    return processed_df


def cat_vs_num_bivariate(data, cat_col: str, num_col: str, return_plots=False):
    """Get bivaraite data for categorical vs categorical features. Limited to top 10 most frequent categorical values.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cat_col: str
        first categorical column
    num_col: str
        second numerical column
    return_plots:  bool, default=False
        If false return bivariate data dataframe, else return both bivariate data and bivariate plot data

    Returns
    -------
    q_df: pyspark.sql.DataFrame
        contains bivariate data
    chart: holoviews.element.chart.Violin
        return when return_plots is True, contains hv violin plot
    """
    cat_vs_num_df = data.select(col(cat_col), col(num_col)).dropna()
    q_df = (
        cat_vs_num_df.groupBy(cat_col)
        .agg(
            percentile_approx(num_col, 0.25, lit(10000)).alias("q1"),
            percentile_approx(num_col, 0.75, lit(10000)).alias("q3"),
            F.min(num_col).alias("min"),
            F.max(num_col).alias("max"),
        )
        .sort(col(cat_col))
    )
    if return_plots:
        cat_vs_num_top_10_categories = (
            cat_vs_num_df.groupBy(cat_col).count().sort(desc("count")).limit(10)
        )
        pandas_df = cat_vs_num_top_10_categories.toPandas()
        column_list = pandas_df[cat_col].tolist()
        filtered_df = cat_vs_num_df.filter(cat_vs_num_df[cat_col].isin(column_list))
        filtered_pandas_df = filtered_df.toPandas()
        chart = hvPlot(filtered_pandas_df).violin(
            y=num_col, by=cat_col, height=400, width=400, legend=False
        )
        return q_df, chart
    return q_df


def get_bivariate(data, features=None, corr_table=None, return_plots=False, top_n=10):
    """Extract Bivariate data from the dataframe.

    Generates all the numerical vs numerical, categorical vs numerical and categorical
    vs numerical feature interaction data and plots

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    features: list, default = 'None'
        By deafult we consider all features, else provide list of features to get bivariates for
    corr_table: pd.DataFrame, default = 'None'
        corr_table required for feature interactions
    return_plots:  bool, default=False
        If false return bivariate data, else return both bivariate data and bivariate plots
    top_n: int, default = 10
        Number of bivariates to be generated, if top_n > rows in corr_table we display all rows


    Examples
    --------
    >>> from tigerml.pyspark.eda import get_bivariate
    >>> from tigerml.pyspark.eda import correlation_table
    >>> df = spark.read.parquet("train.parquet")
    >>> features = df.columns[0:2]
    >>> data = df.select([features[0],features[1]])
    >>> correlation = correlation_table(data, plot="table")
    >>> get_bivariate(data, features=features, corr_table=correlation, return_plots=False, top_n=10)

    Returns
    -------
    bivariate_data_dict: dict
        contains bivariate data
    bivariate_plot_dict: dict
        return when return_plots is True, conatins bivariate plots
    """
    if features is None:
        data = data
    else:
        data = data.select(features)
    if corr_table is None:
        corr_table = correlation_table(data=data)

    corr_table = corr_table[corr_table.var1 != corr_table.var2]
    corr_table["corr_coef"] = abs(corr_table["corr_coef"])
    corr_table = corr_table.sort_values("corr_coef", ascending=False).head(top_n)
    correlation_array = corr_table.to_numpy()
    date_cols = list_datelike_columns(data=data)
    num_cat_cols = list_numerical_categorical_columns(data)
    cat_cols = (
        list_boolean_columns(data=data)
        + list_categorical_columns(data)
        + date_cols
        + num_cat_cols
    )
    num_cols = list_numerical_columns(data=data)
    num_cols = list(set(num_cols) - set(cat_cols))
    bivariate_data_dict = {}
    bivariate_plot_dict = {}
    # Generates list_of_bivariate_plots
    for i, bi_variate_corr in enumerate(correlation_array):
        num_of_bivariate = top_n if top_n is not None else len(data.columns)
        if (len(bivariate_data_dict)) >= num_of_bivariate:
            break
        col_type = []
        # Loop for identifying col type
        for j, var in enumerate(bi_variate_corr[0:2]):
            # identify var type
            if "label_encoded_" in var:
                correlation_array[i][j] = var.replace("label_encoded_", "")
                col_type.append("categorical")
            elif var in cat_cols:
                col_type.append("categorical")
            elif var in num_cols:
                col_type.append("numerical")
            else:
                col_type.append("categorical")
        # Redirect to respective plots based on col types
        if col_type[0] == "numerical" and col_type[1] == "numerical":
            bivariate_data = num_vs_num_bivariate(
                data, bi_variate_corr[0], bi_variate_corr[1], return_plots=return_plots
            )

        elif col_type[0] == "numerical" and col_type[1] == "categorical":
            bivariate_data = cat_vs_num_bivariate(
                data, bi_variate_corr[1], bi_variate_corr[0], return_plots=return_plots
            )

        elif col_type[0] == "categorical" and col_type[1] == "numerical":
            bivariate_data = cat_vs_num_bivariate(
                data, bi_variate_corr[0], bi_variate_corr[1], return_plots=return_plots
            )

        elif col_type[0] == "categorical" and col_type[1] == "categorical":
            bivariate_data = cat_vs_cat_bivariate(
                data, bi_variate_corr[0], bi_variate_corr[1], return_plots=return_plots
            )
        key = f"{bi_variate_corr[0]} vs {bi_variate_corr[1]}"
        if return_plots:
            bivariate_df, bivariate_plot_data = bivariate_data
            bivariate_plot_data = bivariate_plot_data.opts(
                title=f"Correlation {round(bi_variate_corr[2], 3)}"
            )
            bivariate_data_dict[key] = bivariate_df
            bivariate_plot_dict[key] = bivariate_plot_data
        else:
            bivariate_df = bivariate_data
            bivariate_data_dict[key] = bivariate_df
    if return_plots:
        return bivariate_data_dict, bivariate_plot_dict
    else:
        return bivariate_data_dict


def feature_interactions(data, save_as=None, save_path=""):
    """Compiles outputs from correlation_table, correlation_heatmap, covariance_heatmap and bivariate_plots as a report.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    save_as : str, default=None
        Name of the report. By default name is auto generated from system timestamp.
    save_path : str, default=''
        Location where report to be saved. By default report saved in working directory.

    Examples
    --------
    >>> from tigerml.pyspark.eda import feature_interactions
    >>> df = spark.read.parquet("train.parquet")
    >>> feature_interactions(df, save_as=".html", save_path="PySpark_Reports/")
    """
    feature_interactions_report = {}
    feature_interactions_report["correlation_table"] = [correlation_table(data)]
    feature_interactions_report["correlation_heatmap"] = [correlation_table(data, "")]
    bivariate_data, bivariate_plots = get_bivariate(data, return_plots=True)
    feature_interactions_report[
        "bivariate_plots (Top 10 correlations)"
    ] = bivariate_plots

    if save_as:
        default_report_name = "feature_interactions_report_at_{}".format(
            time_now_readable()
        )
        save_path = append_file_to_path(save_path, default_report_name + save_as)
        create_report(
            feature_interactions_report, path=save_path, format=save_as,
        )
    return feature_interactions_report


# ------------
# Key Features
# ------------
def correlation_with_target(data, target_var, cols=None):
    """Get a barplot with correlation with target variable.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    target_var: str
        target variable of the data
    cols: list
        List of numerical columns.
        default - considers all the numeriacal features in the data

    Returns
    -------
    plot: hvplot of bars related to the correlation with target variable
    """
    # Renaming the string categorical col if it is target variable
    cat_cols = dp.list_categorical_columns(data)
    if target_var in cat_cols:
        target_var = f"label_encoded_{target_var}"
    else:
        target_var = target_var

    if not cols:
        if len(cat_cols):
            data = label_encode(data, cat_cols)

        cols = dp.list_numerical_columns(data)
    outputCol_name = custom_column_name("features", data.columns)
    assembler = VectorAssembler(
        inputCols=cols, outputCol=outputCol_name, handleInvalid="skip"
    )
    df_vector = assembler.transform(data).select(outputCol_name)
    corr_mat = Correlation.corr(df_vector, outputCol_name, method="pearson")
    corr_mat = corr_mat.collect()[0].asDict()[f"pearson({outputCol_name})"]
    corr_df = pd.DataFrame(corr_mat.toArray())
    corr_df.index, corr_df.columns = cols, cols
    corr_df = corr_df[[target_var]]
    corr_df = corr_df[corr_df.index != target_var]
    corr_df = corr_df[~corr_df[target_var].isna()]
    corr_df.rename(
        columns={target_var: "Pearson_correlation_with_Target"}, inplace=True
    )
    corr_df.sort_values(by="Pearson_correlation_with_Target", inplace=True)
    plot = hvPlot(corr_df).bar(
        invert=True, title="Feature Correlation with Target Function"
    )
    return plot


def label_encode(data, cat_cols):
    """Obtain label encoding of categorical variables.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cat_cols: list(str)
        list of categorical column names

    Returns
    -------
    data: pyspark.sql.DataFrame
        dataframe with label encoding of categorical columns
    """
    indexers = []
    for cat_ in cat_cols:
        stringIndexer = StringIndexer(
            inputCol=cat_, outputCol=f"label_encoded_{cat_}", handleInvalid="keep"
        )
        indexers += [stringIndexer]
    pipeline = Pipeline(stages=indexers)
    data = pipeline.fit(data).transform(data)
    return data


def feature_importance(data, target_var, classification=False):
    """Get feature importance plot based on RandomForestRegressor or RandomForestClassifier.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    target_var: str
        target variable of the data
    classification: bool, default is False.
        choosen basen on Regression or Classification problem.

    Returns
    -------
    plot: hvplot
        feature importance plot based on Random Forests
    """
    cat_cols = dp.list_categorical_columns(data)
    unique_list = []
    for c in cat_cols:
        nunique = data.select(c).dropna().distinct().count()
        unique_list.append(nunique)
    if unique_list:
        bins = max(unique_list) + 1
    else:
        bins = 100
    if len(cat_cols):
        data = label_encode(data, cat_cols)
    input_cols = dp.list_numerical_columns(data)
    if target_var in input_cols:
        input_cols.remove(target_var)
    elif f"label_encoded_{target_var}" in input_cols:
        input_cols.remove(f"label_encoded_{target_var}")
    outputCol_name = custom_column_name("features", data.columns)
    assembler = VectorAssembler(
        inputCols=input_cols, outputCol=outputCol_name, handleInvalid="skip"
    )
    model_data = assembler.transform(data)
    if classification:
        from pyspark.ml.classification import RandomForestClassifier
        if target_var in cat_cols:
            label = f"label_encoded_{target_var}"
        else:
            label = target_var
        rf = RandomForestClassifier(
            numTrees=3, maxDepth=20, labelCol=label, featuresCol=outputCol_name, maxBins=bins, seed=42
        )
    else:
        from pyspark.ml.regression import RandomForestRegressor
        label = target_var
        rf = RandomForestRegressor(
            numTrees=3, maxDepth=20, labelCol=target_var, featuresCol=outputCol_name, maxBins=bins, seed=42
        )

    model_data = model_data.select([outputCol_name, label])

    # BUILD THE MODEL
    model = rf.fit(model_data)

    # FEATURE IMPORTANCES
    feature_importance = pd.DataFrame.from_dict(
        dict(zip(input_cols, model.featureImportances.toArray())), orient="index"
    ).rename(columns={0: "Feature Importance"})
    feature_importance.sort_values(by="Feature Importance", inplace=True)
    plot = hvPlot(feature_importance).bar(
        invert=True, title="Feature Importances from RF"
    )
    return plot


def feature_analysis_pca(data, target_var):
    """Get feature importance dataframe based on RandomForestRegressor or RandomForestClassifier.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    target_var: str
        target variable of the data

    Returns
    -------
    transformed: pyspark.sql.DataFrame
        modified dataframe with pca column
    """
    from pyspark.ml.feature import PCA as PCAml

    cat_cols = dp.list_categorical_columns(data)
    cat_cols = [i for i in cat_cols if i != target_var]
    if len(cat_cols):
        data = label_encode(data, cat_cols)
    num_cols = dp.list_numerical_columns(data)
    num_cols = [i for i in num_cols if i != target_var]
    outputCol_name = custom_column_name("features", data.columns)
    assembler = VectorAssembler(
        inputCols=num_cols, outputCol=outputCol_name, handleInvalid="skip"
    )
    model_data = assembler.transform(data)
    pca = PCAml(k=2, inputCol=outputCol_name, outputCol=custom_column_name("pca", data.columns))
    model = pca.fit(model_data)
    transformed = model.transform(model_data)
    return transformed


class EDAReportPyspark:
    """EDA toolkit for classification and regression models.

    To evaluate and generate reports to summarize, data health, univariate & bivariate analyis, interactions and keydrivers.

    Parameters
    ----------
    data: pyspark.sql.DataFrame

    y : string, default=None
        Name of the target column

    is_classification : bool, default=None
        Set to True, for classificaiton target

    Examples
    --------
    >>> from tigerml.pyspark.eda import EDAReportPyspark
    >>> anp = EDAReportPyspark(data=train_df, is_classification=True, y="target_var")
    >>> anp.get_report(y="target_var",save_path="PySpark_Reports/")
    """

    def __init__(self, data, is_classification, y=None):
        self.data = data
        self.y = y
        self.is_classification = is_classification

    def _set_y_cols(self, y=None):
        if isinstance(y, str):
            y = [y]
        if y:
            return y
        else:
            return []

    def _get_x_cols(self, y=None):
        data = self.data
        if y is not None:
            y = [y] if isinstance(y, str) else y
        else:
            y = self.y_cols
        if self.y_cols:
            return [col for col in data.columns if col not in y]
        else:
            return list(data.columns)

    def _set_xy_cols(self, y):
        self.y_cols = self._set_y_cols(y=y)
        self.x_cols = self._get_x_cols()

    def key_drivers(
        self, y=None, features=None, quick=True, save_as=None, save_path=""
    ):
        """Univariate analysis for the columns.

        Generate summary_stats, distributions and normality tests for columns.

        Parameters
        ----------
        y : Target column name (String)
        features : list, default=[]
            list of columns in the dataframe for analysis. By default all are used.
        save_as : str, default=None
            Name of the report. By default name is auto generated from system timestamp.
        save_path : str, default=''
            Location where report to be saved. By default report saved in working directory.

        Examples
        --------
        >>> from tigerml.pyspark.eda import feature_interactions
        >>> anp = EDAReportPyspark(data=train_df, is_classification=True, y="target_var")
        >>> anp.key_drivers(y="target_var", save_as=".html", save_path="PySpark_Reports/")
        """
        if y:
            assert isinstance(y, str) or isinstance(y, list)
            ys = y if isinstance(y, list) else [y]
            self._set_xy_cols(ys)
        else:
            raise Exception("dependent variable name needs to be passed")

        if features:
            features = list(set(features) & set(self.x_cols))
        else:
            features = self.x_cols

        key_drivers = {}
        for y in ys:
            key_drivers[y] = {}
            key_drivers[y]["feature_scores"] = correlation_with_target(self.data, y)
            key_drivers[y]["feature_importances"] = feature_importance(
                self.data, y, classification=self.is_classification
            )
            # key_drivers[y]["pca_analysis"] = self.get_pca_analysis(features=features)

        # key_drivers[y]['tsne_projection'] = self.get_tsne_projection()
        if save_as:
            default_report_name = "key_drivers_report_at_{}".format(time_now_readable())
            save_path = append_file_to_path(save_path, default_report_name + save_as)
            create_report(
                key_drivers, path=save_path, format=save_as, split_sheets=True,
            )
        self.key_drivers_report = key_drivers
        return key_drivers

    def _create_report(self, y=None, quick=True, corr_threshold=None):
        if y:
            self._set_xy_cols(y)
        self.report = {}
        self.report["data_preview"] = {"head": [self.data.limit(5).toPandas()]}
        self.report["health_analysis"] = health_analysis(self.data)
        # self.report['data_preview']['pre_processing'] = self._prepare_data(corr_threshold)
        self.report["feature_analysis"] = feature_analysis(self.data)
        self.report["feature_interactions"] = feature_interactions(self.data)
        if self.y_cols:
            self.report["key_drivers"] = self.key_drivers(quick=quick, y=self.y_cols)
        else:
            _LOGGER.info(
                "Could not generate key drivers report as dependent variable is not defined"
            )

    def _save_report(self, format=".html", name="", save_path="", tiger_template=False):
        if not name:
            name = "data_exploration_report_at_{}".format(time_now_readable())
        create_report(
            self.report,
            name=name,
            path=save_path,
            format=format,
            split_sheets=True,
            tiger_template=tiger_template,
        )
        del self.report
        gc.collect()

    def get_report(
        self,
        format=".html",
        name="",
        y=None,
        corr_threshold=None,
        quick=True,
        save_path="",
        tiger_template=False,
    ):
        """Create consolidated report on data preview,feature analysis,feature interaction and health analysis.

        The consolidated report also includes key driver report if y(target dataframe) is passed while
        calling create_report.

        Parameters
        ----------
        y : str, default = None
        format : str, default='.html'
            format of report to be generated. possible values '.xlsx', '.html'
        name : str, default=None
            Name of the report. By default name is auto generated from system timestamp.
        save_path : str, default=''
            location with filename where report to be saved. By default is auto generated from system timestamp and saved in working directory.
        quick : boolean, default=True
            If true,calculate SHAP values and create bivariate plots
        corr_threshold : float, default=None
            To specify correlation threshold
        """
        self._create_report(y=y, quick=quick, corr_threshold=corr_threshold)
        return self._save_report(
            format=format, name=name, save_path=save_path, tiger_template=tiger_template
        )
