import random
import re
import string
from cytoolz import curry
from pyspark.ml import Estimator, Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as DT


def check_column_data_consistency(df):
    """Read the spark dataframe and check for each column data consistency.

    Parameters
    ----------
    df: spark.DataFrame()
      spark dataframe to check the consistency

    Returns
    -------
      None
    """
    flag = 1
    cat_cols = list_categorical_columns(df)
    cols_consistency_failed = []
    for col in cat_cols:
        df = df.withColumn(col + "_lower", F.lower(F.col(col)))
        temp_list = (
            df.select(F.countDistinct(col), F.countDistinct(col + "_lower"))
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        if temp_list[0] != temp_list[1]:
            cols_consistency_failed.append(col)
            flag = 0
    if flag == 1:
        return "Consistency Check Passed - No inconsistency in data"
    else:
        return "Consistency Check Failed - Convert the below columns to lowercase {cols_consistency_failed}"


# -----------------------------------------------------------------------
# Read Data
# -----------------------------------------------------------------------
def read_data(spark, paths, fs, fmt="parquet", header=True, inferschema=True):
    """Read data in the specified filesystem and format, and return it as a Spark dataframe.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        The Spark session to use for reading the data.
    paths: list
        path of the data in the given file system.
    fs: str
        The filesystem where the data is stored, e.g., "s3", "dbfs", "file", etc.
    fmt: str, optional (default="parquet")
        The format of the data. Supported formats include "csv", "parquet", "json", etc.
    header: bool, optional (default=True)
        Whether to treat the first line(s) of the file(s) as a header that specifies column names.
    inferschema: bool, optional (default=True)
        Whether to infer the schema of the data automatically from its contents.
    schema: pyspark.sql.DataFrame.schema
        If `inferschema` is False, this is the schema to use for interpreting the data.

    Returns
    -------
    pyspark.sql.DataFrame
        A Spark dataframe representing the data read from the specified file(s).

    Examples
    --------
    # Example 1: Read data from a local file
    >>> from pyspark.sql import SparkSession

    >>> # Create a SparkSession
    >>> spark = SparkSession.builder.appName("ReadDataExample").getOrCreate()

    >>> # Define the path and filesystem for the local file
    >>> paths = ["/path/to/local/file.parquet"]
    >>> fs = "file"

    >>> # Read the data from the local file as a Spark dataframe
    >>> df = read_data(spark, paths, fs, fmt="parquet", header=True, inferschema=True)

    >>> # Display the dataframe
    >>> df.show()
    +----+-----+
    |col1|col2 |
    +----+-----+
    |1   |data1|
    |2   |data2|
    +----+-----+

    """
    # Checks/Tests/Modifications needed to make it extensive for other filesystems

    if fs.lower() == "file":
        fpath = [path for path in paths]

    else:
        fpath = [fs + ":" + path for path in paths]

    df = spark.read.format(fmt).load(fpath, header=header, inferSchema=inferschema)
    return df


def save_data(data, path, *, fs=None, **kwargs):
    """Save data into the given path. type of data is inferred automatically.

    ``.csv`` and ``.parquet`` are compatible now

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    """
    # FIXME: Move io utils to a separate module and make things generic
    return data.write.mode("overwrite").parquet(path)


# -----------------------------------------------------------------------
# Clean Data
# -----------------------------------------------------------------------
def get_shape(df):
    """Get the shape of data as (nrows, ncols)."""
    return (df.count(), len(df.columns))


@curry
def clean_columns(data, strip=True, lower=False, sep="_"):
    """Standardize the column names of the dataframe. Converts camelcase into snakecase.

    Parameters
    ----------
    spark:
        spark instance
    data: spark.DataFrame()
        spark dataframe to clean the columns
    strip: bool
        strip to remove trailing whitespaces
    lower: bool
        convert the string to lowercase

    Returns
    -------
    df: spark.DataFrame
     spark dataframe with new cleaned columns
    """
    if not re.search(r"[ _-]", sep):
        raise ValueError(f"Separator cannot be {sep}. Only _, -, and space is allowed")
    old_cols = data.columns
    new_cols = []
    for column in old_cols:
        if strip:
            column = column.strip()
        # Replace spaces and special characters with underscores
        column = re.sub(r"[^a-zA-Z0-9_]", sep, column)
        if lower:
            # Converts camelcase into snakecase
            column = column.lower()
        # Make sure the column name does not start with a number
        if column[0].isnumeric():
            column = "col_" + column
        # Replace multipe underscores with single underscore
        column = re.sub(f"{sep}+", sep, column)
        new_cols.append(column)
    df = data.toDF(*new_cols)
    return df


@curry
def _clean_string_val(
    df,
    cols=[],
    special_chars_to_keep="._,$&",
    remove_chars_in_braces=True,
    strip=True,
    lower=False,
):
    """To clean any special characters or phonetics in the column values in character data.

    Parameters
    ----------
    str_value: str
    special_chars_to_keep: str
    remove_chars_in_braces: bool
    strip: bool
    lower: bool

    Returns
    -------
    str_value: str
        returns the cleaned string
    """
    for col_ in cols:
        if strip:
            # Remove multiple spaces
            df = df.withColumn(col_, F.regexp_replace(F.col(col_), "\s+", " "))  # noqa
            # Remove leading and trailing spaces
            df = df.withColumn(col_, F.trim(F.col(col_)))
        if lower:
            # Convert names to lowercase
            df = df.withColumn(col_, F.lower(F.col(col_)))
        if remove_chars_in_braces:
            # Remove characters between square and round braces
            df = df.withColumn(
                col_,
                F.regexp_replace(F.col(col_), "\(.*\)|\[.*\]", ""),  # noqa
            )  # noqa
        else:
            # Add braces to special character list, so that they will not be
            # removed further
            special_chars_to_keep = special_chars_to_keep + "()[]"
        if special_chars_to_keep:
            # Keep only alphanumeric character and some special
            # characters(.,_-&)
            reg_str = "[^\\w" + "\\".join(list(special_chars_to_keep)) + " ]"
            df = df.withColumn(col_, F.regexp_replace(F.col(col_), reg_str, ""))
        return df


@curry
def list_numerical_categorical_columns(data, threshold=5):
    """List the names of numerical categorical columns in the spark dataframe.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    threshold: int
            percentage of unique value

    Returns
    -------
    numeric_cat_cols: list
            list of columns with numeric categories
    """
    schema = data.dtypes
    no_of_rows = data.count()
    numeric_cat_cols = []

    # Consider all the columns excluding datatypes -
    # ["date", "boolean", "timestamp", "double", "float", "string"]
    possible_cat_cols = [
        x[0]
        for x in schema
        if x[1] not in ["date", "boolean", "timestamp", "double", "float", "string"]
    ]
    for col in possible_cat_cols:
        if (
            len(data.select(col).distinct().rdd.map(lambda r: r[0]).collect())
            / no_of_rows
        ) * 100 < threshold:
            numeric_cat_cols.append(col)
    return numeric_cat_cols


@curry
def list_numerical_columns(data):
    """List the names of numerical columns in the spark dataframe.

    Parameters
    ----------
    spark: SparkSession
            spark instance
    data: pyspark.sql.DataFrame

    Returns
    -------
    numerical_cols: list
            list of columns with numerical values
    """
    schema = data.dtypes
    numerical_cols = [
        x[0] for x in schema if x[1] not in ["string", "date", "boolean", "timestamp"]
    ]
    return numerical_cols


@curry
def list_categorical_columns(data):
    """List the names of categorical columns in the spark dataframe.

    Parameters
    ----------
    spark: SparkSession
            spark instance
    data: pyspark.sql.DataFrame

    Returns
    -------
    numerical_cols: list
            list of categorical columns in the data
    """
    schema = data.dtypes
    cat_cols = [x[0] for x in schema if x[1] in ["string"]]
    return cat_cols


@curry
def list_datelike_columns(data):
    """List the names of datelike columns in the spark dataframe.

    Parameters
    ----------
      spark: SparkSession
          spark instance
      data: pyspark.sql.DataFrame

    Returns
    -------
      d_cols: list
          list of categorical columns in the data
    """
    schema = data.dtypes
    d_cols = [x[0] for x in schema if x[1] in ["date", "timestamp"]]
    return d_cols


@curry
def list_boolean_columns(data):
    """List the names of boolean columns in the spark dataframe.

    Parameters
    ----------
      spark: SparkSession
          spark instance
      data: pyspark.sql.DataFrame

    Returns
    -------
      date_cols: list
          list of categorical columns in the data
    """
    schema = data.dtypes
    bool_cols = [x[0] for x in schema if x[1] in ["boolean"]]
    return bool_cols


def identify_col_data_type(data, col):
    """Identify the datatype of column in the data.

    Parameters
    ----------
      spark: SparkSession
      data: pyspark.sql.DataFrame
      col: str
        column name

    Returns
    -------
    str - one of "boolean","numerical","date_like","categorical"
    """
    num_cols = list(
        set(list_numerical_columns(data))
        - set(list_numerical_categorical_columns(data))
    )
    cat_cols = list_categorical_columns(data) + list_numerical_categorical_columns(data)
    bool_cols = list_boolean_columns(data)
    date_cols = list_datelike_columns(data)

    if col in num_cols:
        return "numerical"
    elif col in cat_cols:
        return "categorical"
    elif col in bool_cols:
        return "boolean"
    elif col in date_cols:
        return "date_like"
    else:
        raise ValueError("Unidentified Data Type")


def _drop_duplicates(data):
    """Drop Duplicates in the data.

    Parameters
    ----------
            spark: SparkSession
            data: pyspark.sql.DataFrame
    """
    # FIX ME: We can just use df.dropDuplicates(subset=[....]) here.
    # Should we actually require this function
    df = data.dropDuplicates()
    return df


# -----------------------------------------------------------------------
# Outlier Treatment
# -----------------------------------------------------------------------
# FIX ME - Currently +-1.5 IQR  and Mean
# Others have to be included (Refer below link)
# https://mapr.com/ebooks/spark/08-unsupervised-anomaly-detection-apache-spark.html
def handle_outliers(
    data, cols=[], drop=True, cap=False, method="iqr", prefix="", **kwargs
):
    """Handle the Outliers in data that learns about the outliers of training data.

    Parameters
    ----------
        data: pyspark.sql.DataFrame
        cols: list
            list of columns to check the outliers
            by default all numerical columns are considered
        drop: bool
            flag to drop or keep the outliers
        cap: bool
            flag to cap the outliers with the bounds
        method: str
            'iqr' or 'sdv'
        prefix: str
            relevant when cap=True
            prefix to be added to the new column where outliers are capped

    Returns
    -------
        bounds: dict
            a dictionay that contains each columns, lower and upper bound

    """
    num_cols = list(
        set(list_numerical_columns(data))
        - set(list_numerical_categorical_columns(data))
    )
    if not cols:
        cols = num_cols
    else:
        for col in cols:
            assert (  # noqa
                col in num_cols
            ), "{0} is not a valid numerical column in the input data"

    bounds = identify_outliers(data, cols=cols, method=method, **kwargs)
    return bounds


def _calculate_outlier_bounds_iqr(data, cols, iqr_multiplier=1.5):
    """Calculate the Outlier Bounds based on IQR in the data.

    Parameters
    ----------
    data: spark.DataFrame
    cols: list
    iqr_multiplier: numeric
        Multiplier to use to define the lower and upper bounds based on IQR

    """
    num_cols = list(
        set(list_numerical_columns(data))
        - set(list_numerical_categorical_columns(data))
    )
    if not cols:
        cols = num_cols
    else:
        for col in cols:
            assert (  # noqa
                col in num_cols
            ), "{0} is not a valid numerical column in the input data"

    temp = data.approxQuantile(cols, [0.25, 0.75], 0)
    bounds = {col: dict(zip(["q1", "q3"], temp[i])) for i, col in enumerate(cols)}

    # bounds = {
    #    c: dict(zip(["q1", "q3"], data.approxQuantile(c, [0.25, 0.75], 0)))
    #    for c in data.columns
    #    if c in cols
    # }
    for c in bounds:
        iqr = bounds[c]["q3"] - bounds[c]["q1"]
        bounds[c]["min_b"] = bounds[c]["q1"] - (iqr * iqr_multiplier)
        bounds[c]["max_b"] = bounds[c]["q3"] + (iqr * iqr_multiplier)

    return bounds


def _calculate_outlier_bounds_sdv(data, cols, sdv_multiplier=3):
    """Calculate the Outlier Bounds based on Mean and SDV.

    Parameters
    ----------
    spark: SparkSession
    data: pyspark.sql.DataFrame
    cols: list
    sdv_multiplier: numeric
        Multiplier to use to define the lower and upper bounds based on mean and sdv

    Returns
    -------
    bounds: dict()

    """
    num_cols = list(
        set(list_numerical_columns(data))
        - set(list_numerical_categorical_columns(data))
    )
    if not cols:
        cols = num_cols
    else:
        for col in cols:
            assert (  # noqa
                col in num_cols
            ), "{0} is not a valid numerical column in the input data"

    mean_expr = [F.mean(x) for x in cols]
    stddev_expr = [F.stddev(x) for x in cols]
    temp = data.agg(*mean_expr, *stddev_expr).rdd.flatMap(lambda x: x).collect()
    bounds = {
        col: dict(zip(["mean", "stddev"], [temp[i], temp[i + len(cols)]]))
        for i, col in enumerate(cols)
    }

    # bounds = {
    #    c: data.select(c)
    #    .describe()
    #    .filter(F.col("summary").isin("mean", "stddev"))
    #    .withColumn(c, F.col(c).cast("float"))
    #    .rdd.collectAsMap()
    #    for c in data.columns
    #    if c in cols
    # }
    for c in bounds:
        bounds[c]["min_b"] = bounds[c]["mean"] - (bounds[c]["stddev"] * sdv_multiplier)
        bounds[c]["max_b"] = bounds[c]["mean"] + (bounds[c]["stddev"] * sdv_multiplier)

    return bounds


def identify_outliers(data, cols=[], method="iqr", **kwargs):
    """Calculate the Outlier Counts per column in the data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list
    method: str
        method used for outlier detection.
        'iqr' and 'sdv' are valid currently
    **kwargs:
        for iqr - iqr_multipler
        for sdv - sdv_multiplier

    Returns
    -------
        bounds: dict()
    """
    num_cols = list(
        set(list_numerical_columns(data))
        - set(list_numerical_categorical_columns(data))
    )
    if not cols:
        cols = num_cols
    else:
        for col in cols:
            assert (  # noqa
                col in num_cols
            ), "{0} is not a valid numerical column in the input data"

    if method == "iqr":
        if "iqr_multiplier" in kwargs.keys():
            bounds = _calculate_outlier_bounds_iqr(
                data, cols, iqr_multiplier=kwargs["iqr_multiplier"]
            )
        else:
            bounds = _calculate_outlier_bounds_iqr(data, cols)

    if method == "sdv":
        if "sdv_multiplier" in kwargs.keys():
            bounds = _calculate_outlier_bounds_sdv(
                data, cols, sdv_multiplier=kwargs["sdv_multiplier"]
            )
        else:
            bounds = _calculate_outlier_bounds_sdv(data, cols)

    return bounds


def treat_outliers_transform(df, drop, cap, bounds, prefix="outlier_"):
    """Treats outliers based on the relevant argument inputs."""
    if drop:
        for col_name in bounds.keys():
            df = df.filter(
                (F.col(col_name) > bounds[col_name]["min_b"])
                & (F.col(col_name) < bounds[col_name]["max_b"])
            )
        return df

    if cap:
        for col_name in bounds.keys():
            df = df.withColumn(
                prefix + col_name,
                F.when(
                    F.col(col_name) < bounds[col_name]["min_b"],
                    bounds[col_name]["min_b"],
                ).otherwise(
                    F.when(
                        F.col(col_name) > bounds[col_name]["max_b"],
                        bounds[col_name]["max_b"],
                    ).otherwise(F.col(col_name))
                ),
            )

    return df


class Outlier_Treatment(Estimator, Transformer):
    """Custom Transformer to treat outliers.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    cols: list
        list of columns to check the outliers
        by default all numerical columns are considered
    drop: bool
        flag to drop or keep the outliers
    cap: bool
        flag to cap the outliers with the bounds
    method: str
        'iqr' or 'sdv'
    prefix: str
        relevant when cap=True
        prefix to be added to the new column where outliers are capped


    """

    def __init__(
        self, cols=[], drop=True, cap=False, method="iqr", prefix="", **kwargs
    ):
        super().__init__()
        self.cols = cols
        self.drop = drop
        self.cap = cap
        self.method = method
        self.prefix = prefix
        self.kwargs = kwargs

    def _fit(self, df: DataFrame):
        self.bounds = handle_outliers(
            df,
            self.cols,
            self.drop,
            self.cap,
            self.method,
            self.prefix,
            **self.kwargs,
        )
        print(self.bounds)
        return self

    def _transform(self, df: DataFrame) -> DataFrame:
        df = treat_outliers_transform(df, self.drop, self.cap, self.bounds)
        return df


# -----------------------------------------------------------------------
# Missing Value Treatment
# -----------------------------------------------------------------------
# FIX ME - Should add the following methods aswell - mice-mean, mice-mode, mice-median, knn for continuous variables
#        - Relevance and availability of these methods in spark has to evaluated
def handle_missing_values(data, cols=[], rules={}):
    """Learning about Missing Values in the train data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame()
        data for which missing values have to be imputed
    columns: list
        default: []
        list of columns where missing values have to be imputed
        considers all columns by default

    rules: {col_name:args_dict}
        default: {}
        Existing SPecified Rules setup for the columns
        By default uses mean for numerical and mode for categorical or bool
        "method" is a compulsary argument in the rule arguments. relevant other arguments have to be provided.

    Returns
    -------
    rules: {col_name:args_dict}
    """

    # FIX ME - there should to be a better way to identify columns wiht missing values
    if not cols:
        missing_df = identify_missing_values(data)
        missing_df = missing_df.toPandas().sum(axis=0)
        cols = list(missing_df[missing_df > 0].index)
    rules = _consolidate_rules(data, cols=cols, rules=rules)
    imputed_data = data
    imputed_dict = {}
    for col_name in cols:
        method = rules[col_name]["method"]
        impute_val = None
        if "impute_val" in rules[col_name].keys():
            impute_val = rules[col_name]["impute_val"]
        imputed_dict[col_name] = _find_impute_val(
            data=imputed_data,
            col_name=col_name,
            method=method,
            impute_val=impute_val,
        )
    return imputed_dict


def identify_missing_values(data):
    """Identify columns in the data with missing Values.

    Parameters
    ----------
        data: pyspark.sql.DataFrame
    Returns
    -------
        df: pyspark.sql.DataFrame
            with columns as the
    """
    try:
        df = data.select(
            [
                F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
                for c in data.columns
            ]
        )
    except Exception:
        date_bool_cols = list_datelike_columns(data) + list_boolean_columns(data)
        for col in date_bool_cols:
            data = data.withColumn(col, F.col(col).cast(DT.StringType()))
        df = data.select(
            [
                F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
                for c in data.columns
            ]
        )
    return df


def _consolidate_rules(data, cols=[], rules={}):
    """Consolidate the Null Imputation Rules in the data.

    Parameters
    ----------
    data: spark.DataFrame
        data for which missing values have to be imputed
    columns: list
        default: []
        list of columns where missing values have to be imputed
        considers all columns by default

    rules: {col_name:args_dict}
        default: {}
        Existing SPecified Rules setup for the columns
        By default uses mean for numerical and mode for categorical or bool
        "method" is a compulsary argument in the rule arguments. relevant other arguments have to be provided.

    Returns
    -------
        rules: {col_name:args_dict}
    """
    if not cols:
        cols = list(data.columns)

    num_cols = list_numerical_columns(data)
    cat_cols = list_categorical_columns(data)
    bool_cols = list_boolean_columns(data)
    date_cols = list_datelike_columns(data)

    for col in cols:
        if col in num_cols:
            col_type = "numerical"
        elif col in cat_cols:
            col_type = "categorical"
        elif col in bool_cols:
            col_type = "boolean"
        elif col in date_cols:
            col_type = "datelike"

        if col not in rules.keys():
            rules[col] = _generate_default_rule(col_type)
    return rules


def _generate_default_rule(col_type):
    """Based on the column type generate the default rule for imputation.

    Parameters
    ----------
        col_type: str
            valid values are "numerical", "categorical", "boolean", "datelike"
    """
    if col_type == "numerical":
        return {"method": "mean"}

    if col_type == "categorical":
        return {"method": "mode"}

    if col_type == "boolean":
        return {"method": "mode"}

    if col_type == "datelike":
        raise NotImplementedError(
            "No Imputer Currently in place for Missing value imputation in a Datelike column"
        )
    else:
        raise ValueError("Invalid column_type - " + col_type)


def _find_impute_val(data, col_name, method, impute_val=None, prefix=""):
    """Impute missing data in a column in the data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    col_name: str
    method: str
        list of valid entries are "mean", "median", "mode", "constant", "regression"
        for categorical and bool types, only mode or constant is applicable
    impute_val: str for categorical
                float for numerical
                bool for boolean
        valid only when method is "constant"
        0 is imputed by default for numerical
        "NA" is imputed for Categorical as well as boolean
    prefix: str
        default: ""
        prefix to be added to the imputed_cols
        be default existing column will be replaced

    Returns
    -------
    df: pyspark.sql.DataFrame
        dataframe with the missing values in the column imputed
    """

    # FIX ME -
    # Relevance of usage of pyspark.ml.feature.Imputer objects here

    valid_methods = ["mean", "median", "mode", "constant", "regression"]
    num_cols = list_numerical_columns(data)
    cat_cols = list_categorical_columns(data)
    bool_cols = list_boolean_columns(data)
    date_cols = list_datelike_columns(data)

    if col_name in cat_cols or bool_cols:
        valid_methods.remove("mean")
        valid_methods.remove("median")
        valid_methods.remove("regression")

    if col_name in date_cols:
        raise NotImplementedError(
            "No Imputer Currently in place for Missing value imputation in a DateLike column"
        )

    if method not in valid_methods:
        raise ValueError(
            "Method specified not relevant for the given column in the data. Please use from the list {}",
            valid_methods,
        )

    if method == "mean":
        x = data.select(F.mean(F.col(col_name)).alias("mean")).collect()
        val = x[0]["mean"]

    if method == "median":
        val = data.approxQuantile(col_name, [0.5], 0)

    if method == "mode":
        x = data.groupby(col_name).count()
        val = x.orderBy(x["count"].desc()).collect()[0][0]
    if method == "constant":
        val = impute_val
        if val is None:
            if col_name in num_cols:
                val = 0
            else:
                val = "NA"

    if method == "regression":
        Exception("WIP -  regression is not yet incorporated in the current version.")

    return val


def _impute_missing_val(data, impute_col_value_dict):
    """Imputed the missing value based on the learning from train data.

    Parameters
    ----------
    data: pyspark.sql.DataFrame
    impute_col_value_dict: dict.
        dictionary that contains, the value to be imputed on each column for missing values.

    Returns
    -------
    df: pyspark.sql.DataFrame
    """
    for col_, val_ in impute_col_value_dict.items():
        data = data.withColumn(
            col_,
            F.when(F.isnan(F.col(col_)) | F.col(col_).isNull(), val_).otherwise(
                F.col(col_)
            ),
        )
    return data


class Imputer(Estimator, Transformer):
    """A custom Transformer imputes missing values.

    Parameters
    ----------
    columns: list
        default: []
        list of columns where missing values have to be imputed
        considers all columns by default

    rules: {col_name:args_dict}
        default: {}
        Existing SPecified Rules setup for the columns
        By default uses mean for numerical and mode for categorical or bool
        "method" is a compulsary argument in the rule arguments. relevant other arguments have to be provided.

    """

    def __init__(self, cols=[], rules={}):
        super().__init__()
        self.cols = cols
        self.rules = rules

    def _fit(self, df: DataFrame):
        self.impute_dict = handle_missing_values(df, self.cols, self.rules)
        print(f"Imputation cols and values:\t{self.impute_dict}")
        return self

    def _transform(self, df: DataFrame) -> DataFrame:
        df = _impute_missing_val(df, self.impute_dict)
        return df


# -----------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------
# from LE codebase
def sampling(
    data,
    target,
    target_type,
    n_rows=35000,
    type="rule_based",
    random_state=42,
    stratify=False,
    max_sample_size=150000,
):
    """Sample the data with two types of sampling.

    First is random sampling. Second is a rule_based, as is implementation of status quo SAS logic.

    Parameters
    ----------
        data: input dataframe
        target: target variable name
        target_type: type of target binary_categorical/continuous
        n_rows: no of observation to be sampled, when type is rule_based this parameter is ignored
        type: type of sampling. This is defaulted to rule based.
        random_state: set seed for reproducibility
        stratify: If the sample should be should be stratified, when type is rule_based this parameter is ignored
        max_sample_size: Maximum responders sample size defaulted to 150K when type is random this parameter is
                         ignored.

    Returns
    -------
        data: sampled data frame
    """
    if type == "random":
        nrow = data.count()
        if target_type == "binary_categorical":
            test_size = 1 - min(n_rows / nrow, 1)
            data, test = test_train_split(
                data,
                target,
                random_state,
                test_size=test_size,
                stratify=stratify,
            )
        elif target_type == "continuous":
            data = data.filter(data[target] > 0)
            sample_ratio = min(n_rows / data.count(), 1.0)
            data = data.filter(data[target] > 0).sample(
                False, sample_ratio, seed=random_state
            )

    if type == "rule_based":
        if target_type == "binary_categorical":
            # max_sample_size = 150000
            offered = data.count()
            responded = data.filter(data[target] == 1).count()
            non_responded = offered - responded
            resp_sample_size = min(responded, max_sample_size)
            non_resp_sample_size = min(resp_sample_size, non_responded)
            if non_resp_sample_size < 1000:
                print("Non responders are {}".format(non_resp_sample_size))
            sample_size = min(resp_sample_size, non_resp_sample_size)
            segment_resp_df = data.filter((F.col(target) == 1))
            segment_non_resp_df = data.filter((F.col(target) == 0))
            resp_sample_ratio = min(sample_size / segment_resp_df.count(), 1)
            non_resp_sample_ratio = min(sample_size / segment_non_resp_df.count(), 1)
            resp_sample_df = segment_resp_df.sample(
                False, resp_sample_ratio, seed=random_state
            )
            non_resp_sample_df = segment_non_resp_df.sample(
                False, non_resp_sample_ratio, seed=random_state
            )
            data = resp_sample_df.union(non_resp_sample_df)
        elif target_type == "continuous":
            # max_sample_size = 150000
            data = data.filter(data[target] > 0)
            sample_ratio = min(max_sample_size / data.count(), 1.0)
            data = data.sample(False, sample_ratio, seed=random_state)

    return data


# -----------------------------------------------------------------------
# Model Data Gen
# -----------------------------------------------------------------------
def generate_features_vector(spark, data, feature_cols, output_col="features"):
    """Generate the features for the train data for the data.

    Parameters
    ----------
        spark: SparkSession
        data: pyspark.sql.DataFrame
        feature_cols: list
            list of columns to be considered as features
        output_col: str
            column name for the output features vector

    Returns
    -------
        df: pyspark.sql.DataFrame
    """
    assembler = VectorAssembler(inputCols=feature_cols, outputCol=output_col)
    df = assembler.transform(data)
    return df


def test_train_split(
    spark,
    data,
    target_col,
    train_prop,
    random_seed=42,
    stratify=False,
    target_type="continuous",
):
    """Test Train Split of the data.

    Parameters
    ----------
    spark: SparkSession
    data: pyspark.sql.DataFrame
    target_col: str
    train_prop: float
        (0-1)proportion of train data
    random_seed: int
        used for seed genration
    stratify: bool
        True represents spilt of data wrt the class in the target_col
    target_type: str
        "continuous"/"categorical"

    Returns
    -------
        train, test
        pyspark.sql.DataFrame, pyspark.sql.DataFrame
    """
    if target_type == "continuous":
        return data.randomSplit([train_prop, 1 - train_prop], seed=random_seed)
    elif target_type == "categorical":
        if stratify is False:
            return data.randomSplit([train_prop, 1 - train_prop], seed=random_seed)
        else:
            test_prop = 1 - train_prop
            # split dataframes between 0s and 1s
            non_events = data.filter(data[target_col] == 0)
            events = data.filter(data[target_col] == 1)
            # split datasets into training and testing
            train_ne, test_ne = non_events.randomSplit(
                [1 - (test_prop / 2), (test_prop / 2)], seed=random_seed
            )
            train_e, test_e = events.randomSplit(
                [1 - (test_prop / 2), (test_prop / 2)], seed=random_seed
            )
            # stack datasets back together
            train = train_ne.union(train_e)
            test = test_ne.union(test_e)
            return (train, test)


def custom_column_name(col_name, data_columns):
    """This function is used to customize column names in a dataframe.

    Which helps in avoiding conflicts caused by having variable names same as column names.

    We add a hash of length 5 to the existing column name for customisation.


    Parameters
    ----------
    col_name: str
    data_columns: list

    Returns
    -------
        name: str
    """
    if col_name in data_columns:
        name = col_name + "".join(random.choices(string.ascii_letters, k=5))
    else:
        name = col_name
    return name
