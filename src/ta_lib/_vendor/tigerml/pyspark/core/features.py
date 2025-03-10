"""Functions to carry out different feature generation/selection/emimination techniques in spark framework."""

import numpy as np
import pandas as pd
from pyspark.ml import Estimator, Pipeline, Transformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (
    Bucketizer,
    MinMaxScaler,
    OneHotEncoder,
    QuantileDiscretizer,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .dp import list_categorical_columns, list_numerical_columns


# -----------------------------
# Categorical Features Imputing
# -----------------------------
# onehot, ordinal, target ......
def encode_categorical_features(data, cols=[], rules=dict(), prefix=""):
    """Encode the Categorical Columns in the data.

    This Encoding expects that the missing values in the categorical columns are handled

    Parameters
    ----------
        spark: SparkSession
        data: pyspark.sql.DataFrame
        target_col: str
            target column in the data
            relevant if target encoding is choosed for any of the columns
        cols: list
            default: []
            list of columns to encode. by default considers the categorical variables in the data
        rules: dict(col_name: rule_dict)
            rule_dict contains method, and the relevant arguments to the method.
            example rule_dicts
                - onehot - {'method':'onehot'}
                - ordinal - {'method':'ordinal','mapper':{<class>:<value_to_be_encoded>}}
                - target - {'method':'target','target_col':<target column name in data>, 'metric':<one of mean, median>}

        prefix: str
            prefix to add to the encoded column - {prefix}_{col_name}
            default - {method}_encoded_{col_name}

    Returns
    -------
        encoded_data: pyspark.sql.DataFrame
            New data with Endoded Colums
    """
    cat_cols = list_categorical_columns(data)
    if not cols:
        cols = cat_cols

    # FIX ME
    # check for missing values in the cols, if so raise error

    for col_name in cols:
        if col_name not in rules.keys():
            rules[col_name] = _generate_default_encoding_rule()
    encoder_list = []
    encoded_data = data
    for col_name in cols:
        print(col_name, rules[col_name])
        method = rules[col_name]["method"]
        target = None
        mapper = dict()
        if "target_col" in rules[col_name].keys():
            target = rules[col_name]["target_col"]

        if "mapper" in rules[col_name].keys():
            mapper = rules[col_name]["mapper"]

        encoder = _encode_categorical_col(
            encoded_data, col_name, method, mapper=mapper, target=target, prefix=prefix
        )
        print(encoder)
        encoder_list += [encoder]

    return encoder_list


def _generate_default_encoding_rule():
    """Create a rule dict with the default method to use for the encoding.Current Default is one_hot_encoding."""
    return {"method": "onehot"}


def _encode_categorical_col(
    data, col_name, method, mapper=dict(), target=None, prefix=""
):
    """Encode the categorical column with the data based on relevant arguments.

    Parameters
    ----------
        spark - SparkSession
        data - pyspark.sql.DataFrame
        col_name - str
        method - str
            valid methods currently are 'one_hot', 'ordinal', 'target'
        mapper - {class:rank}
            valid for ordinal encoding - classes will be ordered by val
        target - str
            target column name - valid for target encoding
        prefix: str
            prefix to add to the encoded column - {prefix}_{col_name}
            default - {method}_encoded_{col_name}

    Returns
    -------
        df - pyspark.sql.DataFrame
    """
    if not prefix:
        prefix = method + "_encoded_"
    if method == "onehot":
        encoder = {}
        encoder["mapper"] = _onehot_encode_fit(data, col_name, prefix)
        encoder["prefix"] = method
        return encoder
    elif method == "ordinal":
        encoder = {"col_name": col_name, "mapper": mapper, "prefix": prefix}
        return encoder
    elif method == "target":
        encoder = {"col_name": col_name}
        encoder["mapper"], encoder["prefix"] = _target_encode(
            data, col_name, target, metric="mean", prefix=""
        )
        return encoder
    else:
        raise ValueError(
            "Invalid method {0} for encoding. Use one of onehot, ordinal, target".format(
                method
            )
        )
    return


def _onehot_encode_fit(data, col_name, prefix):
    """One hot Encode the input column.

    Parameters
    ----------
        data: spark.DataFrame
        col_name: str
            input_column for encoding
        prefix: str
            prefix to add to the output column name
            default - 'onehot_encoded_'{col_name}

    Returns
    -------
        df: spark.dataFrame
            with the encoded column
    """
    list1 = []
    string_indexer = StringIndexer(
        inputCol=col_name, outputCol=col_name + "_index", handleInvalid="keep"
    )
    encoder = OneHotEncoder(
        inputCol=col_name + "_index",
        outputCol=prefix + col_name,
        handleInvalid="keep",
    )
    list1 += [string_indexer]
    list1 += [encoder]
    pipeline = Pipeline(stages=list1)
    indexed_df = pipeline.fit(data)
    return indexed_df


def _onehot_encode_transform(data, indexer):
    df = indexer.transform(data)
    return df


def _ordinal_encode(data, col_name, mapper, prefix):
    """Encode a column given a ordnial mapper dictionery defining ordered priority of the classes in the column.

    Parameters
    ----------
        data: pyspark.sql.DataFrame
        col_name: str
            input_column for encoding
        mapper: dict(class_val:rank)
            mapping dictionery for ordinal encoding
        prefix: str
            prefix to add to the output column name
            by default 'ordinal_encoded_'{col_name}

    Returns
    -------
        df: pyspark.sql.DataFrame
            with the encoded column
    """
    if not mapper:
        raise ValueError("Mapper cant be {0} for Ordinal Encoding".format(str(mapper)))

    # check if all classes in the data present in the mapper
    from itertools import chain
    from pyspark.sql.functions import create_map, lit

    mapping_expr = create_map([lit(x) for x in chain(*mapper.items())])

    df = data.withColumn(
        prefix + col_name, mapping_expr.getItem(F.col(col_name)).cast("float")
    )
    return df


def _target_encode(data, col_name, target, metric="mean", prefix=""):
    """Target encode a categorical feature based on the metric of the target variable.

    Parameters
    ----------
        data: pyspark.sql.DataFrame
        col_name: str
            input_column for encoding
        target: str
            target_col in the data. should be numerical
        metric: str
            metric to consider in the target variable for encoding. valid values are "mean", "median"
        prefix: str
            prefix to add to the output column name
            by default 'target_encoded_'{col_name}

    Returns
    -------
        df: pyspark.sql.dataFrame
            with the encoded column
    """
    if metric == "mean":
        mapper = dict(
            data.groupby(col_name)
            .agg(F.mean(F.col(target)).alias("encoding_val"))
            .rdd.collectAsMap()
        )
    elif metric == "median":
        mapper = dict(
            data.groupBy(col_name)
            .agg(F.expr("percentile_approx(" + target + ", 0.5)").alias("encoding_val"))
            .rdd.collectAsMap()
        )
    else:
        raise ValueError(
            "Invalid metric {0} for target encoding. Use one of mean, median".format(
                metric
            )
        )

    if not prefix:
        prefix = "target_" + metric + "_encoded"

    return mapper, prefix


# onehot, ordinal, target ......
class Encoder(Estimator, Transformer):
    """Custom Transformer to encode Categorical features."""

    def __init__(self, cols=[], rules={}, prefix=""):
        super().__init__()
        self.cols = cols
        self.rules = rules
        self.prefix = prefix

    def _fit(self, df: DataFrame):
        self.encoders = encode_categorical_features(
            df, self.cols, self.rules, self.prefix
        )
        return self

    def _transform(self, df: DataFrame) -> DataFrame:
        for enc_ in self.encoders:
            if "onehot" in enc_["prefix"]:
                df = _onehot_encode_transform(df, enc_["mapper"])
            else:
                df = _ordinal_encode(
                    df, enc_["col_name"], enc_["mapper"], enc_["prefix"]
                )

        return df


# --------------
# Binning
# --------------
# TBD: supervised binning currently not supported in pyspark, need to think a better way
def binning(data, col_name, n=None, value=None, size=None, left=True, target=None):
    """Bin values into discrete intervals.

    Parameters
    ----------
        n         : None(default); integer;
                        n number of equal-sized buckets
        value     : None(default); array of numbers(int/float);
                        Buckets produced based on values provided
        size      : None(default); number(integer/float);
                        Size of each interval/buckets,
                        size = (upper bound-lower bound) of bucket
        left      : True(default); boolean;
                        If True buckets' size counting will start from
                        left, else from right
        target:   : Target variable for supervised binning

    Returns
    -------
        bucketizer: pyspark.ml.feature.bucketizer object that should be fit on data

    """
    # FIXME: Handle Key Error Runtime Exception
    try:
        if n:
            # n number of bins
            bucketizer = QuantileDiscretizer(
                numBuckets=n,
                inputCol=col_name,
                outputCol=f"bucketed_{col_name}",
                relativeError=0.01,
                handleInvalid="error",
            )
        elif value:
            # Actual value specified on which intervals has to be created
            bucketizer = Bucketizer(
                splits=value, inputCol=col_name, outputCol=f"bucketed_{col_name}"
            )
        elif size:
            # Size of interval is specified
            describe_df = data.select(col_name).describe().toPandas()
            if left:
                # Left TRUE: intervals will be created starting from left
                # Array initialized with minimum value
                value = [-float("Inf")]
                # First cut point will be one size above the minimum value
                temp = (
                    float(
                        describe_df[describe_df.summary == "min"][col_name].tolist()[0]
                    )
                    + size
                )
                max_ = float(
                    describe_df[describe_df.summary == "max"][col_name].tolist()[0]
                )
                while temp < max_:
                    # Iteratively appending cut points with specified size
                    value.append(temp)
                    temp = temp + size
                # Right side is bounded with highest value
                # Right-most interval will not necessarily have same size
                value.append(float("Inf"))
            else:
                # Left FALSE: intervals will be created starting from right
                # Array initialized with maximum value
                value = [float("Inf")]
                # First cut point will be one size below the maximum value
                temp = (
                    describe_df[describe_df.summary == "max"][col_name].tolist()[0]
                    - size
                )
                min_ = describe_df[describe_df.summary == "min"][col_name].tolist()[0]
                while temp > min_:
                    # Iteratively appending cut points with specified size
                    value.append(temp)
                    temp = temp - size
                # Left side is bounded with lowest value
                # Left-most interval will not necessarily have same size
                value.append(-float("Inf"))
                # Value array is created in descending order, it is needed
                # to be reversed
                value.reverse()
            bucketizer = Bucketizer(
                splits=value, inputCol=col_name, outputCol=f"bucketed_{col_name}"
            )
        elif target:
            # Yet to implement
            raise Exception("Currently supervised binning is not supported")
        else:
            print("Input not proper")
            return None
        return bucketizer
    except ValueError as e:
        print(e)


def wrapper_binning(data, cols=[], rules={}):
    """Wrap the Binning Methods."""
    if len(cols) and len(rules.keys()):
        bucketizer_list = []
        for col_ in cols:
            if "n" in rules[col_].keys():
                bucketizer = binning(data, col_, n=rules[col_]["n"])
            elif "value" in rules[col_].keys():
                bucketizer = binning(data, col_, value=rules[col_]["value"])
            elif "size" in rules[col_].keys():
                bucketizer = binning(data, col_, size=rules[col_]["size"])
                if "left" in rules[col_].keys():
                    bucketizer = binning(
                        data, col_, size=rules[col_]["size"], left=rules[col_]["left"]
                    )
                else:
                    bucketizer = binning(data, col_, size=rules[col_]["size"])
            bucketizer_list += [bucketizer]
        pipeline = Pipeline(stages=bucketizer_list)
        indexed_df = pipeline.fit(data)
        return indexed_df


class Binner(Estimator, Transformer):
    """Transformer on Binning methods.

    Parameters
    ----------
        cols: list
            default: []
            list of columns to bin.
        rules: dict(col_name: rule_dict)
            rule_dict contains type of bin, Quantile or bin by values, and the relevant arguments to the method.
            example rule_dicts
                {'median_house_value':
                {'n': n equal quantile buckets
                'value': array of numbers(int/float);
                        Buckets produced based on values provided
                'size': Size of each interval/buckets,

                'left': True(default); boolean;
                        If True buckets' size counting will start from
                        left, else from right
                }}

    """

    def __init__(self, cols=[], rules={}):
        super().__init__()
        self.cols = cols
        self.rules = rules

    def _fit(self, df: DataFrame):
        self.binner = wrapper_binning(df, self.cols, self.rules)
        return self

    def _transform(self, df: DataFrame):
        return self.binner.transform(df)


# --------------
# Feature Selection/ Elimination
# --------------
# from LE codebase


class FeatureEliminator:  # Estimator,Transformer
    """Class for Feature Elimiations.

    Parameters
    ----------
        spark: pyspark.sql.session.SparkSession
             spark session object
        dataset: spark.DataFrame

    """

    def __init__(self, spark, dataset, target_col, target_type="continuous"):
        # super().__init__()

        self.target_col = target_col
        self.target_type = target_type
        self.dataset = dataset
        self.spark = spark

    def transform(self, method, **kwargs):
        """Transform the function."""
        if method == "sparseness":
            return feature_elimination_by_sparseness(self.dataset, **kwargs)
        elif method == "cv":
            return feature_elimination_by_cv(self.dataset, **kwargs)
        elif method == "missing_values":
            return feature_elimination_by_missing_values(self.dataset, **kwargs)
        elif method == "correlation":
            return feature_elimination_by_correlation(
                self.dataset, target_col=self.target_col, **kwargs
            )
        elif method == "mutual_value":
            return feature_elimination_by_mutual_value(
                self.spark, self.dataset, target_col=self.target_col, **kwargs
            )
        elif method == "lasso":
            return feature_elimination_by_lasso(
                self.dataset,
                target_col=self.target_col,
                target_type=self.target_type,
                **kwargs,
            )
        else:
            raise ValueError(
                f"method {method} is not defined, chose among [sparseness,cv,missing_values,correlation,mutual_value,lasso]"
            )


def feature_elimination_by_sparseness(data, threshold=0.1):
    """Feature Elimination based on the percentage of zero values.

    Parameters
    ----------
        data (pyspark.sql.dataframe.DataFrame): input dataframe
        threshold (float): A number between 0 & 1. Features with zero values
                           density > threshold will be returned.

    Returns
    -------
        feature_list (list): list of features to be dropped
    """
    cols = list_numerical_columns(data)
    nrow = data.count()

    z = data.select([(F.count(F.when(F.col(c) == 0, c)) / nrow).alias(c) for c in cols])
    z = (z.collect()[0]).asDict()
    sparse_cols = [c for c in z if z[c] >= threshold]
    return sparse_cols


def feature_elimination_by_cv(data, threshold=0.001):
    """Feature Elimination based on the coefficient of variation.

    Parameters
    ----------
        threshold (float): Features with cv < threshold will be returned.
        data (pyspark.sql.dataframe.DataFrame): data frame to eliminate
        features

    Returns
    -------
        feature_list (list): list of features to be dropped
    """
    cols = list_numerical_columns(data)

    z = data.select([(F.stddev(c) / F.mean(c)).alias(c) for c in cols])
    z = (z.collect()[0]).asDict()
    lowvar_cols = [c for c in z if z[c] is not None if z[c] <= threshold]
    return lowvar_cols


def feature_elimination_by_missing_values(data, threshold=0.1):
    """Feature Elimiation based on the percentage of missing values.

    Parameters
    ----------
        threshold (float): A number between 0 & 1. Features with missing
        values density > threshold will be returned.
        data (pyspark.sql.dataframe.DataFrame): data frame to eliminate features

    Returns
    -------
        feature_list (list): list of features to be dropped
    """

    cols = data.columns
    nrow = data.count()

    z = data.select(
        [(F.count(F.when(F.col(c).isNull(), c)) / nrow).alias(c) for c in cols]
    )
    z = (z.collect()[0]).asDict()
    null_cols = [c for c in z if z[c] >= threshold]
    return null_cols, z


def feature_elimination_by_correlation(data, target_col="target", threshold=0.1):
    """Feature Elimiation based on the absolute pearson's correlation coeff with target variable.

    WARNING: We are calculating pairwise correlation, can be optimized if we do only for feature vs target.

    Parameters
    ----------
        threshold (float): A number between 0 & 1. Features with absolute
        pearson's correlation coeff < threshold will be returned.
        data (pyspark.sql.dataframe.DataFrame): data frame

    Returns
    -------
        feature_list (list): list of features to be dropped
    """
    cols = list_numerical_columns(data)
    assembler = VectorAssembler(
        inputCols=cols, outputCol="features", handleInvalid="keep"
    )
    df_vector = assembler.transform(data).select("features")
    corr_mat = Correlation.corr(df_vector, "features", method="pearson")
    corr_mat = corr_mat.collect()[0].asDict()["pearson(features)"]
    corr_df = pd.DataFrame(corr_mat.toArray())
    corr_df.index, corr_df.columns = cols, cols
    corr_df = corr_df.abs()
    corr_df.reset_index(inplace=True)
    cols_to_exclude = corr_df[corr_df[target_col] <= threshold]["index"].tolist()
    return cols_to_exclude


def feature_elimination_by_mutual_value(
    spark, data, threshold=0.1, target_col="response"
):
    """Feature Elimination by mutual value information with target variable.

    Parameters
    ----------
        threshold (float): A number between 0 & 1. Features with mutual information < threshold will be returned.
        data (pyspark.sql.dataframe.DataFrame): data frame

    Returns
    -------
        feature_list (list): list of features to be dropped
    """

    def mi_val(job):
        """Calculate the mi value of a column wrt target.

        Parameters
        ----------
            job (list): List of parameters
                target: Name of the Target variable.
                bucket_col: Name of the column for which MI needs to be calculated
                num_flag: Is bucket column numeric, if yes then it needs to be bucketized
                df: Pandas dataframe with Two column Target & bucket_col
                events_counts: Number of responders in df
                non_events_counts: Number of non responders in df

        Returns
        -------
            Dict: {bucket_col: MI Value}
        """
        import pandas as pdx

        target, bucket_col, num_flag, df, events_counts, non_events_counts = job
        if num_flag:
            buckets = pdx.qcut(
                df[bucket_col].astype(float), 5, labels=False, duplicates="drop"
            )
            df = pdx.concat([df[[target]], buckets], axis=1)

        event_col_per = pdx.DataFrame(
            dict(df[df[target] == 1][bucket_col].value_counts()).items(),
            columns=["bucket", "event_col_per"],
        )
        event_col_per["event_col_per"] = event_col_per["event_col_per"] / events_counts

        non_event_col_per = pdx.DataFrame(
            dict(df[df[target] == 0][bucket_col].value_counts()).items(),
            columns=["bucket", "non_event_col_per"],
        )
        non_event_col_per["non_event_col_per"] = (
            non_event_col_per["non_event_col_per"] / non_events_counts
        )

        col_target_per = event_col_per.merge(non_event_col_per, on=["bucket"])
        col_target_per["WoE"] = np.log(
            col_target_per["non_event_col_per"] / col_target_per["event_col_per"]
        )
        col_target_per["IV"] = (
            col_target_per["non_event_col_per"] - col_target_per["event_col_per"]
        ) * col_target_per["WoE"]
        return [bucket_col, col_target_per.IV.sum()]

    cols = data.columns
    num_cols = list_numerical_columns(data)
    cols = [c for c in cols if c != target_col]
    num_cols = [c for c in num_cols if c != target_col]
    events_counts = max(data.filter(data[target_col] == 1).count(), 0.001)
    non_events_counts = max(data.filter(data[target_col] == 0).count(), 0.001)

    # Creating Jobs to parallelize
    jobs = []
    data_df = data.toPandas()
    for c in cols:
        if c in num_cols:
            jobs.append(
                [
                    target_col,
                    c,
                    True,
                    data_df[[target_col, c]],
                    events_counts,
                    non_events_counts,
                ]
            )
        else:
            jobs.append(
                [
                    target_col,
                    c,
                    False,
                    data_df[[target_col, c]],
                    events_counts,
                    non_events_counts,
                ]
            )

    rdd = spark.sparkContext.parallelize(jobs, 400)
    iv = rdd.map(mi_val).collect()
    cols_to_exclude = [c[0] for c in iv if c[1] <= threshold]
    return cols_to_exclude


def feature_elimination_by_lasso(
    data, target_col="target", target_type="continuous", alpha=1
):
    """Elimate Features based on lasso/linear/logit regression.

    Parameters
    ----------
        data (pyspark.sql.dataframe.DataFrame): data frame
        target_col (string): target variable
        target_type (string): type target variable binary_categorical/continuous
        alpha (float): A number between 0 & 1.

    Returns
    -------
        feature_list (list): list of features to be dropped
    """
    cols = data.columns
    num_cols = list_numerical_columns(data)
    cols = [c for c in cols if c != target_col]
    num_cols = [c for c in num_cols if c != target_col]

    assembler = VectorAssembler(
        inputCols=num_cols, outputCol="features", handleInvalid="skip"
    )
    model_data = assembler.transform(data)
    model_data = model_data.select(["features", target_col])
    if target_type == "binary_categorical":
        model = LogisticRegression(
            labelCol=target_col,
            featuresCol="features",
            elasticNetParam=1,
            regParam=alpha,
        )
    elif target_type == "continuous":
        model = LinearRegression(
            labelCol=target_col,
            featuresCol="features",
            elasticNetParam=1,
            regParam=alpha,
        )
    model = model.fit(model_data)

    coeff_ = model.coefficients.toArray()
    cols_to_exclude = [c for i, c in enumerate(num_cols) if coeff_[i] == 0]
    return cols_to_exclude


# --------------
# Scaling or Normalising Data
# --------------
def scale_data(spark, data, column="features", method="min_max", prefix="scaled_"):
    """Min Max scaling of numerical columns.

    Parameters
    ----------
        spark: SParkSession
        data: pyspark.sql.DataFrame
        column: str
            column name of the features vector.
        method: string
            scaling method 'min_max' or 'standard'
        prefix: str
            prefic string to be added to the new column name

    Returns
    -------
        df: pyspark.sql.DataFrame
            with the scaled features
    """
    if method == "min_max":
        scaler = MinMaxScaler(inputCol=column, outputCol=prefix + column)
    elif method == "standard":
        scaler = StandardScaler(inputCol=column, outputCol=prefix + column)
    else:
        raise ValueError(
            "Invalid method {method} for scaling. Use one of 'min_max', 'standard'"
        )
    scaler_model = scaler.fit(data)
    return scaler_model.transform(data)
