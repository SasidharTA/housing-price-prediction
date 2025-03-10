import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis.extra.pandas import column, columns, data_frames, range_indexes
from hypothesis.strategies import (
    booleans,
    composite,
    data,
    dates,
    floats,
    integers,
    lists,
    sampled_from,
    tuples,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DataType,
    FloatType,
    IntegerType,
    LongType,
    Row,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_iris,
)
from tigerml.pyspark.core import dp
from tigerml.pyspark.core.dp import (
    list_categorical_columns,
    list_numerical_columns,
)
from tigerml.pyspark.model_eval import (
    ClassificationReport,
    RegressionReport,
    get_binary_classification_metrics,
    get_regression_metrics,
)
from tigerml.pyspark.model_eval.model_eval import (
    exp_var,
    get_classification_scores,
    mape,
    wmape,
)

# import findspark
# findspark.init()  # to make sure SPARK_HOME env variable is set correctly

# import os
# os.environ['SPARK_HOME'] = "D:/GitHub_AR/pyspark_trials/Installations/spark-3.0.1"  # path to Spark binaries
# os.environ['HADOOP_HOME'] = "D:/GitHub_AR/pyspark_trials/Installations/hadoop27"


@composite
def y_pred_df(draw, required_cols='all', non_empty_row='have', have_na=False, is_classification=False):
    """ An hypothesis.composite strategy for creating y_prediction pandas_df."""
    if required_cols == 'all':
        required_cols = ['y_col', 'y_pred_col']
    float_options = floats(allow_infinity=False, allow_nan=False)
    if is_classification:
        float_options = sampled_from([0, 1])
    if have_na:
        float_options = floats(allow_infinity=False, allow_nan=True)
        if is_classification:
            float_options = sampled_from([0, 1, np.NAN])

    column_list = []
    if 'y_col' in required_cols:
        # column_list += [column('float_col', elements=floats(allow_infinity=True, allow_nan=True))]
        column_list += [column('y_col', elements=float_options)]
    if 'y_pred_col' in required_cols:
        column_list += [column('y_pred_col', elements=float_options)]

    df = draw(data_frames(index=range_indexes(min_size=3), columns=column_list))
    if non_empty_row == 'have':
        non_empty_row = True
    else:
        non_empty_row = False  # draw(sampled_from([True, False]))
    if non_empty_row:
        additional_values = {'y_col': [5.7, 2.4, 3.2], 'y_pred_col': [6.7, 3.4, 4.2]}
        if is_classification:
            additional_values = {'y_col': [0, 1, 1], 'y_pred_col': [0, 1, 1]}

        for i in range(3):
            new_row = {}
            for col in required_cols:
                new_row[col] = additional_values[col][i]
            df = df.append(pd.Series(new_row), ignore_index=True)
    return df


def get_ml_data(classification=True, multiclass=False):
    if classification:
        if multiclass:
            data_ = load_iris()
        else:
            data_ = load_breast_cancer()
    else:
        data_ = fetch_california_housing()
    X = pd.DataFrame(data_['data'], columns=data_['feature_names'])
    y = pd.DataFrame(data_['target'], columns=['label'])
    df = pd.concat([X, y], axis=1)
    return df


def get_spark_data(spark, df, y_col='label', classification=False, train_prop=0.7, stratify=True):
    df = spark.createDataFrame(df)
    target_type = "categorical" if classification else "continuous"
    train_df, test_df = dp.test_train_split(spark, data=df, target_col=y_col, train_prop=train_prop, random_seed=1234,
                                            stratify=stratify, target_type=target_type)
    return train_df, test_df


@composite
def get_testing_data(draw, is_classification, spark):
    train_prop = draw(sampled_from([0.7, 0.8, 0.9]))
    df = get_ml_data(classification=is_classification)
    train_df, test_df = get_spark_data(spark, df, classification=is_classification, train_prop=train_prop)
    return train_df, test_df


def pandas_to_spark(spark, pandas_df):
    """ Will return a spark dataframe for given pandas dataframe."""
    def equivalent_type(f):
        if f == 'datetime64[ns]':
            return TimestampType()
        elif f == 'int64':
            return LongType()
        elif f == 'int32':
            return IntegerType()
        elif f == 'float64':
            return FloatType()
        else:
            return StringType()

    cols = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for col, type_ in zip(cols, types):
        struct_list.append(StructField(col, equivalent_type(type_)))
    p_schema = StructType(struct_list)
    return spark.createDataFrame(pandas_df, p_schema)


# Function to give meaningful test_ids for all the test combinations
def id_func(param):
    id_dict = {'T1': 'DfShape_test', 'T2': 'ListLength_test', 'T3': 'DfValue_test', 'T4': 'Result_gen_test',
               'True': 'Columns_with_na', 'False': 'Columns_without_na', 'Only_nums_df': 'Only_nums_df',
               'Only_cat_df': 'Only_cat_df'}
    return id_dict[str(param)]


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T1"],
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=y_pred_df(required_cols='all'))
def test_get_regression_metrics(test_df, n_threads, n_partitions, test_type):
    """ test for get_regression_metrics function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = get_regression_metrics(spark_session, test_sdf, 'y_col', 'y_pred_col')
    if test_type == "T1":
        assert results.shape == (7, 2)


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2"])
@settings(max_examples=10, deadline=None)
@given(test_df=y_pred_df(required_cols='all'))
def test_wmape(test_df, n_threads, n_partitions):
    """ test for wmape function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = wmape(spark_session, test_sdf, 'y_col', 'y_pred_col')


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2"])
@settings(max_examples=10, deadline=None)
@given(test_df=y_pred_df(required_cols='all'))
def test_mape(test_df, n_threads, n_partitions):
    """ test for mape function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = mape(spark_session, test_sdf, 'y_col', 'y_pred_col')


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2"])
@settings(max_examples=10, deadline=None)
@given(test_df=y_pred_df(required_cols='all'))
def test_exp_var(test_df, n_threads, n_partitions):
    """ test for exp_var function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = exp_var(spark_session, test_sdf, 'y_col', 'y_pred_col')


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T1"],
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=y_pred_df(required_cols='all', is_classification=True))
def test_get_binary_classification_metrics(test_df, n_threads, n_partitions, test_type):
    """ test for get_binary_classification_metrics function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = get_binary_classification_metrics(spark_session, test_sdf, 'y_col', y_pred_col='y_pred_col')
    if test_type == "T1":
        assert results.shape == (8, 2)


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("test_type",
                         ["T1"],
                         ids=id_func)
@settings(max_examples=10, deadline=None)
@given(test_df=y_pred_df(required_cols='all', is_classification=True))
def test_get_classification_scores(test_df, n_threads, n_partitions, test_type):
    """ test for get_classification_scores function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    test_sdf = pandas_to_spark(spark_session, test_df)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = get_classification_scores(test_sdf, 'y_col', y_pred_col='y_pred_col')
    if test_type == "T1":
        assert len(results) == 3


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("report_type",
                         [1])
@settings(max_examples=10, deadline=None)
@given(test_df=data())
def test_get_regression_report(test_df, n_threads, n_partitions, report_type):
    """ test for get_regression_report function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    df_train, df_test = test_df.draw(get_testing_data(is_classification=False, spark=spark_session))
    if report_type == 1:
        reg_obj1 = RegressionReport(spark=spark_session, train_df=df_train, label_Col='Target', model=LinearRegression(),
                                    test_df=df_test)
        reg_obj1.get_report(feature_cols=['CRIM', 'ZN', 'AGE'])


@pytest.mark.parametrize("n_partitions",
                         ["n_partitions1", "n_partitions2"])
@pytest.mark.parametrize("n_threads",
                         ["n_threads2", "n_threads3", "n_threads4"])
@pytest.mark.parametrize("report_type",
                         [1])
@settings(max_examples=10, deadline=None)
@given(test_df=data())
def test_get_classification_report(test_df, n_threads, n_partitions, report_type):
    """ test for get_classification_report function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = SparkSession.builder.master(threads_str).appName("pyspark.eda_local_testing").getOrCreate()
    df_train, df_test = test_df.draw(get_testing_data(is_classification=True, spark=spark_session))
    if report_type == 1:
        cls_obj1 = ClassificationReport(spark=spark_session, train_df=df_train, label_Col='Target',
                                        model=LogisticRegression(), test_df=df_test)
        cls_obj1.get_report(feature_cols=['mean radius', 'mean texture'])
