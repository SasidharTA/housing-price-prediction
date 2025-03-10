import holoviews as hv
import os
import pandas as pd
import pathlib
import pyspark
import pytest
import random
from hypothesis import HealthCheck, given, settings
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
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tigerml.pyspark.core.dp import (
    list_categorical_columns,
    list_numerical_categorical_columns,
    list_numerical_columns,
    clean_columns,
)
from tigerml.pyspark.eda.eda import (
    EDAReportPyspark,
    _missing_values,
    column_values_summary,
    correlation_table,
    correlation_with_target,
    data_health_recommendations,
    density_plots_numerical,
    describe_categoricaldata,
    describe_data,
    feature_analysis_pca,
    feature_analysis_table,
    feature_density_plots,
    feature_importance,
    get_bivariate,
    get_datatypes,
    get_missing_values_summary,
    get_outliers_table,
    health_analysis,
    setanalyse,
)

# import findspark
# findspark.init()  # to make sure SPARK_HOME env variable is set correctly

# import os
# os.environ['SPARK_HOME'] = "D:/GitHub_AR/pyspark_trials/Installations/spark-3.0.1"  # path to Spark binaries
# os.environ['HADOOP_HOME'] = "D:/GitHub_AR/pyspark_trials/Installations/hadoop27"

threads = ["n_threads2", "n_threads3", "n_threads4"]
partitions = ["n_partitions1", "n_partitions2"]


@composite
def df_with_required_cols(
    draw, required_cols="all", non_empty_row="have", have_na=False
):
    """An hypothesis.composite strategy for creating generic pandas_df."""
    if required_cols == "all":
        required_cols = ["int_col", "float_col", "cat_col", "dt_col", "bool_col"]
    date_options = [
        pd.Timestamp("1/1/1995"),
        pd.Timestamp("1/1/2005"),
        pd.Timestamp("1/1/2015"),
    ]
    float_options = floats(allow_infinity=False, allow_nan=False)
    string_options = ["red", "Blue", "GREEN"]
    bool_options = [True, False]
    if have_na:
        date_options += [pd.NaT]
        float_options = floats(allow_infinity=False, allow_nan=True)
        string_options += ["NA", None]
        bool_options += [None]

    column_list = []
    if "int_col" in required_cols:
        column_list += [column("int_col", elements=integers())]
    if "int_col2" in required_cols:
        column_list += [column("int_col2", elements=integers())]
    if "float_col" in required_cols:
        # column_list += [column('float_col', elements=floats(allow_infinity=True, allow_nan=True))]
        column_list += [column("float_col", elements=float_options)]
    if "float_col2" in required_cols:
        column_list += [column("float_col2", elements=float_options)]
    if "cat_col" in required_cols:
        column_list += [column("cat_col", elements=sampled_from(string_options))]
    if "cat_col2" in required_cols:
        column_list += [column("cat_col2", elements=sampled_from(string_options))]
    if "dt_col" in required_cols:
        column_list += [column("dt_col", elements=sampled_from(date_options))]
    if "dt_col2" in required_cols:
        column_list += [column("dt_col2", elements=sampled_from(date_options))]
    if "bool_col" in required_cols:
        column_list += [column("bool_col", elements=sampled_from(bool_options))]
    if "bool_col2" in required_cols:
        column_list += [column("bool_col2", elements=sampled_from(bool_options))]

    df = draw(data_frames(index=range_indexes(min_size=3), columns=column_list))
    if non_empty_row == "have":
        non_empty_row = True
    else:
        non_empty_row = False  # draw(sampled_from([True, False]))
    if non_empty_row:
        additional_values = {
            "int_col": [57, 24, 32],
            "int_col2": [67, 34, 42],
            "float_col": [5.7, 2.4, 3.2],
            "float_col2": [6.7, 3.4, 4.2],
            "cat_col": ["red", "Blue", "GREEN"],
            "cat_col2": ["red", "Blue", "GREEN"],
            "dt_col": [
                pd.Timestamp("1/1/1965"),
                pd.Timestamp("1/1/1975"),
                pd.Timestamp("1/1/1985"),
            ],
            "dt_col2": [
                pd.Timestamp("1/1/1965"),
                pd.Timestamp("1/1/1975"),
                pd.Timestamp("1/1/1985"),
            ],
            "bool_col": [True, False, False],
            "bool_col2": [True, False, False],
        }

        for i in range(3):
            new_row = {}
            for col in required_cols:
                new_row[col] = additional_values[col][i]
            df = df.append(pd.Series(new_row), ignore_index=True)
    return df


def pandas_to_spark(spark, pandas_df):
    """Will return a spark dataframe for given pandas dataframe."""

    def equivalent_type(f):
        if f == "datetime64[ns]":
            return TimestampType()
        elif f == "int64":
            return LongType()
        elif f == "int32":
            return IntegerType()
        elif f == "float64":
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
    id_dict = {
        "T1": "DfShape_test",
        "T2": "ListLength_test",
        "T3": "DfValue_test",
        "True": "Columns_with_na",
        "False": "Columns_without_na",
        "Only_nums_df": "Only_nums_df",
        "Only_cat_df": "Only_cat_df",
    }
    return id_dict[str(param)]


# Fixture for assigning the type of feature columns (called by @pytest.mark.parametrize in each test case)
@pytest.fixture(
    params=["Mixed_df", "Only_nums_df", "Only_cat_df", "Only_dt_df", "Only_bool_df"]
)
def df_type(request):
    df_type_dict = {
        "Mixed_df": ["int_col", "float_col", "cat_col", "dt_col", "bool_col"],
        "Mixed_df2": [
            "int_col",
            "float_col",
            "cat_col",
            "dt_col",
            "bool_col",
            "int_col2",
            "float_col2",
            "cat_col2",
            "dt_col2",
            "bool_col2",
        ],
        "Only_nums_df": ["int_col", "float_col"],
        "Only_cat_df": ["cat_col"],
        "Only_dt_df": ["dt_col"],
        "Only_bool_df": ["bool_col"],
    }
    return df_type_dict[request.param]


# Generating paths of the csv files from data folder
def generate_data_paths():
    directory = "data"
    paths_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and f[-4:] == ".csv":
            paths_list.append(f)
    return paths_list


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads1"])
@pytest.mark.parametrize("test_type", ["T3"], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_setanalyse_(value, n_threads, n_partitions, test_type):
    """Test for setanalyse function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = setanalyse(test_sdf_cleaned, test_sdf_cleaned, test_sdf_cleaned.columns)
    if test_type == "T3":
        assert results["A-B"] + results["B-A"] == 0


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads1"])
@pytest.mark.parametrize("test_type", ["T1"], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_column_values_summary(value, n_threads, n_partitions, test_type):
    """Test for column_values_summary function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = column_values_summary(test_sdf_cleaned)
    if test_type == "T1":
        assert results.shape == (3, len(test_sdf_cleaned.columns)) and isinstance(
            results, pd.core.frame.DataFrame
        )


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads1"])
@pytest.mark.parametrize("test_type", ["T2"], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_get_datatypes(value, n_threads, n_partitions, test_type):
    """Test for get_datatypes function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = get_datatypes(test_sdf_cleaned)
    if test_type == "T2":
        assert len(results[0]) + len(results[1]) == len(test_sdf_cleaned.columns)


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("test_type", ["T2"], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_get_missing_values_summary(value, n_threads, n_partitions, test_type):
    """Test for missing_values_summary function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = get_missing_values_summary(test_sdf_cleaned)
    if test_type == "T2":
        assert list(results.columns) == [0]


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("test_type", ["T2"], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_missing_values(value, n_threads, n_partitions, test_type):
    """Test for _missing_values function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = _missing_values(test_sdf_cleaned)
    if test_type == "T2":
        assert list(results.columns) == [
            "Variable Name",
            "No of Missing",
            "Percentage Missing",
        ] and isinstance(results, pd.core.frame.DataFrame)


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("test_type", ["T2"], ids=id_func)
@pytest.mark.parametrize("get_index", [False, True], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_get_outliers_table(value, n_threads, n_partitions, test_type, get_index):
    """Test for get_outliers_table function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    if get_index:
        results = get_outliers_table(test_sdf_cleaned, True)
    else:
        results = get_outliers_table(test_sdf_cleaned, False)
    if test_type == "T2":
        if get_index:
            assert results.columns.tolist() == [
                "< (mean-3*std)",
                "> (mean+3*std)",
                "< (1stQ - 1.5 * IQR)",
                "> (3rdQ + 1.5 * IQR)",
                "< (mean-3*std) Index",
                "> (mean+3*std) Index",
                "< (1stQ - 1.5 * IQR) Index",
                "> (3rdQ + 1.5 * IQR) Index",
            ] and isinstance(results, pd.core.frame.DataFrame)
        else:
            assert results.columns.tolist() == [
                "< (mean-3*std)",
                "> (mean+3*std)",
                "< (1stQ - 1.5 * IQR)",
                "> (3rdQ + 1.5 * IQR)",
            ] and isinstance(results, pd.core.frame.DataFrame)


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("test_type", ["T2"], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_describe_data(value, n_threads, n_partitions, test_type):
    """Test for describe_data function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = describe_data(test_sdf_cleaned, columns=list_numerical_columns(test_sdf_cleaned))
    print(results.columns.tolist())
    if test_type == "T2":
        assert results.columns.tolist() == [
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
        ] and isinstance(results, pd.core.frame.DataFrame)


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("test_type", ["T2"], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_describe_categoricaldata(value, n_threads, n_partitions, test_type):
    """Test for describe_categoricaldata function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    results = describe_categoricaldata(
        test_sdf_cleaned, cat_cols=list_categorical_columns(test_sdf_cleaned)
    )
    if test_type == "T2":
        assert results.columns.tolist() == [
            "nunique",
            "samples",
            "mode",
            "mode_freq",
        ] and isinstance(results, pd.core.frame.DataFrame)


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("test_type", ["T2"], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_feature_analysis_table(value, n_threads, n_partitions, test_type):
    """Test for feature_analysis_table function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    numerical_description, categorical_description = feature_analysis_table(test_sdf_cleaned)
    if test_type == "T2":
        assert (
            numerical_description.columns.tolist()
            == [
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
            and categorical_description.columns.tolist()
            == ["nunique", "samples", "mode", "mode_freq"]
            and isinstance(categorical_description, pd.core.frame.DataFrame)
            and isinstance(numerical_description, pd.core.frame.DataFrame)
        )


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("test_type", ["T2"], ids=id_func)
@pytest.mark.parametrize("value", generate_data_paths())
def test_correlation_table(value, n_threads, n_partitions, test_type):
    """Test for correlation_table function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    result = correlation_table(test_sdf_cleaned, plot="table")
    if test_type == "T2":
        assert result.columns.tolist() == ["var1", "var2", "corr_coef"] and isinstance(
            result, pd.core.frame.DataFrame
        )


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("value", generate_data_paths())
def test_density_plots_numerical(value, n_threads, n_partitions):
    """Test for density_plots_numerical function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    result = density_plots_numerical(test_sdf)
    assert list(result.keys()) == list(
        set(list_numerical_columns(test_sdf))
        - set(list_numerical_categorical_columns(test_sdf))
    )


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("value", generate_data_paths())
def test_feature_density_plots(value, n_threads, n_partitions):
    """Test for feature_density_plots function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    result, result2 = feature_density_plots(test_sdf_cleaned)
    numerical_cols = list(
        set(list_numerical_columns(test_sdf_cleaned))
        - set(list_numerical_categorical_columns(test_sdf_cleaned))
    )
    assert (
        list(result.keys()).sort() == numerical_cols.sort()
        and list(result.keys()).sort() == list_categorical_columns(test_sdf).sort()
    )


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("value", generate_data_paths())
def test_feature_analysis_pca(value, n_threads, n_partitions):
    """Test for feature_analysis_pca function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    num_cols = list(
        set(list_numerical_columns(test_sdf))
        - set(list_numerical_categorical_columns(test_sdf))
    )
    result = feature_analysis_pca(test_sdf, num_cols[0])
    assert isinstance(result, pyspark.sql.dataframe.DataFrame)


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("value", generate_data_paths())
def test_feature_importance(value, n_threads, n_partitions):
    """Test for feature_importance function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    num_cols = list_numerical_columns(test_sdf_cleaned)
    result = feature_importance(test_sdf_cleaned, target_var=num_cols[0])
    assert isinstance(result, hv.element.chart.Bars)


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("value", generate_data_paths())
def test_correlation_with_target(value, n_threads, n_partitions):
    """Test for correlation_with_target function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    num_cols = list_numerical_columns(test_sdf_cleaned)
    result = correlation_with_target(test_sdf_cleaned, num_cols[0])
    assert isinstance(result, hv.element.chart.Bars)


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("value", generate_data_paths())
def test_health_analysis(value, n_threads, n_partitions):
    """Test for health_analysis function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    result = health_analysis(
        test_sdf_cleaned,
        save_as=".html",
        save_path=os.path.join(os.getcwd(), "health_analysis_report.html"),
    )
    file = pathlib.Path(os.path.join(os.getcwd(), "health_analysis_report.html"))
    assert file.exists()


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("is_classification", [True, False])
@pytest.mark.parametrize("value", generate_data_paths())
def test_key_drivers(value, n_threads, n_partitions, is_classification):
    """Test for key_drivers function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    if is_classification:
        cat_col = list_categorical_columns(test_sdf_cleaned)
        if len(cat_col) > 0:
            result = EDAReportPyspark(test_sdf_cleaned, is_classification=True, y=cat_col[0])
            result.key_drivers(
                y=cat_col[0],
                save_as=".html",
                save_path=os.path.join(os.getcwd(), "key_drivers_report.html"),
            )
            file = pathlib.Path(os.path.join(os.getcwd(), "key_drivers_report.html"))
            assert file.exists()
    else:
        num_cols = list_numerical_columns(test_sdf_cleaned)
        if len(num_cols) > 0:
            result = EDAReportPyspark(test_sdf_cleaned, is_classification=False, y=num_cols[0])
            result.key_drivers(
                y=num_cols[0],
                save_as=".html",
                save_path=os.path.join(os.getcwd(), "key_drivers_report.html"),
            )
            file = pathlib.Path(os.path.join(os.getcwd(), "key_drivers_report.html"))
            assert file.exists()


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("is_classification", [True, False])
@pytest.mark.parametrize("value", generate_data_paths())
def test_get_report(value, n_threads, n_partitions, is_classification):
    """Test for get_report function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    if is_classification:
        cat_col = list_categorical_columns(test_sdf_cleaned)
        if len(cat_col) > 0:
            result = EDAReportPyspark(test_sdf_cleaned, is_classification=True, y=cat_col[0])
            result.get_report(
                y=cat_col[0],
                format=".html",
                save_path=os.path.join(os.getcwd(), "key_drivers_report.html"),
            )
            file = pathlib.Path(os.path.join(os.getcwd(), "key_drivers_report.html"))
            assert file.exists()
    else:
        num_cols = list(
            set(list_numerical_columns(test_sdf_cleaned))
            - set(list_numerical_categorical_columns(test_sdf_cleaned))
        )
        if len(num_cols) > 0:
            result = EDAReportPyspark(test_sdf_cleaned, is_classification=False, y=num_cols[0])
            result.get_report(
                y=num_cols[0],
                format=".html",
                save_path=os.path.join(os.getcwd(), "key_drivers_report.html"),
            )
            file = pathlib.Path(os.path.join(os.getcwd(), "key_drivers_report.html"))
            assert file.exists()


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("test_type", ["T2"], ids=id_func)
@pytest.mark.parametrize("return_plots", [False, True])
@pytest.mark.parametrize("top_n", [10, 15])
@pytest.mark.parametrize("value", generate_data_paths())
def test_get_bivariate(value, n_threads, n_partitions, test_type, return_plots, top_n):
    """Test for get_bivariate function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    test_sdf_cleaned = clean_columns(test_sdf)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    if return_plots:
        results = get_bivariate(test_sdf_cleaned, return_plots=True, top_n=top_n)
    else:
        results = get_bivariate(test_sdf_cleaned, return_plots=False, top_n=top_n)
    if test_type == "T2":
        if return_plots:
            assert isinstance(results, tuple)
        else:
            assert isinstance(results, dict) and results is not None


@pytest.mark.parametrize("n_partitions", ["n_partitions1"])
@pytest.mark.parametrize("n_threads", ["n_threads2"])
@pytest.mark.parametrize("value", generate_data_paths())
def test_data_health_recommendations(value, n_threads, n_partitions):
    """Test for feature_analysis_pca function."""
    threads_str = "local[{}]".format(int(n_threads[-1]))
    spark_session = (
        SparkSession.builder.master(threads_str)
        .appName("pyspark.eda_local_testing")
        .getOrCreate()
    )
    test_sdf = spark_session.read.csv(value, header=True, inferSchema=True)
    spark_session.conf.set("spark.sql.shuffle.partitions", int(n_partitions[-1]))
    result = data_health_recommendations(test_sdf)
    assert isinstance(result, pd.core.frame.DataFrame) and result.columns.tolist() == ['Recommendations', 'Reason For Recommendation']
