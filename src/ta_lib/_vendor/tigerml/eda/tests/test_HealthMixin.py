import holoviews
import numpy as np
import os
import pandas as pd
import pathlib
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.pandas import columns, data_frames, range_indexes
from hypothesis.strategies import (
    booleans,
    composite,
    dates,
    floats,
    integers,
    sampled_from,
    text,
    tuples,
)

from tigerml.core.utils.constants import SUMMARY_KEY_MAP
from tigerml.eda import Analyser
from tigerml.eda.helpers import is_missing
from tigerml.eda.plotters.health_analysis.HealthMixin import HealthMixin

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------
# ---------------------- duplicate_columns testing ---------------------------------
@composite
def tuple_with_dups(draw):
    float_val = draw(
        floats(
            allow_nan=True,
            allow_infinity=True,
        )
    )
    text_val = draw(sampled_from(["Red", "Blue", "Green", np.nan, None]))
    mixed_val = draw(sampled_from(["text", 5.27, 5, np.nan, None]))
    return tuple([float_val, text_val, mixed_val] * 2)


@composite
def df_with_dups(draw):
    df = draw(
        data_frames(
            index=range_indexes(min_size=5),
            columns=columns(
                [
                    "float_col",
                    "string_col",
                    "mixed_col",
                    "float_col_2",
                    "string_col_2",
                    "mixed_col_2",
                ],
                dtype=float,
            ),
            rows=tuple_with_dups(),
        )
    )
    df.loc[len(df)] = [1, "Red", "text", 1, "Red", "text"]
    return df


@settings(max_examples=50, deadline=None)
@given(test_df=df_with_dups())
def test_duplicate_columns(test_df):
    obj = Analyser(test_df.copy())
    # running the duplicate_columns function on test data
    output_df = obj.duplicate_columns()
    # expected output
    inferred_dtypes = test_df.apply(pd.api.types.infer_dtype)
    if inferred_dtypes.str.startswith("mixed").any():
        # mixed col will be removed in DataProcessor
        exp_output = ["float_col", "string_col"]
    else:
        exp_output = ["float_col", "string_col", "mixed_col"]
    assert (sorted(output_df[SUMMARY_KEY_MAP.variable_names].tolist())) == (
        sorted(exp_output)
    )


# --------------------------------------------------------------------------------------
# ------------------- missing_value_summary testing -----------------------------
def fixed_given(func):
    return given(
        test_df=data_frames(
            # columns=columns(["float_col", "string_col", "mixed_col"], dtype=float,),
            columns=columns(
                ["float_col", "string_col"],
                dtype=float,
            ),
            rows=tuples(
                floats(allow_nan=True, allow_infinity=True),
                text(),
                # sampled_from(["text", 5.27, 5, np.nan, None]),
            ),
        )
    )(func)


@settings(max_examples=6, deadline=None)
@fixed_given
def test_missing_value_summary(test_df):
    obj = Analyser(test_df.copy())
    # running the missing_value_summary function on test data
    num_out = obj.missing_value_summary()
    # getting the expected missing_value_summary for  test data
    expected_df = test_df.isnull().sum().reset_index()
    expected_df = expected_df[expected_df[0] != 0]
    if expected_df[0].sum() == 0:
        assert num_out == "No Missing Values"
    else:
        num_out.sort_values("Variable Name", inplace=True)
        expected_df.sort_values("index", inplace=True)
        assert (
            (num_out[SUMMARY_KEY_MAP.variable_names].tolist())
            == (expected_df["index"].tolist())
        ) and (
            (
                num_out[
                    SUMMARY_KEY_MAP.num_missing + " (out of " + str(len(test_df)) + ")"
                ].tolist()
            )
            == (expected_df[0].tolist())
        )


# --------------------------------------------------------------------------------------
# ------------------------ missing_value_plot testing ------------------------------
@settings(max_examples=6, deadline=None)
@fixed_given
def test_missing_plot(test_df):
    obj = Analyser(test_df.copy())
    # running the missing_plot function on test data
    plot_out = obj.missing_plot()
    # getting the expected No. of columns with no missing values
    exp_missing_val = test_df.isnull().sum().reset_index()
    exp_missing_val = exp_missing_val[exp_missing_val[0] == 0].shape[0]
    # getting the expected No. of columns
    exp_total_val = test_df.shape[1]
    assert (
        (plot_out.data["0"][0] == exp_missing_val)
        and (plot_out.data["0"].sum() == exp_total_val)
        and (type(plot_out) == holoviews.element.chart.Bars)
    )


# --------------------------------------------------------------------------------------
# --------------- Testing of Duplicate Observations pie in data_health -----------------
@composite
def tuple_with_id(draw):
    float_val = draw(
        floats(
            allow_nan=True,
            allow_infinity=True,
        )
    )
    text_val = draw(sampled_from(["Red", "Blue", "Green", np.nan, None]))
    mixed_val = draw(sampled_from(["text", 5.27, 5, np.nan, None]))
    id_val = "_".join([str(float_val), str(text_val), str(mixed_val)])
    return tuple([id_val, float_val, text_val, mixed_val])


def given_dup_rows(func):
    return given(
        test_df=data_frames(
            index=range_indexes(min_size=10),
            columns=columns(
                ["id_col", "float_col", "string_col", "mixed_col"],
                dtype=float,
            ),
            rows=tuple_with_id(),
        )
    )(func)


@pytest.mark.parametrize(
    "test_type",
    [
        "check_dtypes_sum",
        "check_missingval_sum",
        "check_duplicateval_sum",
        "check_duplicatecol_sum",
        "check_plot",
    ],
)
@settings(max_examples=6, deadline=None)
@given_dup_rows
def test_data_health(test_df, test_type):
    obj = Analyser(test_df.copy())
    # running the test_data_health function on test data
    output_df = obj.data_health()
    exp_output = 100  # sum should be 1
    if test_type == "check_dtypes_sum":
        assert sum(output_df.Bars[0].data["value"]) == pytest.approx(
            exp_output, 0.0000000001
        )
    elif test_type == "check_missingval_sum":
        assert sum(output_df.Bars[1].data["value"]) == pytest.approx(
            exp_output, 0.0000000001
        )
    elif test_type == "check_duplicateval_sum":
        assert sum(output_df.Bars[2].data["value"]) == pytest.approx(
            exp_output, 0.0000000001
        )
    elif test_type == "check_duplicatecol_sum":
        assert sum(output_df.Bars[3].data["value"]) == pytest.approx(
            exp_output, 0.0000000001
        )
    else:
        assert type(output_df) == holoviews.core.layout.Layout


# --------------------------------------------------------------------------------------
# ------------------------------ Testing of _plot_data_health-----------------------------
@composite
def tuple_with_all(draw):
    f = draw(sampled_from([1.1, 5.52, 3.12, 5.27, np.nan, None]))
    t = draw(sampled_from(["Red", "Blue", "Green", np.nan, None]))
    m = draw(sampled_from(["text", 5.27, 5, np.nan, None]))
    d = draw(dates())
    b = draw(booleans())
    i = draw(integers())
    return tuple([f, t, m, d, b, i])


@composite
def df_all(
    draw,
):
    df = draw(
        data_frames(
            index=range_indexes(min_size=1, max_size=10),
            columns=columns(
                ["col1", "col2", "col3", "col4", "col5", "col6"],
                dtype=float,
            ),
            rows=tuple_with_all(),
        )
    )
    return df


@settings(max_examples=6, deadline=None)
@given(test_df=df_all())
def test_plot_data_health(test_df):
    obj = Analyser(test_df.copy())
    df = obj._compute_data_health()
    # running the _data_health_plot function on test data
    output_df = obj._plot_data_health(df)
    assert type(output_df) == holoviews.core.layout.Layout


# --------------------------------------------------------------------------------------
# ------------------------------ Testing of _missing_values-----------------------------
@composite
def tuple_with_fts(draw):
    float_val = draw(sampled_from([1.1, 5.52, 3.12, 5.27, np.nan, None]))
    text_val = draw(sampled_from(["Red", "Blue", "Green", np.nan, None]))
    # mixed_val = draw(sampled_from(["text", 5.27, 5, np.nan, None]))
    # return tuple([float_val, text_val, mixed_val])
    return tuple([float_val, text_val])


@composite
def df_ftm(
    draw,
):
    df = draw(
        data_frames(
            index=range_indexes(min_size=1, max_size=10),
            # columns=columns(["col1", "col2", "col3"], dtype=float,),
            columns=columns(
                ["col1", "col2"],
                dtype=float,
            ),
            rows=tuple_with_fts(),
        )
    )
    return df


@settings(max_examples=6, deadline=None)
@given(test_df=df_ftm())
def test_missing_values(test_df):
    obj = Analyser(test_df.copy())
    # running the _missing_values on test data
    output_df = obj._missing_values()
    # expected output
    data = {
        "Variable Name": test_df.columns.to_list(),
        "No of Missing": [
            test_df.loc[:, col].isnull().sum() for col in test_df.columns
        ],
    }
    exp_output = pd.DataFrame(data)
    exp_output["Per of Missing"] = (
        exp_output["No of Missing"] / float(test_df.shape[0]) * 100
    )
    np.testing.assert_array_equal(
        exp_output["Per of Missing"].values, output_df["Per of Missing"].values
    )
    np.testing.assert_array_equal(
        exp_output["Variable Name"].values, output_df["Variable Name"].values
    )
    np.testing.assert_array_equal(
        exp_output["No of Missing"].values, output_df["No of Missing"].values
    )


# --------------------------------------------------------------------------------------
# ---------------------------Testing for Outlier Analysis-------------------------------


def test_outlier_analysis():
    NUM_OF_MEAN_OUTLIERS = 50
    NUM_OF_MEDIAN_OUTLIERS = 200
    NUM_OF_INFS = 50
    NUM_OF_FEATURES = 5

    def get_outliers_data(samples, num_of_features):
        random_data = pd.DataFrame(
            np.random.normal(0, 3, (1000 - samples, num_of_features))
        )
        col_names = ["col" + str(idx) for idx in range(1, num_of_features + 1)]
        random_data.columns = col_names
        random_outliers1 = pd.DataFrame()
        for col_name in col_names:
            outliers = np.random.randint(90, 100, samples)
            random_outliers1[col_name] = outliers
        testing_data = pd.concat([random_data, random_outliers1, -random_outliers1])
        return testing_data

    def get_inf_data(samples, num_of_features):
        random_data = pd.DataFrame(
            np.random.normal(0, 3, (1000 - samples, num_of_features))
        )
        col_names = ["col" + str(idx) for idx in range(1, num_of_features + 1)]
        random_data.columns = col_names
        inf_df = pd.DataFrame()
        for col_name in col_names:
            inf_df[col_name] = [np.inf if i < 10 else -np.inf for i in range(samples)]
        testing_data = pd.concat([random_data, inf_df])
        return testing_data

    df = get_outliers_data(NUM_OF_MEAN_OUTLIERS, NUM_OF_FEATURES)
    df = df.select_dtypes(include=np.number)
    obj = Analyser(df)
    out_df = obj.get_outliers_df()
    # testing if the output given is of the correct form
    assert type(out_df) == pd.core.frame.DataFrame
    # testing if the the features in the expected output are same as the columns in the test dataframe.
    assert (out_df.index == df.columns).sum() == df.shape[1]

    # testing if the values calculated are correct.

    # for mean method
    exp_outliers_mean = [NUM_OF_MEAN_OUTLIERS * 2] * NUM_OF_FEATURES
    # out_df.columns = out_df.columns.droplevel()
    out_df_mean = out_df["< (mean-3*std)"] + out_df["> (mean+3*std)"]
    assert (exp_outliers_mean == out_df_mean).sum() == NUM_OF_FEATURES

    # for median method
    df_median = get_outliers_data(NUM_OF_MEDIAN_OUTLIERS, NUM_OF_FEATURES)
    out_df_med = Analyser(df_median).get_outliers_df()
    # out_df_med.columns = out_df_med.columns.droplevel()
    exp_outliers_median = [NUM_OF_MEDIAN_OUTLIERS * 2] * NUM_OF_FEATURES
    out_df_median = (
        out_df_med["< (1stQ - 1.5 * IQR)"] + out_df_med["> (3rdQ + 1.5 * IQR)"]
    )
    assert (exp_outliers_median == out_df_median).sum() == NUM_OF_FEATURES

    # for infinity values
    df_inf = get_inf_data(NUM_OF_INFS, NUM_OF_FEATURES)
    out_inf = Analyser(df_inf).get_outliers_df()
    # out_inf.columns = out_inf.columns.droplevel()
    exp_outliers_inf = [NUM_OF_INFS] * NUM_OF_FEATURES
    out_df_inf = out_inf["-inf"] + out_inf["+inf"]
    assert (exp_outliers_inf == out_df_inf).sum() == NUM_OF_FEATURES


# --------------------------------------------------------------------------------------
# ---------------------------Testing for health_analysis-------------------------------
@composite
def df_all(
    draw,
):
    df = draw(
        data_frames(
            index=range_indexes(min_size=1, max_size=10),
            columns=columns(
                ["col1", "col2", "col3", "col4", "col5", "col6"],
                dtype=float,
            ),
            rows=tuple_with_all(),
        )
    )
    return df


@settings(max_examples=6, deadline=None)
@given(test_df=df_all())
def test_health_analysis(test_df):
    obj = Analyser(test_df.copy())
    # running the health_analysis on test data
    output_df = obj.health_analysis(
        save_as=".html",
        save_path=os.path.join(os.getcwd(), "health_analysis_report.html"),
    )
    # expected output
    expected_keys = [
        "duplicate_columns",
        "health_plot",
        "missing_plot",
        "missing_value_summary",
        "outliers_in_features",
    ]
    assert sorted(list(output_df.keys())) == expected_keys
    # check whether file was saved
    file = pathlib.Path(os.path.join(os.getcwd(), "health_analysis_report.html"))
    assert file.exists()


# --------------------------------------------------------------------------------------
# ---------------------------Test for data_health_summary-------------------------------
@composite
def tuple_all_dups(draw):
    float_val = draw(sampled_from([1.1, 5.52, 3.12, 5.27, np.nan, None]))
    text_val = draw(sampled_from(["Red", "Blue", "Green", np.nan, None]))
    text_val2 = draw(sampled_from(["a", "b", "c", np.nan, None]))
    # mixed_val = draw(sampled_from(["text", 5.27, 5, np.nan, None]))
    date_val = draw(dates())
    bool_val = draw(booleans())
    int_val = draw(integers())
    # return tuple([float_val, text_val, mixed_val, date_val, bool_val, int_val] * 2)
    return tuple([float_val, text_val, text_val2, date_val, bool_val, int_val] * 2)


@composite
def df_all_dups(
    draw,
):
    col_names = ["col" + str(i) for i in range(1, 13)]
    df = draw(
        data_frames(
            index=range_indexes(min_size=1, max_size=10),
            columns=columns(
                col_names,
                dtype=float,
            ),
            rows=tuple_all_dups(),
        )
    )
    df[["col4", "col10"]] = df[["col4", "col10"]].apply(
        lambda x: pd.to_datetime(x, errors="coerce")
    )
    df.loc[len(df)] = [1.1, "Red", "x", pd.to_datetime("2015-1-1"), False, 1] * 2
    df.loc[len(df)] = [1.2, "Green", "y", pd.to_datetime("2016-1-1"), True, 2] * 2
    df.loc[len(df)] = [1.3, "Blue", "z", pd.to_datetime("2017-1-1"), False, 3] * 2
    return df


@settings(max_examples=50, deadline=None, suppress_health_check=HealthCheck.all())
@given(test_df=df_all_dups())
def test_compute_data_health(test_df):
    obj = Analyser(test_df.copy())
    print(test_df.dtypes.astype(str))
    # running data_health_summary on test data
    output_df = obj._compute_data_health()
    # expected output
    expected_output = {}
    expected_output["type"] = [
        "Datatypes",
        "Datatypes",
        "Datatypes",
        "Missing Values",
        "Missing Values",
        "Duplicate Values",
        "Duplicate Values",
        "Duplicate Columns",
        "Duplicate Columns",
    ]
    expected_output["labels"] = [
        "Date",
        "Numeric",
        "Others",
        "Available",
        "Missing",
        "Unique",
        "Duplicate",
        "Unique",
        "Duplicate",
    ]
    expected_output["values"] = [
        1 / 6 * 100,
        1 / 3 * 100,
        1 / 2 * 100,
        (1 - ((is_missing(test_df, obj.NA_VALUES).sum().sum()) / test_df.size)) * 100,
        ((is_missing(test_df, obj.NA_VALUES).sum().sum()) / test_df.size) * 100,
        (1 - ((test_df.duplicated().sum()) / len(test_df))) * 100,
        test_df.duplicated().sum() / len(test_df) * 100,
        1 / 2 * 100,
        1 / 2 * 100,
    ]
    expected_output_df = pd.DataFrame(expected_output)
    expected_output_df = expected_output_df.set_index(["type", "labels"])
    pd.testing.assert_frame_equal(output_df, expected_output_df)
