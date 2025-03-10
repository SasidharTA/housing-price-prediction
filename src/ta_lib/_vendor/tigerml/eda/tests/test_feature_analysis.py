import os
import pathlib

import holoviews
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.pandas import columns, data_frames, range_indexes
from hypothesis.strategies import (
    booleans,
    composite,
    data,
    dates,
    floats,
    integers,
    sampled_from,
    text,
    tuples,
)
from sklearn.datasets import make_classification, make_regression

from tigerml.core.dataframe.helpers import tigerify
from tigerml.core.utils import get_bool_cols
from tigerml.core.utils.constants import SUMMARY_KEY_MAP

from ..Analyser import Analyser


class TestFeatureAnalysis:
    """Test FeatureAnalysis class."""

    def fixed_given(self):
        """Returns Given."""
        return given(
            test_df=data_frames(
                index=range_indexes(min_size=1),
                columns=columns(
                    ["float_col1", "string_col1", "date_col1", "bool_col1", "int_col1"],
                    dtype=float,
                ),
                rows=tuples(
                    floats(allow_nan=True, allow_infinity=True),
                    text(),
                    dates(),
                    booleans(),
                    integers(),
                ),
            )
        )(self)

    @settings(max_examples=6, deadline=None)
    @fixed_given
    def test_numeric_summary(self, test_df):
        """Returns tests numerical summeric."""
        # ---------------------- numeric summary testing ------------------------------------
        # runs the numeric summary with test data
        analyser = Analyser(test_df.copy(deep=True))
        num_out = analyser.numeric_summary()
        test_df = tigerify(test_df)
        cols = [
            col for col in test_df.numeric_columns if col not in get_bool_cols(test_df)
        ]
        if cols:
            # picking expected numeric columns
            expected_num_col_list = (
                test_df[cols].select_dtypes(include=np.number).columns.tolist()
            )
            assert (
                expected_num_col_list
                == num_out[SUMMARY_KEY_MAP.variable_names].tolist()
            )

            # verifying maximum values
            expected_max_values = test_df[expected_num_col_list].max()
            np.testing.assert_array_equal(
                expected_max_values.values, num_out[SUMMARY_KEY_MAP.max_value].values
            )

            # verifying min values
            expected_min_values = test_df[expected_num_col_list].min()
            np.testing.assert_array_equal(
                expected_min_values.values, num_out[SUMMARY_KEY_MAP.min_value].values
            )

            # verifying mean
            expected_mean_values = test_df[expected_num_col_list].mean()
            np.testing.assert_allclose(
                expected_mean_values.values, num_out[SUMMARY_KEY_MAP.mean_value].values
            )

            # verifying percentile , in this case for 75%
            expected_percentile_75 = test_df[expected_num_col_list].quantile(0.75)
            np.testing.assert_allclose(
                expected_percentile_75.values,
                num_out[SUMMARY_KEY_MAP.percentile_75].values,
            )
        else:
            assert num_out == "No Numerical columns in the data"

    @settings(max_examples=5, deadline=None)
    @fixed_given
    def test_non_numeric_summary(self, test_df):
        """Returns print statement if data is empty."""
        if test_df.empty:
            return print(
                "DataFrame is empty. non_numeric_summary() fails with empty dataframe"
            )
        analyser = Analyser(test_df.copy(deep=True))
        non_num_out = analyser.non_numeric_summary()
        test_df = tigerify(test_df)
        non_num_cols = [
            col for col in test_df.columns if col not in test_df.numeric_columns
        ]
        cat_cols = list(set(non_num_cols + get_bool_cols(test_df)))
        if cat_cols:
            # verify non numeric variables
            expected_non_num_cols = test_df.select_dtypes(
                exclude=np.number
            ).columns.tolist()
            if not test_df.empty:
                assert all(expected_non_num_cols) == all(
                    non_num_out[SUMMARY_KEY_MAP.variable_names].tolist()
                )
        else:
            assert non_num_out == "No categorical columns"

    @settings(max_examples=5, deadline=None)
    @fixed_given
    def test_variable_summary(self, test_df):
        """Returns test variable summary."""
        analyser = Analyser(test_df.copy(deep=True))
        var_sum_out = analyser.variable_summary()
        # verify unique values
        expected_unique_count = [test_df[col].nunique() for col in test_df]
        assert expected_unique_count == var_sum_out[SUMMARY_KEY_MAP.num_unique].tolist()
        # verify variable names
        expected_col_list = test_df.columns.to_list()
        assert expected_col_list == var_sum_out[SUMMARY_KEY_MAP.variable_names].tolist()

    @settings(max_examples=5, deadline=None)
    @fixed_given
    def test_non_numeric_plots(self, test_df):
        """Tests non numerical plots function."""
        analyser = Analyser(test_df.copy(deep=True))
        plot_dic = analyser.non_numeric_frequency_plot()
        if test_df.empty:
            plot_dic = "No columns in data"
        # verify non numeric variables
        else:
            for col, plots in plot_dic.items():
                if col in analyser.get_non_numeric_columns():
                    assert type(plots[0]) == holoviews.element.chart.Bars
                    assert type(plots[1]) == holoviews.element.tabular.Table

    @settings(max_examples=5, deadline=None)
    @fixed_given
    def test_feature_normality(self, test_df):
        """Tests non feature normality function."""
        analyser = Analyser(test_df.copy(deep=True))
        if not test_df.empty and len(test_df.float_col1) >= 3:
            plot = analyser.feature_normality()
            assert type(plot) == holoviews.element.chart.Bars
        elif test_df.empty or len(test_df.float_col1) < 3:
            # with pytest.raises(ValueError):
            plot = analyser.feature_normality()


@composite
def tuple_fsm(draw, col_type=None):
    """Returns tuple."""
    f = draw(
        floats(allow_nan=False, allow_infinity=False, max_value=5e14)
    )  # np.histogram([5e+15], bins=20) fails
    s = draw(sampled_from(["Red", "Blue", "Green"]))
    # m = draw(sampled_from(["text", 5.27, 5]))
    if col_type == "fff":
        return tuple([f, f, f])
    elif col_type == "sss":
        return tuple([s, s, s])
    return tuple([f, s, s])


@composite
def df_fsm(draw, col_type=None):
    df = draw(
        data_frames(
            index=range_indexes(min_size=1, max_size=10),
            columns=columns(["col1", "col2", "col3"], dtype=float,),
            rows=tuple_fsm(col_type),
        )
    )
    return df


@pytest.mark.parametrize("col_type", ["fff", "sss", "fss"])
@settings(max_examples=10, deadline=None)
@given(test_df_=data())
def test_numeric_distributions(test_df_, col_type):
    test_df = test_df_.draw(df_fsm(col_type))
    analyser = Analyser(test_df.copy())
    return_val = analyser.numeric_distributions()
    if col_type in ["fff", "fss"]:
        assert type(return_val) == holoviews.core.spaces.DynamicMap
    else:
        assert return_val == "No numeric columns in data"


@pytest.mark.parametrize("col_type", ["fff", "sss", "fss"])
@settings(max_examples=5, deadline=None)
@given(test_df_=data())
def test_feature_distributions(test_df_, col_type):
    test_df = test_df_.draw(df_fsm(col_type))
    if col_type == "fff":
        if test_df.iloc[0][0] >= 1e15:
            test_df.iloc[0][0] = test_df.iloc[0][0] + 1
            test_df.iloc[0][1] = test_df.iloc[0][1] + 1
            test_df.iloc[0][2] = test_df.iloc[0][2] + 1
        analyser = Analyser(test_df.copy())
        return_val = analyser.feature_distributions()
        assert (
            type(return_val["numeric_variables"]["col1"])
            == holoviews.core.layout.Layout
        )
        assert (
            type(return_val["numeric_variables"]["col2"])
            == holoviews.core.layout.Layout
        )
        assert (
            type(return_val["numeric_variables"]["col3"])
            == holoviews.core.layout.Layout
        )
        assert (
            return_val["non_numeric_variables"] == "No categorical variables in data."
        )
    analyser = Analyser(test_df.copy())
    return_val = analyser.feature_distributions()
    if col_type == "sss":
        assert (
            type(return_val["non_numeric_variables"]["col1"])
            == holoviews.core.layout.Layout
        )
        assert (
            type(return_val["non_numeric_variables"]["col2"])
            == holoviews.core.layout.Layout
        )
        assert (
            type(return_val["non_numeric_variables"]["col3"])
            == holoviews.core.layout.Layout
        )
        assert return_val["numeric_variables"] == "No numeric variables in data."
    if col_type == "fss":
        if test_df.iloc[0][0] >= 1e15:
            test_df.iloc[0][0] = test_df.iloc[0][0] + 1
        assert (
            type(return_val["numeric_variables"]["col1"])
            == holoviews.core.layout.Layout
        )
        assert (
            type(return_val["non_numeric_variables"]["col2"])
            == holoviews.core.layout.Layout
        )
        assert (
            type(return_val["non_numeric_variables"]["col3"])
            == holoviews.core.layout.Layout
        )


@composite
def tuple_num(draw, num_col=None):
    num = draw(floats(allow_nan=False, allow_infinity=False,))
    return tuple([num] * num_col)


@composite
def df_num(draw, num_col=None):
    col_name = []
    i = 0
    while i < num_col:
        col_name.append("col" + str(i))
        i += 1
    df = draw(
        data_frames(
            index=range_indexes(min_size=1, max_size=10),
            columns=columns(col_name, dtype=float,),
            rows=tuple_num(num_col),
        )
    )
    return df


@pytest.mark.parametrize("num_col", [0, 1, 2])
@settings(max_examples=5, deadline=None)
@given(test_df_=data())
def test_percentile_plots(test_df_, num_col):
    test_df = test_df_.draw(df_num(num_col))
    analyser = Analyser(test_df.copy())
    plot = analyser.percentile_plots()
    if test_df.empty:
        assert plot == "No numeric columns in data."
    elif num_col == 1:
        assert type(plot) == holoviews.core.layout.Layout
    else:
        assert type(plot) == holoviews.core.spaces.DynamicMap


# --------------------------------------------------------------------------------------
# ---------------------------Test for density_plots-------------------------------
@composite
def tuple_cn(draw, num_col=None, cat_col=None):
    num = draw(floats(allow_nan=False, allow_infinity=False,))
    cat = draw(sampled_from(["Red", "Blue", "Green", "Orange", "Yellow", "Violet"]))
    return tuple([num] * num_col + [cat] * cat_col)


@composite
def df_cn(draw, num_col=None, cat_col=None):
    col_name = []
    i = 0
    while i < num_col + cat_col:
        col_name.append("col" + str(i))
        i += 1
    df = draw(
        data_frames(
            index=range_indexes(min_size=3, max_size=10),
            columns=columns(col_name, dtype=float,),
            rows=tuple_cn(num_col, cat_col),
        )
    )
    return df


@pytest.mark.parametrize("num_col", [0, 1, 2])
@pytest.mark.parametrize("cat_col", [0, 1, 2])
@settings(max_examples=1, deadline=None)
@given(test_df_=data())
def test_density_plots(test_df_, num_col, cat_col):
    test_df = test_df_.draw(df_cn(num_col, cat_col))
    analyser = Analyser(test_df.copy())
    plot = analyser.density_plots()
    num_col = len(list(test_df.select_dtypes(include=np.number).columns))
    x = list(test_df.select_dtypes(exclude=np.number).columns)
    if x:
        cat_col = cat_col - len(
            [
                col
                for col in x
                if len([col for col in x if len(test_df[col].unique().tolist()) <= 2])
            ]
        )
    if test_df.empty:
        assert plot == "No columns in data"
    elif num_col == 1:
        assert type(plot) == holoviews.core.layout.Layout
    elif num_col == 2:
        assert type(plot) == holoviews.core.spaces.DynamicMap


# --------------------------------------------------------------------------------------
# ---------------------------Testing for feature_analysis-------------------------------
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
def df_all(draw,):
    df = draw(
        data_frames(
            index=range_indexes(min_size=1, max_size=10),
            columns=columns(
                ["col1", "col2", "col3", "col4", "col5", "col6"], dtype=float,
            ),
            rows=tuple_with_all(),
        )
    )
    return df


@settings(max_examples=6, deadline=None)
@given(test_df=df_all())
def feature_analysis(test_df):
    obj = Analyser(test_df.copy())
    # running the feature_analysis on test data
    output_df = obj.feature_analysis(
        save_as=".html",
        save_path=os.path.join(os.getcwd(), "feature_analysis_report.html"),
    )
    # expected output
    expected_keys = ["distributions", "feature_normality", "summary_stats"]
    assert sorted(list(output_df.keys())) == expected_keys
    assert sorted(list(output_df["summary_stats"].keys())) == [
        "non_numeric_variables",
        "numeric_variables",
    ]
    # check whether file was saved
    file = pathlib.Path(os.path.join(os.getcwd(), "feature_analysis_report.html"))
    assert file.exists()


# ---------------------------------------------------------------------------------
# ---------- composite strategy to generate Classification dataset ----------------
@composite
def classification_data(draw):
    """
    Creates dataset of sizes upto 100k using hypothesis library and makes it into classfication data using.

    Sklearn.make_classfication.
    """
    n_samples_val = draw(integers(min_value=1000, max_value=100000))
    # n_samples_val = draw(integers(min_value=100, max_value=1000))
    n_features_val = draw(integers(min_value=7, max_value=50))
    n_informative_val = draw(integers(min_value=3, max_value=n_features_val - 2))
    hypercube_val = draw(sampled_from([True, False]))
    random_state_val = draw(integers(min_value=10, max_value=1000))
    array_data = make_classification(
        n_samples=n_samples_val,
        n_features=n_features_val,
        n_informative=n_informative_val,
        hypercube=hypercube_val,
        random_state=random_state_val,
    )
    x_data = array_data[0]
    y_data = array_data[1]
    df = pd.DataFrame(
        data=x_data[0:, 0:],
        index=[i for i in range(x_data.shape[0])],
        columns=["Col_" + str(i + 1) for i in range(x_data.shape[1])],
    )
    df["DV"] = y_data
    return df


# ---------------------------------------------------------------------------------
# ---------- composite strategy to generate regression dataset ----------------
@composite
def regression_data(draw):
    n_samples_val = draw(integers(min_value=100, max_value=1000))
    n_features_val = draw(integers(min_value=7, max_value=50))
    n_informative_val = draw(integers(min_value=3, max_value=n_features_val - 2))
    random_state_val = draw(integers(min_value=10, max_value=1000))
    X, y = make_regression(
        n_samples=n_samples_val,
        n_features=n_features_val,
        n_informative=n_informative_val,
        random_state=random_state_val,
    )

    df = pd.DataFrame(
        data=X[0:, 0:],
        index=[i for i in range(X.shape[0])],
        columns=["Col_" + str(i + 1) for i in range(X.shape[1])],
    )
    df["DV"] = y
    return df


# ---------------------------------------------------------------------------------
# ------------------------------ testing target_distribution -------------------------------------

# Test for Categorical Target function
@settings(max_examples=10, deadline=None, suppress_health_check=HealthCheck.all())
@given(test_df=classification_data())
def test_target_distribution(test_df):
    an = Analyser(test_df, y="DV")
    output_plot = an.target_distribution()
    assert "holoviews" in str(type(output_plot))


# Test for continuous target function
@settings(max_examples=10, deadline=None)
@given(regression_data())
def test_target_distribution_cat(test_df):
    an = Analyser(test_df, y="DV")
    output_plot = an.target_distribution()
    assert "holoviews" in str(type(output_plot))
