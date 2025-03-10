import holoviews as hv
import numpy as np
import os
import pandas as pd
import panel
import pathlib
import pytest
import seaborn as sns
from hypothesis import given, settings
from hypothesis.extra.pandas import columns, data_frames, range_indexes
from hypothesis.strategies import (
    booleans,
    characters,
    composite,
    dates,
    floats,
    integers,
    sampled_from,
    text,
    tuples,
)

from ..Analyser import Analyser


class TestFeatureInteractions:
    """Test Feature interaction class."""

    def fixed_given(self):
        """Returns given."""
        return given(
            test_df=data_frames(
                columns=columns(
                    [
                        "float_col1",
                        "string_col1",
                        "date_col1",
                        "bool_col1",
                        "int_col1",
                        "cat_col1",
                        "bool_col2",
                    ],
                    dtype=float,
                ),
                rows=tuples(
                    floats(allow_nan=True, allow_infinity=True),
                    text(),
                    dates(),
                    booleans(),
                    integers(),
                    characters(),
                    booleans(),
                ),
            )
        )(self)

    @settings(max_examples=10, deadline=None)
    @fixed_given
    def test_correlation_heatmap(self, test_df):
        """Tests Correlations heatmap function."""
        analyser = Analyser(test_df.copy(deep=True))
        plot_out = analyser.correlation_heatmap()
        if not test_df.empty:
            assert type(plot_out) == hv.element.raster.HeatMap

    @settings(max_examples=10, deadline=None)
    @fixed_given
    def test_correlation_table(self, test_df):
        """Tests correlation_table function."""
        analyser = Analyser(test_df.copy(deep=True))
        plot_out = analyser.correlation_table()
        if not test_df.empty:
            assert type(plot_out) == pd.core.frame.DataFrame

    @settings(max_examples=10, deadline=None)
    @fixed_given
    def test_covariance_heatmap(self, test_df):
        """Tests covariance_heatmap function."""
        analyser = Analyser(test_df.copy(deep=True))
        plot_out = analyser.covariance_heatmap()
        if not test_df.empty:
            assert type(plot_out) == hv.element.raster.HeatMap

    @pytest.mark.parametrize(
        "x_vars_val, y_vars_val, return_dict_val, plot_keys, plot_types",
        [
            (
                ["sex"],
                ["smoker", "smoker_bool"],
                False,
                None,
                hv.core.spaces.DynamicMap,
            ),
            (
                ["sex"],
                ["smoker", "smoker_bool"],
                True,
                ["smoker", "smoker_bool"],
                [hv.element.chart.Bars, hv.element.chart.Bars],
            ),
            (["total_bill"], ["tip", "sex"], False, None, panel.layout.Column),
            (
                ["total_bill"],
                ["tip", "sex"],
                True,
                ["tip", "sex"],
                [hv.element.chart.Scatter, hv.element.stats.Violin],
            ),
        ],
    )
    def test_joint_plots(
        self, x_vars_val, y_vars_val, return_dict_val, plot_keys, plot_types
    ):
        """Tests joint plots function."""
        tips_df = sns.load_dataset("tips")
        tips_df["smoker_bool"] = np.where(tips_df["smoker"] == "Yes", True, False)
        analyser = Analyser(tips_df.copy(deep=True))
        plot_out = analyser.bivariate_plots(
            x_vars=x_vars_val, y_vars=y_vars_val, return_dict=return_dict_val
        )
        if plot_keys:
            for index in range(0, len(plot_keys)):
                assert type(plot_out[plot_keys[index]]) == plot_types[index]
        else:
            assert type(plot_out) == plot_types

    @pytest.mark.parametrize(
        "x_vars_val, y_vars_val, return_dict_val, plot_keys, plot_types",
        [
            (
                ["method"],
                ["number", "orbital_period", "mass", "distance", "year"],
                False,
                None,
                hv.core.spaces.DynamicMap,
            ),
            (
                ["method"],
                ["number", "orbital_period", "mass", "distance", "year"],
                True,
                ["number", "orbital_period", "mass", "distance", "year"],
                [hv.element.stats.Violin] * 5,
            ),
        ],
    )
    def test_joint_plots_cont(
        self, x_vars_val, y_vars_val, return_dict_val, plot_keys, plot_types
    ):
        """Tests joint plots function."""
        planets_df = sns.load_dataset("planets")
        analyser = Analyser(planets_df.copy(deep=True))
        plot_out = analyser.bivariate_plots(
            x_vars=x_vars_val, y_vars=y_vars_val, return_dict=return_dict_val
        )
        if plot_keys:
            for index in range(0, len(plot_keys)):
                assert type(plot_out[plot_keys[index]]) == plot_types[index]
        else:
            assert type(plot_out) == plot_types

    @settings(
        max_examples=2, deadline=None
    )  # For testing in CI max_examples is set to 2, Can be increased upto 10 for developers
    @fixed_given
    def test_joint_plots_hypothesise(self, test_df):
        """Tests jointl plots hypothesis function."""
        test_df.loc[len(test_df)] = [
            1.1,
            "red",
            pd.Timestamp("2000-01-01"),
            True,
            1,
            "a",
            True,
        ]
        test_df.loc[len(test_df)] = [
            1.2,
            "green",
            pd.Timestamp("2000-01-02"),
            False,
            2,
            "b",
            False,
        ]
        test_df.loc[len(test_df)] = [
            1.3,
            "yellow",
            pd.Timestamp("2000-01-03"),
            True,
            3,
            "c",
            True,
        ]
        test_df["bool_col1"] = test_df["bool_col1"].apply(
            lambda x: 1.0 if x is True else 0.0
        )
        analyser = Analyser(test_df.copy())
        plot_out = analyser.bivariate_plots(x_vars=["float_col1"], y_vars=["int_col1"])
        assert type(plot_out) == hv.element.chart.Scatter

        plot_out = analyser.bivariate_plots(
            x_vars=["float_col1", "int_col1", "bool_col1"], y_vars=["cat_col1"]
        )
        assert type(plot_out) == panel.layout.Column
        plot_out = analyser.bivariate_plots(
            x_vars=["float_col1", "int_col1", "bool_col1"],
            y_vars=["cat_col1"],
            return_dict=True,
        )
        assert type(plot_out["float_col1"]) == hv.element.stats.Violin
        assert type(plot_out["int_col1"]) == hv.element.stats.Violin
        assert type(plot_out["bool_col1"]) == hv.element.chart.Bars


# --------------------------------------------------------------------------------------
# ---------------------------Testing for feature_interactions---------------------------
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
def feature_interactions(test_df):
    obj = Analyser(test_df.copy())
    # running the feature_analysis on test data
    output_df = obj.feature_interactions(
        save_as=".html",
        save_path=os.path.join(os.getcwd(), "feature_interactions_report.html"),
    )
    # expected output
    expected_keys = ["correlation_table", "correlation_heatmap", "covariance_heatmap"]
    if len(test_df.columns) > 7:
        expected_keys.append("bivariate_plots (Top 50 Correlations)")
    else:
        expected_keys.append("bivariate_plots")
    assert sorted(list(output_df.keys())) == sorted(expected_keys)
    # check whether file was saved
    file = pathlib.Path(os.path.join(os.getcwd(), "feature_interactions_report.html"))
    assert file.exists()
