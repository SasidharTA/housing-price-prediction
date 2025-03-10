import holoviews as hv
import matplotlib
import os
import pandas as pd
import pathlib
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, floats, integers, sampled_from
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    make_classification,
    make_regression,
)

from ..Analyser import Analyser


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


@composite
def classification_data(draw):
    n_samples_val = draw(integers(min_value=100, max_value=1000))
    n_features_val = draw(integers(min_value=7, max_value=50))
    n_informative_val = draw(integers(min_value=3, max_value=n_features_val - 2))
    random_state_val = draw(integers(min_value=10, max_value=1000))
    hypercube_val = draw(sampled_from([True, False]))
    X, y = make_classification(
        n_samples=n_samples_val,
        n_features=n_features_val,
        n_informative=n_informative_val,
        random_state=random_state_val,
        hypercube=hypercube_val,
    )

    df = pd.DataFrame(
        data=X[0:, 0:],
        index=[i for i in range(X.shape[0])],
        columns=["Col_" + str(i + 1) for i in range(X.shape[1])],
    )
    df["DV"] = y
    return df


# Test for continuous target function
@settings(
    max_examples=5, deadline=None
)  # For testing in CI max_examples is set to 5, Can be increased upto 10 for developers
@given(regression_data())
def test_feature_scores_cont(test_df):
    analyser = Analyser(test_df, y="DV")

    # FIXED:explicitly pass 'mutual_information in scores list. 'defaul is only "feature_correlation"
    plot = analyser.get_feature_scores(
        scores=["feature_correlation", "mutual_information"]
    )
    # plot = analyser.get_feature_scores();

    # when no parameters to function, performs both feature correlation and mutual information
    assert "feature_correlation" in plot.keys() and (
        "mutual_information" in plot.keys()
    )

    # check the types of plots
    assert type(plot["feature_correlation"]) == hv.element.chart.Bars
    assert type(plot["mutual_information"]) == hv.element.chart.Bars

    # when quick is enabled, only feature correlation plot is expected
    plot = analyser.get_feature_scores(quick=True)
    assert list(plot.keys()) == ["feature_correlation"]

    # should return only the plots given in scores
    plot = analyser.get_feature_scores(scores=["mutual_information"])
    assert list(plot.keys()) == ["mutual_information"]


# Test for Categorical Target function
@settings(
    max_examples=5, deadline=None
)  # For testing in CI max_examples is set to 5, Can be increased upto 10 for developers
@given(classification_data())
def test_feature_scores_cat(test_df):

    # analyser raises error when the target is not a string
    analyser = Analyser(test_df, y="DV")

    # FIXED:explicitly pass 'mutual_information in scores list. 'defaul is only "feature_correlation"
    plot = analyser.get_feature_scores(
        scores=["feature_correlation", "mutual_information"]
    )
    # plot = analyser.get_feature_scores();

    # when no parameters to function, performs both feature correlation and mutual information
    assert ("feature_correlation" in plot.keys()) and (
        "mutual_information" in plot.keys()
    )

    # check the types of plots
    assert type(plot["feature_correlation"]) == hv.element.chart.Bars
    assert type(plot["mutual_information"]) == hv.element.chart.Bars

    # when quick is enabled, only feature correlation plot is expected
    plot = analyser.get_feature_scores(quick=True)
    assert list(plot.keys()) == ["feature_correlation"]

    # should return only the plots given in scores
    plot = analyser.get_feature_scores(scores=["mutual_information"])
    assert list(plot.keys()) == ["mutual_information"]


# @settings(max_examples=10, deadline=None)
# @given(regression_data())
def test_feature_importances_cont():
    X = fetch_california_housing()
    test_df = pd.DataFrame(X.data, columns=X.feature_names)
    test_df["MEDV"] = X.target
    analyser = Analyser(test_df, y="MEDV")
    plot = analyser.feature_importances()
    # when no parameters to function, performs both from_model and shap_values plots
    assert ("from_model" in plot.keys()) and ("shap_values" in plot.keys())

    # check the types of plots
    assert type(plot["from_model"]) == hv.element.chart.Bars

    # when quick is enabled, only feature importance from model plot is expected
    plot = analyser.feature_importances(quick=True)
    assert list(plot.keys()) == ["from_model"]


# @settings(max_examples=10, deadline=None)
# @given(classification_data())
def test_feature_importances_cat():
    X = load_breast_cancer()
    test_df = pd.DataFrame(X.data, columns=X.feature_names)
    test_df["cancer_flag"] = X.target
    # analyser raises error when the target is not a string
    analyser = Analyser(test_df, y=["cancer_flag"])

    plot = analyser.feature_importances()
    # when no parameters to function, performs both from_model and shap_values plots
    assert ("from_model" in plot.keys()) and ("shap_values" in plot.keys())

    # check the types of plots
    assert type(plot["from_model"]) == hv.element.chart.Bars
    assert type(plot["shap_values"] == matplotlib.figure.Figure)

    # when quick is enabled, only feature importance from model plot is expected
    plot = analyser.feature_importances(quick=True)
    assert list(plot.keys()) == ["from_model"]


@settings(
    max_examples=5, deadline=None
)  # For testing in CI max_examples is set to 5, Can be increased upto 10 for developers
@given(regression_data())
def test_pca_analysis_cont(test_df):
    analyser = Analyser(test_df, y="DV")
    plot = analyser.get_pca_analysis()

    assert (
        ("pca_projection" in plot.keys())
        and ("correlation_with_dimension_2 (Y)" in plot.keys())
        and ("correlation_with_dimension_1 (X)" in plot.keys())
    )

    # check the types of plots
    assert type(plot["pca_projection"]) == hv.core.spaces.DynamicMap
    assert type(plot["correlation_with_dimension_2 (Y)"]) == hv.element.chart.Bars
    assert type(plot["correlation_with_dimension_1 (X)"]) == hv.element.chart.Bars


@settings(
    max_examples=5, deadline=None
)  # For testing in CI max_examples is set to 5, Can be increased upto 10 for developers
@given(classification_data())
def test_pca_analysis_cat(test_df):

    # analyser raises error when the target is not a string
    analyser = Analyser(test_df, y="DV")

    plot = analyser.get_pca_analysis()

    assert (
        ("pca_projection" in plot.keys())
        and ("correlation_with_dimension_2 (Y)" in plot.keys())
        and ("correlation_with_dimension_1 (X)" in plot.keys())
    )

    # check the types of plots
    assert type(plot["pca_projection"]) == hv.core.spaces.DynamicMap
    assert type(plot["correlation_with_dimension_2 (Y)"]) == hv.element.chart.Bars
    assert type(plot["correlation_with_dimension_1 (X)"]) == hv.element.chart.Bars


# --------------------------------------------------------------------------------------
# ---------------------------Testing for key_drivers-------------------------------


# Test for continuous target function
@settings(
    max_examples=5, deadline=None
)  # For testing in CI max_examples is set to 5, Can be increased upto 10 for developers
@given(regression_data())
def test_key_drivers(test_df):
    analyser = Analyser(test_df, y="DV")
    # running the feature_analysis on test data
    output_df = analyser.key_drivers(
        save_as=".html", save_path=os.path.join(os.getcwd(), "key_drivers_report.html")
    )
    # expected output
    expected_keys = [
        "feature_scores",
        "feature_importances",
        "pca_analysis",
        "bivariate_plots",
    ]
    assert sorted(list(output_df["DV"].keys())) == sorted(expected_keys)
    # check whether file was saved
    file = pathlib.Path(os.path.join(os.getcwd(), "key_drivers_report.html"))
    assert file.exists()


# Test for Categorical Target function
@settings(
    max_examples=5, deadline=None
)  # For testing in CI max_examples is set to 5, Can be increased upto 10 for developers
@given(classification_data())
def test_key_drivers_cat(test_df):
    analyser = Analyser(test_df, y="DV")
    # running the feature_analysis on test data
    output_df = analyser.key_drivers(
        save_as=".html", save_path=os.path.join(os.getcwd(), "key_drivers_report.html")
    )
    # expected output
    expected_keys = [
        "feature_scores",
        "feature_importances",
        "pca_analysis",
        "bivariate_plots",
    ]
    assert sorted(list(output_df["DV"].keys())) == sorted(expected_keys)
    # check whether file was saved
    file = pathlib.Path(os.path.join(os.getcwd(), "key_drivers_report.html"))
    assert file.exists()
