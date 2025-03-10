import os
import pandas as pd
import pytest
from bokeh.plotting.figure import Figure
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import composite, integers, sampled_from
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.validation import check_is_fitted

from tigerml.core.preprocessing import prep_data

from ..comparison import *


# ---------------------------------------------------------------------------------
# ---------- composite strategy to generate Regression dataset ----------------
@composite
def regression_data(draw):
    """Creates dataset of sizes upto 100k using hypothesis library and makes it into regression data usingsklearn.make_regression."""
    n_samples_val = draw(integers(min_value=100, max_value=100000))
    n_features_val = draw(integers(min_value=7, max_value=50))
    n_informative_val = draw(integers(min_value=3, max_value=n_features_val - 2))
    random_state_val = draw(integers(min_value=100, max_value=1000))
    X, y = make_regression(
        n_samples=n_samples_val,
        n_features=n_features_val,
        n_informative=n_informative_val,
        random_state=random_state_val,
    )

    df = df = pd.DataFrame(
        data=X[0:, 0:],
        index=[i for i in range(X.shape[0])],
        columns=["Col_" + str(i + 1) for i in range(X.shape[1])],
    )
    df["DV"] = y
    return df


# ---------------------------------------------------------------------------------
# ----------- setting up RegressionComparison object for the tests ------------
@pytest.fixture
def sample_report():
    """Simplifies the task of fitting and scoring the models created using RegressionComparison and uses them inall the test cases later."""

    def _get_data(df, scoring=True, return_test_df=False):
        x_train, x_test, y_train, y_test = prep_data(df, dv_name="DV")
        model1 = LinearRegression()
        model2 = Ridge()
        models = [model1, model2]
        report = RegressionComparison(y=y_train, models=models, x=x_train, refit=True)
        # report.fit(x_train, y_train)
        # if scoring:
        #     report.score(x_test, y_test)
        return_val = [report, models, x_train, y_train]
        if return_test_df:
            return_val += [x_test, y_test]
        return return_val

    return _get_data


# ---------------------------------------------------------------------------------
# ------------------------------ fit testing -------------------------------------
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow,),
)
@given(test_df=regression_data())
def test_fit(test_df, sample_report):
    """Checks if the models are fitting on the data or not."""
    report, models, x_train, y_train = sample_report(test_df, scoring=False)
    # check_is_fitted() raises a 'NotFittedError' error if the model is no fitted
    for model in models:
        check_is_fitted(model)


# ---------------------------------------------------------------------------------
# ------------------------------ feature_importances testing -------------------------------------
@settings(
    max_examples=1,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow,),
)
@given(test_df=regression_data())
def test_feature_importances(test_df, sample_report):
    report, model, x_train, y_train = sample_report(test_df)
    plot = report.feature_importances()
    if report.x is not None:
        assert type(plot) == dict
        assert plot is not None
    else:
        assert type(plot) == str


# ---------------------------------------------------------------------------------
# ------------------------------ get_report testing -------------------------------------
@settings(
    max_examples=1,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow,),
)
@given(test_df=regression_data())
def test_get_report(test_df, sample_report):
    """Checks if the report is generated."""
    report, model, x_train, y_train = sample_report(test_df)
    report.get_report(file_path="regression_comparison_report")
    """
    file_path = (
        f'{"regression"}'
        f'_comparison_report_at_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        f'{".html"}'
    )"""
    assert os.path.exists("regression_comparison_report.html")
