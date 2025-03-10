import os
import pathlib

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import composite, floats, integers, sampled_from
from sklearn.datasets import make_classification, make_regression

from tigerml.eda.Analyser import Analyser


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


# # ---------------------------------------------------------------------------------
# # ------------------------------ testing get_report -------------------------------------

# Test for Categorical Target function
@pytest.mark.parametrize("test_type", ["html", "excel"])
@settings(max_examples=1, deadline=None)
@given(test_df=classification_data())
def test_get_report(test_df, test_type):
    """Checks if the models are fitting on the data or not."""
    an = Analyser(test_df, y="DV")
    if test_type == "html":
        an.get_report(
            save_path=os.path.join(os.getcwd(), "data_exploration_report.html")
        )
        # check whether file was saved
        file = pathlib.Path(os.path.join(os.getcwd(), "data_exploration_report.html"))
    else:
        an.get_report(
            save_path=os.getcwd(), format=".xlsx", name="data_exploration_report"
        )
        # check whether file was saved
        file = pathlib.Path(os.path.join(os.getcwd(), "data_exploration_report.xlsx"))
    assert file.exists()


# Test for continuous target function
@pytest.mark.parametrize("test_type", ["html", "excel"])
@settings(
    max_examples=1, deadline=None
)  # For testing in CI max_examples is set to 1, Can be increased upto 10 for developers
@given(test_df=regression_data())
def test_get_report_cat(test_df, test_type):
    """Checks if the models are fitting on the data or not."""
    an = Analyser(test_df, y="DV")
    if test_type == "html":
        an.get_report(
            save_path=os.path.join(os.getcwd(), "data_exploration_report.html")
        )
        # check whether file was saved
        file = pathlib.Path(os.path.join(os.getcwd(), "data_exploration_report.html"))
    else:
        an.get_report(
            save_path=os.getcwd(), format=".xlsx", name="data_exploration_report"
        )
        # check whether file was saved
        file = pathlib.Path(os.path.join(os.getcwd(), "data_exploration_report.xlsx"))
    assert file.exists()


# ---------------------------------------------------------------------------------
# ------------------------------ testing create_report -------------------------------------

# Test for Categorical Target function
@settings(max_examples=1, deadline=None)
@given(test_df=classification_data())
def test_create_report(test_df):
    an = Analyser(test_df, y="DV")
    an._create_report()
    expected_keys = [
        "data_preview",
        "health_analysis",
        "feature_analysis",
        "feature_interactions",
        "key_drivers",
    ]
    assert list(an.report.keys()) == expected_keys


# Test for continuous target function
@settings(
    max_examples=1, deadline=None
)  # For testing in CI max_examples is set to 1, Can be increased upto 10 for developers
@given(test_df=regression_data())
def test_create_report_cat(test_df):
    an = Analyser(test_df, y="DV")
    an._create_report()
    expected_keys = [
        "data_preview",
        "health_analysis",
        "feature_analysis",
        "feature_interactions",
        "key_drivers",
    ]
    assert list(an.report.keys()) == expected_keys


# ---------------------------------------------------------------------------------
# ------------------------------ testing save_report -------------------------------------
@pytest.mark.parametrize("test_type", ["html", "excel"])
# Test for Categorical Target function
@settings(max_examples=1, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=classification_data()
)  # For testing in CI max_examples is set to 1, Can be increased upto 10 for developers
def test_save_report(test_df, test_type):
    an = Analyser(test_df, y="DV")
    an._create_report()
    if test_type == "html":
        an._save_report(
            save_path=os.path.join(os.getcwd(), "data_exploration_report.html")
        )
        # check whether file was saved
        file = pathlib.Path(os.path.join(os.getcwd(), "data_exploration_report.html"))
    else:
        an._save_report(
            save_path=os.getcwd(), format=".xlsx", name="data_exploration_report"
        )
        # check whether file was saved
        file = pathlib.Path(os.path.join(os.getcwd(), "data_exploration_report.xlsx"))
    assert file.exists()


@pytest.mark.parametrize("test_type", ["html", "excel"])
# Test for continuous target function
@settings(
    max_examples=1, deadline=None
)  # For testing in CI max_examples is set to 1, Can be increased upto 10 for developers
@given(test_df=regression_data())
def test_save_report_cat(test_df, test_type):
    an = Analyser(test_df, y="DV")
    an._create_report()
    if test_type == "html":
        an._save_report(
            save_path=os.path.join(os.getcwd(), "data_exploration_report.html")
        )
        # check whether file was saved
        file = pathlib.Path(os.path.join(os.getcwd(), "data_exploration_report.html"))
    else:
        an._save_report(
            save_path=os.getcwd(), format=".xlsx", name="data_exploration_report"
        )
        # check whether file was saved
        file = pathlib.Path(os.path.join(os.getcwd(), "data_exploration_report.xlsx"))
    assert file.exists()
