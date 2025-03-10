import os
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import composite, integers, sampled_from
from pytest import fixture
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from tigerml.core.preprocessing import prep_data
from tigerml.model_eval.base import *


# ---------------------------------------------------------------------------------
# ---------- composite strategy to generate Classification dataset ----------------
@composite
def classification_data(draw):
    """Creates dataset of sizes upto 100k using hypothesis library and makes it into classfication data using sklearn.make_classfication."""
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
# ----------- setting up ClassificationReport object for the tests ------------
@fixture
def sample_report():
    """Simplifies the task of fitting and scoring the models created using ClassificationReport and uses them in all the test cases later."""

    def _get_data(df, scoring=True, return_test_df=False):
        x_train, x_test, y_train, y_test = prep_data(df, dv_name="DV")
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr = model.fit(x_train, np.ravel(y_train))
        yhat_test = lr.predict_proba(x_test)[:, 1]
        yhat_train = lr.predict_proba(x_train)[:, 1]
        report = ClassificationReport(
            y_train, model, x_train, yhat_train, x_test, y_test, yhat_test, refit=True
        )
        return_val = [report, model, x_train, y_train]
        if return_test_df:
            return_val += [x_test, y_test]
        return return_val

    return _get_data


# ---------------------------------------------------------------------------------
# ------------------------------ fit testing -------------------------------------
@settings(
    max_examples=1,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow,),
)
@given(test_df=classification_data())
def test_fit(test_df, sample_report):
    """Checks if the models are fitting on the data or not."""
    report, model, x_train, y_train = sample_report(test_df, scoring=False)
    # check_is_fitted() raises a 'NotFittedError' error if the model is no fitted
    check_is_fitted(model)


# ---------------------------------------------------------------------------------
# ------------------------------ score testing -------------------------------------
@settings(
    max_examples=1,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow,),
)
@given(test_df=classification_data())
def test_score(test_df, sample_report):
    report, model, x_train, y_train = sample_report(test_df, scoring=True)
    assert report.x_test is not None
    assert report.y_test is not None


# ---------------------------------------------------------------------------------
# ------------------------------ get_report testing -------------------------------------
@settings(
    max_examples=1,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow,),
)
@given(test_df=classification_data())
def test_get_report(test_df, sample_report):
    """Checks if the report is generated."""
    report, model, x_train, y_train = sample_report(test_df)
    report.get_report(file_path="classification_report")
    assert os.path.exists("classification_report.html")
