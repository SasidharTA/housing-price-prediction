import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import composite, integers, sampled_from
from sklearn.datasets import make_classification

from tigerml.core.dataframe.helpers import detigerify, tigerify


# ---------------------------------------------------------------------------------
# ---------- composite strategy to generate Classification dataset ----------------
@composite
def classification_data(draw):
    """Classification data.

    Creates dataset of sizes upto 100k using hypothesis library
    and makes it into classfication data using
    sklearn.make_classfication.
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
# ---------------------- detigerify testing ---------------------------------
@settings(max_examples=10, deadline=None, suppress_health_check=HealthCheck.all())
@given(test_df=classification_data())
def test_detigerify(test_df):

    df = tigerify(test_df)
    assert df.__module__.startswith("tigerml.core.dataframe")
    data1 = detigerify(df)
    data2 = detigerify(test_df)

    assert not data1.__module__.startswith("tigerml.core.dataframe")
    assert not data2.__module__.startswith("tigerml.core.dataframe")
