import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_warns_message,
)

from tigerml.core.preprocessing.feature_engg import WoeBinningTransformer

X = pd.DataFrame(
    {
        "X1": [501, 301, 800, -10, 30, 4334, 69, 5509, 1071],
        "X2": [1543, 3634, 8209, 12100, 4678, 0, -615, -550, 0],
        "X3": [-50, -30, -80, -100, -10, -43, -69, -55, -99],
    }
)

y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0])

df_onehot = np.array(
    [
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]
)

df_ordinal = np.array(
    [
        [1, 1, 0],
        [1, 1, 1],
        [1, 2, 0],
        [0, 2, 1],
        [2, 2, 0],
        [0, 0, 1],
        [2, 0, 0],
        [0, 0, 1],
        [2, 0, 0],
    ]
)


@pytest.mark.parametrize(
    "encode, expected", [("onehot", df_onehot), ("ordinal", df_ordinal)]
)
def test_fit_transform(encode, expected):
    est = WoeBinningTransformer(encode=encode)
    est.fit(X, y)
    assert_array_equal(expected, est.transform(X))


def test_invalid_n_features():
    est = WoeBinningTransformer().fit(X, y)
    bad_X = np.arange(25).reshape(5, -1)
    err_msg = "Inconsistent fit and transform"
    with pytest.raises(ValueError, match=err_msg):
        est.transform(bad_X)


def test_transform_1d_behavior():
    X = np.arange(4)
    est = WoeBinningTransformer()
    with pytest.raises(ValueError):
        est.fit(X, np.array([1, 1, 0, 0]))

    est = WoeBinningTransformer()
    est.fit(X.reshape(-1, 1), np.array([1, 1, 0, 0]))
    with pytest.raises(ValueError):
        est.transform(X)


@pytest.mark.parametrize("i", range(5, 9))
def test_numeric_stability(i):
    X_init = np.array([2.0, 4.0, 6.0, 8.0, 10.0]).reshape(-1, 1)
    y = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    Xt_expected = pd.DataFrame({"s_0_woebin": [1, 2, 3, 4, 0]})

    # Test up to discretizing nano units
    X = X_init / 10 ** i
    Xt = WoeBinningTransformer(encode="ordinal").fit_transform(X, y)
    assert_array_equal(Xt_expected, Xt)


def test_invalid_encode_option():
    est = WoeBinningTransformer(encode="invalid-encode")
    err_msg = (
        r"Valid options for 'encode' are "
        r"\('onehot', 'ordinal'\). "
        r"Got encode='invalid-encode' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)
