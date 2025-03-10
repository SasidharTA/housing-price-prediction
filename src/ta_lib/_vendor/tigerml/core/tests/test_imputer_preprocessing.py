import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from tigerml.core.preprocessing.imputer import Imputer


@parametrize_with_checks([Imputer()])
def test_imputer(estimator, check):
    return check(estimator)
