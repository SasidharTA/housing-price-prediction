"""Module contains all the relevant unit-test cases for the feature selection by statistic module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Ridge
from sklearn.utils.estimator_checks import (
    check_estimator,
    parametrize_with_checks,
)
from tigerml.core.preprocessing.feature_selection import (
    FeatureSelector,
    FeatureSelectorStatistic,
)


@parametrize_with_checks([FeatureSelector(Ridge(alpha=0.5))])
def test_feature_selector_model(estimator, check):
    return check(estimator)


@parametrize_with_checks([FeatureSelectorStatistic()])
def test_feature_statistic(estimator, check):
    return check(estimator)
