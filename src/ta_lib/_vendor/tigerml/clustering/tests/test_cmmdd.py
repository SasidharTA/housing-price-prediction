"""Module contains all the relevant test-casesunit-tests for the ClusterMMDD from clustering."""


import numpy as np
import pytest

from tigerml.clustering.cmmdd import ClusterMMDD
from tigerml.clustering.tests.estimator_checks import (
    check_estimator,
    parametrize_with_checks,
)


@parametrize_with_checks([ClusterMMDD()])
def test_cluster_mmdd(estimator, check):
    estimator.n_clustors = 3
    return check(estimator)
