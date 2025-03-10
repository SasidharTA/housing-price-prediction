import pytest
from sklearn.utils.estimator_checks import (
    check_estimator,
    parametrize_with_checks,
)

from tigerml.core.preprocessing.feature_engg.transformers import (
    SupervisedTransformer,
    UnsupervisedTransformer,
    WoeBinningTransformer,
)


@pytest.mark.parametrize("Transformer", [WoeBinningTransformer])
def test_all_transformers(Transformer):
    return check_estimator(Transformer)


@parametrize_with_checks([WoeBinningTransformer()])
def test_sklearn_binner(estimator, check):
    check(estimator)


@parametrize_with_checks([UnsupervisedTransformer(n=1)])
def test_sklearn_unsupervised(estimator, check):
    check(estimator)


@parametrize_with_checks([SupervisedTransformer()])
def test_sklearn_supervised(estimator, check):
    check(estimator)
