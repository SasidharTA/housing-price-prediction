import joblib
import numpy as np
import pickle
import re
import sys
import traceback
import types
import warnings
from copy import deepcopy
from functools import partial
from inspect import signature
from itertools import chain
from scipy import sparse
from scipy.stats import rankdata
from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    ClusterMixin,
    RegressorMixin,
    clone,
    is_classifier,
    is_outlier_detector,
    is_regressor,
)
from sklearn.datasets import (
    fetch_california_housing,
    load_iris,
    make_blobs,
    make_multilabel_classification,
    make_regression,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import (
    DataConversionWarning,
    NotFittedError,
    SkipTestWarning,
)
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.metrics.pairwise import (
    linear_kernel,
    pairwise_distances,
    rbf_kernel,
)
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.model_selection._validation import _safe_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import BaseRandomProjection
from sklearn.utils import IS_PYPY, deprecated, is_scalar_nan, shuffle
from sklearn.utils._testing import (
    SkipTest,
    _get_args,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
    assert_array_equal,
    assert_raise_message,
    assert_raises,
    assert_raises_regex,
    assert_warns_message,
    create_memmap_backed_data,
    ignore_warnings,
    set_random_state,
)
from sklearn.utils.validation import _num_samples, has_fit_parameter

CALIFORNIA = None
CROSS_DECOMPOSITION = ["PLSCanonical", "PLSRegression", "CCA", "PLSSVD"]


def _yield_checks(name, estimator):
    tags = estimator._get_tags()
    yield check_no_attributes_set_in_init
    yield check_estimators_dtypes
    yield check_fit_score_takes_y
    yield check_sample_weights_pandas_series
    yield check_sample_weights_not_an_array
    yield check_sample_weights_list
    yield check_sample_weights_shape
    yield check_sample_weights_invariance
    yield check_estimators_fit_returns_self
    yield partial(check_estimators_fit_returns_self, readonly_memmap=True)

    # Check that all estimator yield informative messages when
    # trained on empty datasets
    if not tags["no_validation"]:
        yield check_dtype_object
        yield check_estimators_empty_data_messages

    if name not in CROSS_DECOMPOSITION:
        # cross-decomposition's "transform" returns X and Y
        yield check_pipeline_consistency

    yield check_estimators_overwrite_params

    yield check_estimator_sparse_data

    # Test that estimators can be pickled, and once pickled
    # give the same answer as before.
    yield check_estimators_pickle


@ignore_warnings(category=FutureWarning)
def check_supervised_y_no_nan(name, estimator_orig):
    # Checks that the Estimator targets are not NaN.
    estimator = clone(estimator_orig)
    rng = np.random.RandomState(888)
    X = rng.randn(10, 5)
    y = np.full(10, np.inf)
    y = _enforce_estimator_tags_y(estimator, y)

    errmsg = (
        "Input contains NaN, infinity or a value too large for " "dtype('float64')."
    )
    try:
        estimator.fit(X, y)
    except ValueError as e:
        if str(e) != errmsg:
            raise ValueError(
                "Estimator {0} raised error as expected, but "
                "does not match expected error message".format(name)
            )
    else:
        raise ValueError(
            "Estimator {0} should have raised error on fitting "
            "array y with NaN value.".format(name)
        )


def _yield_transformer_checks(name, transformer):
    # All transformers should either deal with sparse data or raise an
    # exception with type TypeError and an intelligible error message
    if not transformer._get_tags()["no_validation"]:
        yield check_transformer_data_not_an_array
    # these don't actually fit the data, so don't raise errors
    yield check_transformer_general
    yield partial(check_transformer_general, readonly_memmap=True)
    if not transformer._get_tags()["stateless"]:
        yield check_transformers_unfitted
    # Dependent on external solvers and hence accessing the iter
    # param is non-trivial.
    external_solver = [
        "Isomap",
        "KernelPCA",
        "LocallyLinearEmbedding",
        "RandomizedLasso",
        "LogisticRegressionCV",
    ]
    if name not in external_solver:
        yield check_transformer_n_iter


def _yield_clustering_checks(name, clusterer):
    yield check_clusterer_compute_labels_predict
    if name not in ("WardAgglomeration", "FeatureAgglomeration"):
        # this is clustering on the features
        # let's not test that here.
        #         yield check_clustering
        #         yield partial(check_clustering, readonly_memmap=True)
        yield check_estimators_partial_fit_n_features
    yield check_non_transformer_estimators_n_iter


def _yield_all_checks(name, estimator):
    tags = estimator._get_tags()
    if "2darray" not in tags["X_types"]:
        warnings.warn(
            "Can't test estimator {} which requires input "
            " of type {}".format(name, tags["X_types"]),
            SkipTestWarning,
        )
        return
    if tags["_skip_test"]:
        warnings.warn(
            "Explicit SKIP via _skip_test tag for estimator " "{}.".format(name),
            SkipTestWarning,
        )
        return

    for check in _yield_checks(name, estimator):
        yield check
    if is_classifier(estimator):
        for check in _yield_classifier_checks(name, estimator):  # noqa: F821
            yield check
    if is_regressor(estimator):
        for check in _yield_regressor_checks(name, estimator):  # noqa: F821
            yield check
    if hasattr(estimator, "transform"):
        for check in _yield_transformer_checks(name, estimator):
            yield check
    if isinstance(estimator, ClusterMixin):
        for check in _yield_clustering_checks(name, estimator):
            yield check
    if is_outlier_detector(estimator):
        for check in _yield_outliers_checks(name, estimator):  # noqa: F821
            yield check
    yield check_fit2d_predict1d
    yield check_methods_subset_invariance
    yield check_fit2d_1sample
    yield check_fit2d_1feature
    yield check_fit1d
    yield check_get_params_invariance
    yield check_set_params
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters
    yield check_fit_idempotent
    if not tags["no_validation"]:
        yield check_n_features_in


def _set_check_estimator_ids(obj):
    """Create pytest ids for checks.

    When `obj` is an estimator, this returns the pprint version of the
    estimator (with `print_changed_only=True`). When `obj` is a function, the
    name of the function is returned with its keyworld arguments.

    A `_set_check_estimator_ids` is designed to be used as the `id` in
    `pytest.mark.parametrize` where `check_estimator(..., generate_only=True)`
    is yielding estimators and checks.

    Parameters
    ----------
    obj : estimator or function
        Items generated by `check_estimator`

    Returns
    -------
    id : string or None

    See Also
    --------
    check_estimator
    """
    if callable(obj):
        if not isinstance(obj, partial):
            return obj.__name__

        if not obj.keywords:
            return obj.func.__name__

        kwstring = ",".join(["{}={}".format(k, v) for k, v in obj.keywords.items()])
        return "{}({})".format(obj.func.__name__, kwstring)
    if hasattr(obj, "get_params"):
        with config_context(print_changed_only=True):
            return re.sub(r"\s", "", str(obj))


def _construct_instance(Estimator):
    """Construct Estimator instance if possible."""
    required_parameters = getattr(Estimator, "_required_parameters", [])
    if len(required_parameters):
        if required_parameters in (["estimator"], ["base_estimator"]):
            if issubclass(Estimator, RegressorMixin):
                estimator = Estimator(Ridge())
            else:
                estimator = Estimator(LinearDiscriminantAnalysis())
        else:
            raise SkipTest(
                "Can't instantiate estimator {} which requires "
                "parameters {}".format(Estimator.__name__, required_parameters)
            )
    else:
        estimator = Estimator()
    return estimator


# TODO: probably not needed anymore in 0.24 since _generate_class_checks should
# be removed too. Just put this in check_estimator()
def _generate_instance_checks(name, estimator):
    """Generate instance checks."""
    yield from (
        (estimator, partial(check, name))
        for check in _yield_all_checks(name, estimator)
    )


# TODO: remove this in 0.24
def _generate_class_checks(Estimator):
    """Generate class checks."""
    name = Estimator.__name__
    yield (Estimator, partial(check_parameters_default_constructible, name))  # noqa
    estimator = _construct_instance(Estimator)
    yield from _generate_instance_checks(name, estimator)


def _mark_xfail_checks(estimator, check, pytest):
    """Mark (estimator, check) pairs with xfail according to the _xfail_checks_ tag."""
    if isinstance(estimator, type):
        # try to construct estimator instance, if it is unable to then
        # return the estimator class, ignoring the tag
        # TODO: remove this if block in 0.24 since passing instances isn't
        # supported anymore
        try:
            estimator = _construct_instance(estimator)
        except Exception:
            return estimator, check

    xfail_checks = estimator._get_tags()["_xfail_checks"] or {}
    check_name = _set_check_estimator_ids(check)

    if check_name not in xfail_checks:
        # check isn't part of the xfail_checks tags, just return it
        return estimator, check
    else:
        # check is in the tag, mark it as xfail for pytest
        reason = xfail_checks[check_name]
        return pytest.param(estimator, check, marks=pytest.mark.xfail(reason=reason))


def parametrize_with_checks(estimators):
    """Pytest specific decorator for parametrizing estimator checks.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.
    This allows to use `pytest -k` to specify which tests to run::

        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Parameters
    ----------
    estimators : list of estimators objects or classes
        Estimators to generated checks for.

        .. deprecated:: 0.23
           Passing a class is deprecated from version 0.23, and won't be
           supported in 0.24. Pass an instance instead.

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import parametrize_with_checks
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeRegressor

    >>> @parametrize_with_checks([LogisticRegression(),
    ...                           DecisionTreeRegressor()])
    ... def test_sklearn_compatible_estimator(estimator, check):
    ...     check(estimator)

    """
    import pytest

    if any(isinstance(est, type) for est in estimators):
        # TODO: remove class support in 0.24 and update docstrings
        msg = (
            "Passing a class is deprecated since version 0.23 "
            "and won't be supported in 0.24."
            "Please pass an instance instead."
        )
        warnings.warn(msg, FutureWarning)

    checks_generator = chain.from_iterable(
        check_estimator(estimator, generate_only=True) for estimator in estimators
    )

    checks_with_marks = (
        _mark_xfail_checks(estimator, check, pytest)
        for estimator, check in checks_generator
    )

    return pytest.mark.parametrize(
        "estimator, check", checks_with_marks, ids=_set_check_estimator_ids
    )


def check_estimator(Estimator, generate_only=False):
    """Check if estimator adheres to scikit-learn conventions.

    This estimator will run an extensive test-suite for input validation,
    shapes, etc, making sure that the estimator complies with `scikit-learn`
    conventions as detailed in :ref:`rolling_your_own_estimator`.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.

    This test can be applied to classes or instances.
    Classes currently have some additional tests that related to construction,
    while passing instances allows the testing of multiple options. However,
    support for classes is deprecated since version 0.23 and will be removed
    in version 0.24 (class checks will still be run on the instances).

    Setting `generate_only=True` returns a generator that yields (estimator,
    check) tuples where the check can be called independently from each
    other, i.e. `check(estimator)`. This allows all checks to be run
    independently and report the checks that are failing.

    scikit-learn provides a pytest specific decorator,
    :func:`~sklearn.utils.parametrize_with_checks`, making it easier to test
    multiple estimators.

    Parameters
    ----------
    estimator : estimator object
        Estimator to check. Estimator is a class object or instance.

        .. deprecated:: 0.23
           Passing a class is deprecated from version 0.23, and won't be
           supported in 0.24. Pass an instance instead.

    generate_only : bool, optional (default=False)
        When `False`, checks are evaluated when `check_estimator` is called.
        When `True`, `check_estimator` returns a generator that yields
        (estimator, check) tuples. The check is run by calling
        `check(estimator)`.

        .. versionadded:: 0.22

    Returns
    -------
    checks_generator : generator
        Generator that yields (estimator, check) tuples. Returned when
        `generate_only=True`.
    """
    # TODO: remove class support in 0.24 and update docstrings
    if isinstance(Estimator, type):
        # got a class
        msg = (
            "Passing a class is deprecated since version 0.23 "
            "and won't be supported in 0.24."
            "Please pass an instance instead."
        )
        warnings.warn(msg, FutureWarning)

        checks_generator = _generate_class_checks(Estimator)
    else:
        # got an instance
        estimator = Estimator
        name = type(estimator).__name__
        checks_generator = _generate_instance_checks(name, estimator)

    if generate_only:
        return checks_generator

    for estimator, check in checks_generator:
        try:
            check(estimator)
        except SkipTest as exception:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(str(exception), SkipTestWarning)


def _housing_subset(n_samples=200):
    global CALIFORNIA
    if CALIFORNIA is None:
        X, y = fetch_california_housing(return_X_y=True)
        X, y = shuffle(X, y, random_state=0)
        X, y = X[:n_samples], y[:n_samples]
        X = StandardScaler().fit_transform(X)
        CALIFORNIA = X, y
    return CALIFORNIA


@deprecated(
    "set_checking_parameters is deprecated in version "
    "0.22 and will be removed in version 0.24."
)
def set_checking_parameters(estimator):
    _set_checking_parameters(estimator)


def _set_checking_parameters(estimator):
    # set parameters to speed up some estimators and
    # avoid deprecated behaviour
    params = estimator.get_params()
    name = estimator.__class__.__name__
    if "n_iter" in params and name != "TSNE":
        estimator.set_params(n_iter=5)
    if "max_iter" in params:
        if estimator.max_iter is not None:
            estimator.set_params(max_iter=min(5, estimator.max_iter))
        # LinearSVR, LinearSVC
        if estimator.__class__.__name__ in ["LinearSVR", "LinearSVC"]:
            estimator.set_params(max_iter=20)
        # NMF
        if estimator.__class__.__name__ == "NMF":
            estimator.set_params(max_iter=100)
        # MLP
        if estimator.__class__.__name__ in ["MLPClassifier", "MLPRegressor"]:
            estimator.set_params(max_iter=100)
    if "n_resampling" in params:
        # randomized lasso
        estimator.set_params(n_resampling=5)
    if "n_estimators" in params:
        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
    if "max_trials" in params:
        # RANSAC
        estimator.set_params(max_trials=10)
    if "n_init" in params:
        # K-Means
        estimator.set_params(n_init=2)

    if name == "TruncatedSVD":
        # TruncatedSVD doesn't run with n_components = n_features
        # This is ugly :-/
        estimator.n_components = 1

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = min(estimator.n_clusters, 2)

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    if name == "SelectFdr":
        # be tolerant of noisy datasets (not actually speed)
        estimator.set_params(alpha=0.5)

    if name == "TheilSenRegressor":
        estimator.max_subpopulation = 100

    if isinstance(estimator, BaseRandomProjection):
        # Due to the jl lemma and often very few samples, the number
        # of components of the random matrix projection will be probably
        # greater than the number of features.
        # So we impose a smaller number (avoid "auto" mode)
        estimator.set_params(n_components=2)

    if isinstance(estimator, SelectKBest):
        # SelectKBest has a default of k=10
        # which is more feature than we have in most case.
        estimator.set_params(k=1)

    if name in ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
        # The default min_samples_leaf (20) isn't appropriate for small
        # datasets (only very shallow trees are built) that the checks use.
        estimator.set_params(min_samples_leaf=5)

    # Speed-up by reducing the number of CV or splits for CV estimators
    loo_cv = ["RidgeCV"]
    if name not in loo_cv and hasattr(estimator, "cv"):
        estimator.set_params(cv=3)
    if hasattr(estimator, "n_splits"):
        estimator.set_params(n_splits=3)

    if name == "OneHotEncoder":
        estimator.set_params(handle_unknown="ignore")


class _NotAnArray:
    """An object that is convertible to an array.

    Parameters
    ----------
    data : array_like
        The data.
    """

    def __init__(self, data):
        self.data = np.asarray(data)

    def __array__(self, dtype=None):
        return self.data

    def __array_function__(self, func, types, args, kwargs):  # noqa: F811
        if func.__name__ == "may_share_memory":
            return True
        raise TypeError("Don't want to call array_function {}!".format(func.__name__))


@deprecated(
    "NotAnArray is deprecated in version " "0.22 and will be removed in version 0.24."
)
class NotAnArray(_NotAnArray):
    """Not and array class."""

    # TODO: remove in 0.24
    pass


def _is_pairwise(estimator):
    """Returns True if estimator has a _pairwise attribute set to True.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if _pairwise is set to True and False otherwise.
    """
    return bool(getattr(estimator, "_pairwise", False))


def _is_pairwise_metric(estimator):
    """Returns True if estimator accepts pairwise metric.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if _pairwise is set to True and False otherwise.
    """
    metric = getattr(estimator, "metric", None)

    return bool(metric == "precomputed")


@deprecated(
    "pairwise_estimator_convert_X is deprecated in version "
    "0.22 and will be removed in version 0.24."
)
def pairwise_estimator_convert_X(X, estimator, kernel=linear_kernel):
    return _pairwise_estimator_convert_X(X, estimator, kernel)


def _pairwise_estimator_convert_X(X, estimator, kernel=linear_kernel):

    if _is_pairwise_metric(estimator):
        return pairwise_distances(X, metric="euclidean")
    if _is_pairwise(estimator):
        return kernel(X, X)

    return X


def _generate_sparse_matrix(X_csr):
    """Generate sparse matrices with {32,64}bit indices of diverse format.

    Parameters
    ----------
    X_csr: CSR Matrix
        Input matrix in CSR format

    Returns
    -------
    out: iter(Matrices)
        In format['dok', 'lil', 'dia', 'bsr', 'csr', 'csc', 'coo',
        'coo_64', 'csc_64', 'csr_64']
    """

    assert X_csr.format == "csr"
    yield "csr", X_csr.copy()
    for sparse_format in ["dok", "lil", "dia", "bsr", "csc", "coo"]:
        yield sparse_format, X_csr.asformat(sparse_format)

    # Generate large indices matrix only if its supported by scipy
    X_coo = X_csr.asformat("coo")
    X_coo.row = X_coo.row.astype("int64")
    X_coo.col = X_coo.col.astype("int64")
    yield "coo_64", X_coo

    for sparse_format in ["csc", "csr"]:
        X = X_csr.asformat(sparse_format)
        X.indices = X.indices.astype("int64")
        X.indptr = X.indptr.astype("int64")
        yield sparse_format + "_64", X


def check_estimator_sparse_data(name, estimator_orig):
    rng = np.random.RandomState(0)
    X = rng.rand(40, 10)
    X[X < 0.8] = 0
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    X_csr = sparse.csr_matrix(X)
    y = (4 * rng.rand(40)).astype(int)
    # catch deprecation warnings
    with ignore_warnings(category=FutureWarning):
        estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    tags = estimator_orig._get_tags()
    for matrix_format, X in _generate_sparse_matrix(X_csr):
        # catch deprecation warnings
        with ignore_warnings(category=FutureWarning):
            estimator = clone(estimator_orig)
            if name in ["Scaler", "StandardScaler"]:
                estimator.set_params(with_mean=False)
        # fit and predict
        try:
            with ignore_warnings(category=FutureWarning):
                estimator.fit(X, y)
            if hasattr(estimator, "predict"):
                pred = estimator.predict(X)
                if tags["multioutput_only"]:
                    assert pred.shape == (X.shape[0], 1)
                else:
                    assert pred.shape == (X.shape[0],)
            if hasattr(estimator, "predict_proba"):
                probs = estimator.predict_proba(X)
                if tags["binary_only"]:
                    expected_probs_shape = (X.shape[0], 2)
                else:
                    expected_probs_shape = (X.shape[0], 4)
                assert probs.shape == expected_probs_shape
        except (TypeError, ValueError) as e:
            if "sparse" not in repr(e).lower():
                if "64" in matrix_format:
                    msg = (
                        "Estimator %s doesn't seem to support %s matrix, "
                        "and is not failing gracefully, e.g. by using "
                        "check_array(X, accept_large_sparse=False)"
                    )
                    raise AssertionError(msg % (name, matrix_format))
                else:
                    print(
                        "Estimator %s doesn't seem to fail gracefully on "
                        "sparse data: error message state explicitly that "
                        "sparse input is not supported if this is not"
                        " the case." % name
                    )
                    raise
        except Exception:
            print(
                "Estimator %s doesn't seem to fail gracefully on "
                "sparse data: it should raise a TypeError if sparse input "
                "is explicitly not supported." % name
            )
            raise


@ignore_warnings(category=FutureWarning)
def check_sample_weights_pandas_series(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type pandas.Series in the 'fit' function.
    estimator = clone(estimator_orig)
    if has_fit_parameter(estimator, "sample_weight"):
        try:
            import pandas as pd

            X = np.array(
                [
                    [1, 1],
                    [1, 2],
                    [1, 3],
                    [1, 4],
                    [2, 1],
                    [2, 2],
                    [2, 3],
                    [2, 4],
                    [3, 1],
                    [3, 2],
                    [3, 3],
                    [3, 4],
                ]
            )
            X = pd.DataFrame(_pairwise_estimator_convert_X(X, estimator_orig))
            y = pd.Series([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
            weights = pd.Series([1] * 12)
            if estimator._get_tags()["multioutput_only"]:
                y = pd.DataFrame(y)
            try:
                estimator.fit(X, y, sample_weight=weights)
            except ValueError:
                raise ValueError(
                    "Estimator {0} raises error if "
                    "'sample_weight' parameter is of "
                    "type pandas.Series".format(name)
                )
        except ImportError:
            raise SkipTest(
                "pandas is not installed: not testing for "
                "input of type pandas.Series to class weight."
            )


@ignore_warnings(category=(FutureWarning))
def check_sample_weights_not_an_array(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type _NotAnArray in the 'fit' function.
    estimator = clone(estimator_orig)
    if has_fit_parameter(estimator, "sample_weight"):
        X = np.array(
            [
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 1],
                [2, 2],
                [2, 3],
                [2, 4],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ]
        )
        X = _NotAnArray(pairwise_estimator_convert_X(X, estimator_orig))
        y = _NotAnArray([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
        weights = _NotAnArray([1] * 12)
        if estimator._get_tags()["multioutput_only"]:
            y = _NotAnArray(y.data.reshape(-1, 1))
        estimator.fit(X, y, sample_weight=weights)


@ignore_warnings(category=(FutureWarning))
def check_sample_weights_list(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type list in the 'fit' function.
    if has_fit_parameter(estimator_orig, "sample_weight"):
        estimator = clone(estimator_orig)
        rnd = np.random.RandomState(0)
        n_samples = 30
        X = _pairwise_estimator_convert_X(
            rnd.uniform(size=(n_samples, 3)), estimator_orig
        )
        y = np.arange(n_samples) % 3
        y = _enforce_estimator_tags_y(estimator, y)
        sample_weight = [3] * n_samples
        # Test that estimators don't raise any exception
        estimator.fit(X, y, sample_weight=sample_weight)


@ignore_warnings(category=FutureWarning)
def check_sample_weights_shape(name, estimator_orig):
    # check that estimators raise an error if sample_weight
    # shape mismatches the input
    if has_fit_parameter(estimator_orig, "sample_weight") and not (
        hasattr(estimator_orig, "_pairwise") and estimator_orig._pairwise
    ):
        estimator = clone(estimator_orig)
        X = np.array(
            [
                [1, 3],
                [1, 3],
                [1, 3],
                [1, 3],
                [2, 1],
                [2, 1],
                [2, 1],
                [2, 1],
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
                [4, 1],
                [4, 1],
                [4, 1],
                [4, 1],
            ]
        )
        y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2])
        y = _enforce_estimator_tags_y(estimator, y)

        estimator.fit(X, y, sample_weight=np.ones(len(y)))

        assert_raises(
            ValueError, estimator.fit, X, y, sample_weight=np.ones(2 * len(y))
        )

        assert_raises(
            ValueError, estimator.fit, X, y, sample_weight=np.ones((len(y), 2))
        )


@ignore_warnings(category=FutureWarning)
def check_sample_weights_invariance(name, estimator_orig):
    # check that the estimators yield same results for
    # unit weights and no weights
    if has_fit_parameter(estimator_orig, "sample_weight") and not (
        hasattr(estimator_orig, "_pairwise") and estimator_orig._pairwise
    ):
        # We skip pairwise because the data is not pairwise

        estimator1 = clone(estimator_orig)
        estimator2 = clone(estimator_orig)
        set_random_state(estimator1, random_state=0)
        set_random_state(estimator2, random_state=0)

        X = np.array(
            [
                [1, 3],
                [1, 3],
                [1, 3],
                [1, 3],
                [2, 1],
                [2, 1],
                [2, 1],
                [2, 1],
                [3, 3],
                [3, 3],
                [3, 3],
                [3, 3],
                [4, 1],
                [4, 1],
                [4, 1],
                [4, 1],
            ],
            dtype=np.dtype("float"),
        )
        y = np.array(
            [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.dtype("int")
        )
        y = _enforce_estimator_tags_y(estimator1, y)

        estimator1.fit(X, y=y, sample_weight=np.ones(shape=len(y)))
        estimator2.fit(X, y=y, sample_weight=None)

        for method in ["predict", "transform"]:
            if hasattr(estimator_orig, method):
                X_pred1 = getattr(estimator1, method)(X)
                X_pred2 = getattr(estimator2, method)(X)
                if sparse.issparse(X_pred1):
                    X_pred1 = X_pred1.toarray()
                    X_pred2 = X_pred2.toarray()
                assert_allclose(
                    X_pred1,
                    X_pred2,
                    err_msg="For %s sample_weight=None is not"
                    " equivalent to sample_weight=ones" % name,
                )


@ignore_warnings(category=(FutureWarning, UserWarning))
def check_dtype_object(name, estimator_orig):
    # check that estimators treat dtype object as numeric if possible
    rng = np.random.RandomState(0)
    X = _pairwise_estimator_convert_X(rng.rand(40, 10), estimator_orig)
    X = X.astype(object)
    tags = estimator_orig._get_tags()
    y = (X[:, 0] * 4).astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)

    if hasattr(estimator, "transform"):
        estimator.transform(X)

    try:
        estimator.fit(X, y.astype(object))
    except Exception as e:
        if "Unknown label type" not in str(e):
            raise

    if "string" not in tags["X_types"]:
        X[0, 0] = {"foo": "bar"}
        msg = "argument must be a string.* number"
        assert_raises_regex(TypeError, msg, estimator.fit, X, y)
    else:
        # Estimators supporting string will not call np.asarray to convert the
        # data to numeric and therefore, the error will not be raised.
        # Checking for each element dtype in the input array will be costly.
        # Refer to #11401 for full discussion.
        estimator.fit(X, y)


@ignore_warnings
def check_dict_unchanged(name, estimator_orig):
    # this estimator raises
    # ValueError: Found array with 0 feature(s) (shape=(23, 0))
    # while a minimum of 1 is required.
    # error
    if name in ["SpectralCoclustering"]:
        return
    rnd = np.random.RandomState(0)
    if name in ["RANSACRegressor"]:
        X = 3 * rnd.uniform(size=(20, 3))
    else:
        X = 2 * rnd.uniform(size=(20, 3))

    X = _pairwise_estimator_convert_X(X, estimator_orig)

    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    set_random_state(estimator, 1)

    estimator.fit(X, y)
    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            dict_before = estimator.__dict__.copy()
            getattr(estimator, method)(X)
            assert estimator.__dict__ == dict_before, (
                "Estimator changes __dict__ during %s" % method
            )


@deprecated(
    "is_public_parameter is deprecated in version "
    "0.22 and will be removed in version 0.24."
)
def is_public_parameter(attr):
    return _is_public_parameter(attr)


def _is_public_parameter(attr):
    return not (attr.startswith("_") or attr.endswith("_"))


@ignore_warnings(category=FutureWarning)
def check_dont_overwrite_parameters(name, estimator_orig):
    # check that fit method only changes or sets private attributes
    if hasattr(estimator_orig.__init__, "deprecated_original"):
        # to not check deprecated classes
        return
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(int)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [
        key for key in dict_after_fit.keys() if _is_public_parameter(key)
    ]

    attrs_added_by_fit = [
        key for key in public_keys_after_fit if key not in dict_before_fit.keys()
    ]

    # check that fit doesn't add any public attribute
    assert not attrs_added_by_fit, (
        "Estimator adds public attribute(s) during"
        " the fit method."
        " Estimators are only allowed to add private attributes"
        " either started with _ or ended"
        " with _ but %s added" % ", ".join(attrs_added_by_fit)
    )

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [
        key
        for key in public_keys_after_fit
        if (dict_before_fit[key] is not dict_after_fit[key])
    ]

    assert not attrs_changed_by_fit, (
        "Estimator changes public attribute(s) during"
        " the fit method. Estimators are only allowed"
        " to change attributes started"
        " or ended with _, but"
        " %s changed" % ", ".join(attrs_changed_by_fit)
    )


@ignore_warnings(category=FutureWarning)
def check_fit2d_predict1d(name, estimator_orig):
    # check by fitting a 2d array and predicting with a 1d array
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(int)
    tags = estimator_orig._get_tags()
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)
    if tags["no_validation"]:
        # FIXME this is a bit loose
        return

    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            assert_raise_message(
                ValueError, "Reshape your data", getattr(estimator, method), X[0]
            )


def _apply_on_subsets(func, X):
    # apply function on the whole set and on mini batches
    result_full = func(X)
    n_features = X.shape[1]
    result_by_batch = [func(batch.reshape(1, n_features)) for batch in X]

    # func can output tuple (e.g. score_samples)
    if type(result_full) == tuple:
        result_full = result_full[0]
        result_by_batch = list(map(lambda x: x[0], result_by_batch))

    if sparse.issparse(result_full):
        result_full = result_full.A
        result_by_batch = [x.A for x in result_by_batch]

    return np.ravel(result_full), np.ravel(result_by_batch)


@ignore_warnings(category=FutureWarning)
def check_methods_subset_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on mini batches or the whole set
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in [
        "predict",
        "transform",
        "decision_function",
        "score_samples",
        "predict_proba",
    ]:

        msg = (
            "{method} of {name} is not invariant when applied " "to a subset."
        ).format(method=method, name=name)

        if hasattr(estimator, method):
            result_full, result_by_batch = _apply_on_subsets(
                getattr(estimator, method), X
            )
            assert_allclose(result_full, result_by_batch, atol=1e-7, err_msg=msg)


@ignore_warnings
def check_fit2d_1sample(name, estimator_orig):
    # Check that fitting a 2d array with only one sample either works or
    # returns an informative message. The error message should either mention
    # the number of samples or the number of classes.
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(1, 10))
    X = _pairwise_estimator_convert_X(X, estimator_orig)

    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)

    # min_cluster_size cannot be less than the data size for OPTICS.
    if name == "OPTICS":
        estimator.set_params(min_samples=1)

    msgs = [
        "1 sample",
        "n_samples = 1",
        "n_samples=1",
        "one sample",
        "1 class",
        "one class",
    ]

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if all(msg not in repr(e) for msg in msgs):
            raise e


@ignore_warnings
def check_fit2d_1feature(name, estimator_orig):
    # check fitting a 2d array with only 1 feature either works or returns
    # informative message
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(10, 1))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1
    # ensure two labels in subsample for RandomizedLogisticRegression
    if name == "RandomizedLogisticRegression":
        estimator.sample_fraction = 1
    # ensure non skipped trials for RANSACRegressor
    if name == "RANSACRegressor":
        estimator.residual_threshold = 0.5

    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)

    msgs = ["1 feature(s)", "n_features = 1", "n_features=1"]

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if all(msg not in repr(e) for msg in msgs):
            raise e


@ignore_warnings
def check_fit1d(name, estimator_orig):
    # check fitting 1d X array raises a ValueError
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20))
    y = X.astype(int)
    estimator = clone(estimator_orig)
    tags = estimator._get_tags()
    if tags["no_validation"]:
        # FIXME this is a bit loose
        return
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    assert_raises(ValueError, estimator.fit, X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_general(name, transformer, readonly_memmap=False):
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, transformer)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    _check_transformer(name, transformer, X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_data_not_an_array(name, transformer):
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X -= X.min() - 0.1
    X = _pairwise_estimator_convert_X(X, transformer)
    this_X = _NotAnArray(X)
    this_y = _NotAnArray(np.asarray(y))
    _check_transformer(name, transformer, this_X, this_y)
    # try the same with some list
    _check_transformer(name, transformer, X.tolist(), y.tolist())


@ignore_warnings(category=FutureWarning)
def check_transformers_unfitted(name, transformer):
    X, y = _housing_subset()

    transformer = clone(transformer)
    with assert_raises(
        (AttributeError, ValueError),
        msg="The unfitted "
        "transformer {} does not raise an error when "
        "transform is called. Perhaps use "
        "check_is_fitted in transform.".format(name),
    ):
        transformer.transform(X)


def _check_transformer(name, transformer_orig, X, y):
    n_samples, n_features = np.asarray(X).shape
    transformer = clone(transformer_orig)
    set_random_state(transformer)

    # fit

    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[np.asarray(y), np.asarray(y)]
        y_[::2, 1] *= 2
        if isinstance(X, _NotAnArray):
            y_ = _NotAnArray(y_)
    else:
        y_ = y

    transformer.fit(X, y_)
    # fit_transform method should work on non fitted estimator
    transformer_clone = clone(transformer)
    X_pred = transformer_clone.fit_transform(X, y=y_)

    if isinstance(X_pred, tuple):
        for x_pred in X_pred:
            assert x_pred.shape[0] == n_samples
    else:
        # check for consistent n_samples
        assert X_pred.shape[0] == n_samples

    if hasattr(transformer, "transform"):
        if name in CROSS_DECOMPOSITION:
            X_pred2 = transformer.transform(X, y_)
            X_pred3 = transformer.fit_transform(X, y=y_)
        else:
            X_pred2 = transformer.transform(X)
            X_pred3 = transformer.fit_transform(X, y=y_)

        if transformer_orig._get_tags()["non_deterministic"]:
            msg = name + " is non deterministic"
            raise SkipTest(msg)
        if isinstance(X_pred, tuple) and isinstance(X_pred2, tuple):
            for x_pred, x_pred2, x_pred3 in zip(X_pred, X_pred2, X_pred3):
                assert_allclose_dense_sparse(
                    x_pred,
                    x_pred2,
                    atol=1e-2,
                    err_msg="fit_transform and transform outcomes "
                    "not consistent in %s" % transformer,
                )
                assert_allclose_dense_sparse(
                    x_pred,
                    x_pred3,
                    atol=1e-2,
                    err_msg="consecutive fit_transform outcomes "
                    "not consistent in %s" % transformer,
                )
        else:
            assert_allclose_dense_sparse(
                X_pred,
                X_pred2,
                err_msg="fit_transform and transform outcomes "
                "not consistent in %s" % transformer,
                atol=1e-2,
            )
            assert_allclose_dense_sparse(
                X_pred,
                X_pred3,
                atol=1e-2,
                err_msg="consecutive fit_transform outcomes "
                "not consistent in %s" % transformer,
            )
            assert _num_samples(X_pred2) == n_samples
            assert _num_samples(X_pred3) == n_samples

        # raises error on malformed input for transform
        if (
            hasattr(X, "shape")
            and not transformer._get_tags()["stateless"]
            and X.ndim == 2
            and X.shape[1] > 1
        ):

            # If it's not an array, it does not have a 'T' property
            with assert_raises(
                ValueError,
                msg="The transformer {} does "
                "not raise an error when the number of "
                "features in transform is different from"
                " the number of features in "
                "fit.".format(name),
            ):
                transformer.transform(X[:, :-1])


@ignore_warnings
def check_pipeline_consistency(name, estimator_orig):
    if estimator_orig._get_tags()["non_deterministic"]:
        msg = name + " is non deterministic"
        raise SkipTest(msg)

    # check that make_pipeline(est) gives same score as est
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)
    pipeline = make_pipeline(estimator)
    estimator.fit(X, y)
    pipeline.fit(X, y)

    funcs = ["score", "fit_transform"]

    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func_pipeline = getattr(pipeline, func_name)
            result = func(X, y)
            result_pipe = func_pipeline(X, y)
            assert_allclose_dense_sparse(result, result_pipe)


@ignore_warnings
def check_fit_score_takes_y(name, estimator_orig):
    # check that all estimators accept an optional y
    # in fit and score so they can be used in pipelines
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = rnd.uniform(size=(n_samples, 3))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = np.arange(n_samples) % 3
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)

    funcs = ["fit", "score", "partial_fit", "fit_predict", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            args = [p.name for p in signature(func).parameters.values()]
            if args[0] == "self":
                # if_delegate_has_method makes methods into functions
                # with an explicit "self", so need to shift arguments
                args = args[1:]
            assert args[1] in ["y", "Y"], (
                "Expected y or Y as second argument for method "
                "%s of %s. Got arguments: %r."
                % (func_name, type(estimator).__name__, args)
            )


@ignore_warnings
def check_estimators_dtypes(name, estimator_orig):
    rnd = np.random.RandomState(0)
    X_train_32 = 3 * rnd.uniform(size=(20, 5)).astype(np.float32)
    X_train_32 = _pairwise_estimator_convert_X(X_train_32, estimator_orig)
    X_train_64 = X_train_32.astype(np.float64)
    X_train_int_64 = X_train_32.astype(np.int64)
    X_train_int_32 = X_train_32.astype(np.int32)
    y = X_train_int_64[:, 0]
    y = _enforce_estimator_tags_y(estimator_orig, y)

    methods = ["predict", "transform", "decision_function", "predict_proba"]

    for X_train in [X_train_32, X_train_64, X_train_int_64, X_train_int_32]:
        estimator = clone(estimator_orig)
        set_random_state(estimator, 1)
        estimator.fit(X_train, y)

        for method in methods:
            if hasattr(estimator, method):
                getattr(estimator, method)(X_train)


@ignore_warnings(category=FutureWarning)
def check_estimators_empty_data_messages(name, estimator_orig):
    e = clone(estimator_orig)
    set_random_state(e, 1)

    X_zero_samples = np.empty(0).reshape(0, 3)
    # The precise message can change depending on whether X or y is
    # validated first. Let us test the type of exception only:
    with assert_raises(
        ValueError,
        msg="The estimator {} does not"
        " raise an error when an empty data is used "
        "to train. Perhaps use "
        "check_array in train.".format(name),
    ):
        e.fit(X_zero_samples, [])

    X_zero_features = np.empty(0).reshape(3, 0)
    # the following y should be accepted by both classifiers and regressors
    # and ignored by unsupervised models
    y = _enforce_estimator_tags_y(e, np.array([1, 0, 1]))
    msg = r"0 feature\(s\) \(shape=\(3, 0\)\) while a minimum of \d* " "is required."
    assert_raises_regex(ValueError, msg, e.fit, X_zero_features, y)


@ignore_warnings
def check_estimators_pickle(name, estimator_orig):
    """Test that we can pickle all estimators."""
    check_methods = ["predict", "transform", "decision_function", "predict_proba"]

    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )

    # some estimators can't do features less than 0
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)

    #     import pdb
    #     pdb.set_trace()
    tags = estimator_orig._get_tags()
    # include NaN values when the estimator should deal with them
    if tags["allow_nan"]:
        # set randomly 10 elements to np.nan
        rng = np.random.RandomState(42)
        mask = rng.choice(X.size, 10, replace=False)
        X.reshape(-1)[mask] = np.nan

    estimator = clone(estimator_orig)

    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)
    #     import pdb
    #     pdb.set_trace()
    estimator.fit(X, y)

    result = dict()
    for method in check_methods:
        if hasattr(estimator, method):
            result[method] = getattr(estimator, method)(X)

    # pickle and unpickle!
    pickled_estimator = pickle.dumps(estimator)
    if estimator.__module__.startswith("sklearn."):
        assert b"version" in pickled_estimator
    unpickled_estimator = pickle.loads(pickled_estimator)

    result = dict()
    for method in check_methods:
        if hasattr(estimator, method):
            result[method] = getattr(estimator, method)(X)

    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)
        assert_allclose_dense_sparse(result[method], unpickled_result)


@ignore_warnings(category=FutureWarning)
def check_estimators_partial_fit_n_features(name, estimator_orig):
    # check if number of features changes between calls to partial_fit.
    if not hasattr(estimator_orig, "partial_fit"):
        return
    estimator = clone(estimator_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X -= X.min()
    y = _enforce_estimator_tags_y(estimator_orig, y)

    try:
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
    except NotImplementedError:
        return

    with assert_raises(
        ValueError,
        msg="The estimator {} does not raise an"
        " error when the number of features"
        " changes between calls to "
        "partial_fit.".format(name),
    ):
        estimator.partial_fit(X[:, :-1], y)


@ignore_warnings(category=FutureWarning)
def check_clustering(name, clusterer_orig, readonly_memmap=False):
    clusterer = clone(clusterer_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X, y = shuffle(X, y, random_state=7)
    X = StandardScaler().fit_transform(X)
    rng = np.random.RandomState(7)
    X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])

    if readonly_memmap:
        X, y, X_noise = create_memmap_backed_data([X, y, X_noise])

    n_samples, n_features = X.shape
    # catch deprecation and neighbors warnings
    if hasattr(clusterer, "n_clusters"):
        clusterer.set_params(n_clusters=3)
    set_random_state(clusterer)
    if name == "AffinityPropagation":
        clusterer.set_params(preference=-100)
        clusterer.set_params(max_iter=100)

    # fit
    clusterer.fit(X)
    # with lists
    clusterer.fit(X.tolist())

    pred = clusterer.labels_
    assert pred.shape == (n_samples,)
    assert adjusted_rand_score(pred, y) > 0.4
    if clusterer._get_tags()["non_deterministic"]:
        return
    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert pred.dtype in [np.dtype("int32"), np.dtype("int64")]
    assert pred2.dtype in [np.dtype("int32"), np.dtype("int64")]

    # Add noise to X to test the possible values of the labels
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every cluster. Equivalently
    # labels_ should contain all the consecutive values between its
    # min and its max.
    labels_sorted = np.unique(labels)
    assert_array_equal(
        labels_sorted, np.arange(labels_sorted[0], labels_sorted[-1] + 1)
    )

    # Labels are expected to start at 0 (no noise) or -1 (if noise)
    assert labels_sorted[0] in [0, -1]
    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, "n_clusters"):
        n_clusters = getattr(clusterer, "n_clusters")
        assert n_clusters - 1 >= labels_sorted[-1]
    # else labels should be less than max(labels_) which is necessarily true


@ignore_warnings(category=FutureWarning)
def check_clusterer_compute_labels_predict(name, clusterer_orig):
    """Check that predict is invariant of compute_labels."""
    X, y = make_blobs(n_samples=20, random_state=0)
    clusterer = clone(clusterer_orig)
    set_random_state(clusterer)

    if hasattr(clusterer, "compute_labels"):
        # MiniBatchKMeans
        X_pred1 = clusterer.fit(X).predict(X)
        clusterer.set_params(compute_labels=False)
        X_pred2 = clusterer.fit(X).predict(X)
        assert_array_equal(X_pred1, X_pred2)


@ignore_warnings(category=FutureWarning)
def check_estimators_fit_returns_self(name, estimator_orig, readonly_memmap=False):
    """Check if self is returned when calling fit."""
    X, y = make_blobs(random_state=0, n_samples=21)
    # some want non-negative input
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    set_random_state(estimator)
    assert estimator.fit(X, y) is estimator


@ignore_warnings(category=FutureWarning)
def check_estimators_overwrite_params(name, estimator_orig):
    X, y = make_blobs(random_state=0, n_samples=21)
    # some want non-negative input
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    # Fit the model
    estimator.fit(X, y)

    # Compare the state of the model parameters with the original parameters
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate "
            " the parameter %s from %s to %s during fit."
            % (name, param_name, original_value, new_value)
        )


@ignore_warnings(category=FutureWarning)
def check_no_attributes_set_in_init(name, estimator_orig):
    """Check setting during init."""
    estimator = clone(estimator_orig)
    if hasattr(type(estimator).__init__, "deprecated_original"):
        return

    init_params = _get_args(type(estimator).__init__)
    if IS_PYPY:
        # __init__ signature has additional objects in PyPy
        for key in ["obj"]:
            if key in init_params:
                init_params.remove(key)
    parents_init_params = [
        param
        for params_parent in (_get_args(parent) for parent in type(estimator).__mro__)
        for param in params_parent
    ]

    # Test for no setting apart from parameters during init
    invalid_attr = set(vars(estimator)) - set(init_params) - set(parents_init_params)
    assert not invalid_attr, (
        "Estimator %s should not set any attribute apart"
        " from parameters during init. Found attributes %s."
        % (name, sorted(invalid_attr))
    )
    # Ensure that each parameter is set in init
    invalid_attr = set(init_params) - set(vars(estimator)) - {"self"}
    assert not invalid_attr, (
        "Estimator %s should store all parameters"
        " as an attribute during init. Did not find "
        "attributes %s." % (name, sorted(invalid_attr))
    )


# TODO: remove in 0.24
@deprecated(
    "enforce_estimator_tags_y is deprecated in version "
    "0.22 and will be removed in version 0.24."
)
def enforce_estimator_tags_y(estimator, y):
    return _enforce_estimator_tags_y(estimator, y)


def _enforce_estimator_tags_y(estimator, y):
    # Estimators with a `requires_positive_y` tag only accept strictly positive
    # data
    if estimator._get_tags()["requires_positive_y"]:
        # Create strictly positive y. The minimal increment above 0 is 1, as
        # y could be of integer dtype.
        y += 1 + abs(y.min())
    # Estimators with a `binary_only` tag only accept up to two unique y values
    if estimator._get_tags()["binary_only"] and y.size > 0:
        y = np.where(y == y.flat[0], y, y.flat[0] + 1)
    # Estimators in mono_output_task_error raise ValueError if y is of 1-D
    # Convert into a 2-D y for those estimators.
    if estimator._get_tags()["multioutput_only"]:
        return np.reshape(y, (-1, 1))
    return y


def _enforce_estimator_tags_x(estimator, X):
    # Estimators with a `_pairwise` tag only accept
    # X of shape (`n_samples`, `n_samples`)
    if hasattr(estimator, "_pairwise"):
        X = X.dot(X.T)
    # Estimators with `1darray` in `X_types` tag only accept
    # X of shape (`n_samples`,)
    if "1darray" in estimator._get_tags()["X_types"]:
        X = X[:, 0]
    # Estimators with a `requires_positive_X` tag only accept
    # strictly positive data
    if estimator._get_tags()["requires_positive_X"]:
        X -= X.min()
    return X


@ignore_warnings(category=FutureWarning)
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    # Test that estimators that are not transformers with a parameter
    # max_iter, return the attribute of n_iter_ at least 1.

    # These models are dependent on external solvers like
    # libsvm and accessing the iter parameter is non-trivial.
    not_run_check_n_iter = [
        "Ridge",
        "SVR",
        "NuSVR",
        "NuSVC",
        "RidgeClassifier",
        "SVC",
        "RandomizedLasso",
        "LogisticRegressionCV",
        "LinearSVC",
        "LogisticRegression",
    ]

    # Tested in test_transformer_n_iter
    not_run_check_n_iter += CROSS_DECOMPOSITION
    if name in not_run_check_n_iter:
        return

    # LassoLars stops early for the default alpha=1.0 the iris dataset.
    if name == "LassoLars":
        estimator = clone(estimator_orig).set_params(alpha=0.0)
    else:
        estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        iris = load_iris()
        X, y_ = iris.data, iris.target
        y_ = _enforce_estimator_tags_y(estimator, y_)

        set_random_state(estimator, 0)

        estimator.fit(X, y_)

        assert estimator.n_iter_ >= 1


@ignore_warnings(category=FutureWarning)
def check_transformer_n_iter(name, estimator_orig):
    # Test that transformers with a parameter max_iter, return the
    # attribute of n_iter_ at least 1.
    estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        if name in CROSS_DECOMPOSITION:
            # Check using default data
            X = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2.0, 5.0, 4.0]]
            y_ = [[0.1, -0.2], [0.9, 1.1], [0.1, -0.5], [0.3, -0.2]]

        else:
            X, y_ = make_blobs(
                n_samples=30,
                centers=[[0, 0, 0], [1, 1, 1]],
                random_state=0,
                n_features=2,
                cluster_std=0.1,
            )
            X -= X.min() - 0.1
        set_random_state(estimator, 0)
        estimator.fit(X, y_)

        # These return a n_iter per component.
        if name in CROSS_DECOMPOSITION:
            for iter_ in estimator.n_iter_:
                assert iter_ >= 1
        else:
            assert estimator.n_iter_ >= 1


@ignore_warnings(category=FutureWarning)
def check_get_params_invariance(name, estimator_orig):
    # Checks if get_params(deep=False) is a subset of get_params(deep=True)
    e = clone(estimator_orig)

    shallow_params = e.get_params(deep=False)
    deep_params = e.get_params(deep=True)

    assert all(item in deep_params.items() for item in shallow_params.items())


@ignore_warnings(category=FutureWarning)
def check_set_params(name, estimator_orig):
    # Check that get_params() returns the same thing
    # before and after set_params() with some fuzz
    estimator = clone(estimator_orig)

    orig_params = estimator.get_params(deep=False)
    msg = "get_params result does not match what was passed to set_params"

    estimator.set_params(**orig_params)
    curr_params = estimator.get_params(deep=False)
    assert set(orig_params.keys()) == set(curr_params.keys()), msg
    for k, v in curr_params.items():
        assert orig_params[k] is v, msg

    # some fuzz values
    test_values = [-np.inf, np.inf, None]

    test_params = deepcopy(orig_params)
    for param_name in orig_params.keys():
        default_value = orig_params[param_name]
        for value in test_values:
            test_params[param_name] = value
            try:
                estimator.set_params(**test_params)
            except (TypeError, ValueError) as e:
                e_type = e.__class__.__name__
                # Exception occurred, possibly parameter validation
                warnings.warn(
                    "{0} occurred during set_params of param {1} on "
                    "{2}. It is recommended to delay parameter "
                    "validation until fit.".format(e_type, param_name, name)
                )

                change_warning_msg = (
                    "Estimator's parameters changed after "
                    "set_params raised {}".format(e_type)
                )
                params_before_exception = curr_params
                curr_params = estimator.get_params(deep=False)
                try:
                    assert set(params_before_exception.keys()) == set(
                        curr_params.keys()
                    )
                    for k, v in curr_params.items():
                        assert params_before_exception[k] is v
                except AssertionError:
                    warnings.warn(change_warning_msg)
            else:
                curr_params = estimator.get_params(deep=False)
                assert set(test_params.keys()) == set(curr_params.keys()), msg
                for k, v in curr_params.items():
                    assert test_params[k] is v, msg
        test_params[param_name] = default_value


def check_fit_idempotent(name, estimator_orig):
    # Check that est.fit(X) is the same as est.fit(X).fit(X). Ideally we would
    # check that the estimated parameters during training (e.g. coefs_) are
    # the same, but having a universal comparison function for those
    # attributes is difficult and full of edge cases. So instead we check that
    # predict(), predict_proba(), decision_function() and transform() return
    # the same results.

    check_methods = ["predict", "transform", "decision_function", "predict_proba"]
    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params().keys():
        estimator.set_params(warm_start=False)

    n_samples = 100
    X = rng.normal(loc=100, size=(n_samples, 2))
    X = _pairwise_estimator_convert_X(X, estimator)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)

    train, test = next(ShuffleSplit(test_size=0.2, random_state=rng).split(X))
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    # Fit for the first time
    estimator.fit(X_train, y_train)

    result = {
        method: getattr(estimator, method)(X_test)
        for method in check_methods
        if hasattr(estimator, method)
    }

    # Fit again
    set_random_state(estimator)
    estimator.fit(X_train, y_train)

    for method in check_methods:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(X_test)
            if np.issubdtype(new_result.dtype, np.floating):
                tol = 2 * np.finfo(new_result.dtype).eps
            else:
                tol = 2 * np.finfo(np.float64).eps
            assert_allclose_dense_sparse(
                result[method],
                new_result,
                atol=max(tol, 1e-9),
                rtol=max(tol, 1e-7),
                err_msg="Idempotency check failed for method {}".format(method),
            )


def check_n_features_in(name, estimator_orig):
    # Make sure that n_features_in_ attribute doesn't exist until fit is
    # called, and that its value is correct.

    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    n_samples = 100
    X = rng.normal(loc=100, size=(n_samples, 2))
    X = _pairwise_estimator_convert_X(X, estimator)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)

    assert not hasattr(estimator, "n_features_in_")
    estimator.fit(X, y)
    if hasattr(estimator, "n_features_in_"):
        assert estimator.n_features_in_ == X.shape[1]
    else:
        warnings.warn(
            "As of scikit-learn 0.23, estimators should expose a "
            "n_features_in_ attribute, unless the 'no_validation' tag is "
            "True. This attribute should be equal to the number of features "
            "passed to the fit method. "
            "An error will be raised from version 0.25 when calling "
            "check_estimator(). "
            "See SLEP010: "
            "https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html",  # noqa
            FutureWarning,
        )
