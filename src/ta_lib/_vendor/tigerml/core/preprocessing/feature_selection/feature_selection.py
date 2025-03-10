import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import assert_all_finite, check_array
from sklearn.utils.validation import check_is_fitted

from .feature_selection_by_statistic import (
    chi_square,
    corr_coef,
    corr_ratio,
    cramer_v,
    f_score,
    mutual_value,
    woe_iv,
)


def _check_X(X, columns=None):
    """Change array to DataFrame."""
    if isinstance(X, (pd.DataFrame)):
        return X
    elif isinstance(X, (np.ndarray)):
        if columns is None:
            return pd.DataFrame(X)
        else:
            return pd.DataFrame(X, columns=columns)
    elif isinstance(X, pd.Series):
        return pd.DataFrame(X, columns=X.name)
    elif isinstance(X, (list, tuple)):
        X = np.array(X)
        return pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    elif hasattr(X, ("__array__")):
        data = X.__array__()
        return pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Base class for feature selection.

    Parameters
    ----------
    estimator: object
        The base estimator from which the transformer is built.

            - If selection_type is recursion, any estimator can be used.
            - If selection_type is regularization, Lasso/Ridge estimator
              should be passed.

    selection_type: {"recursion","regularization"} str, default = "recursion".
        Method used to select features

            - If recursion, Forward or backward selection is used and
              selection_params has to be passed.
            - If regularization, Lasso or Ridge regression is used.

    selection_params: dict, default = {'forward':False,verbose:False,k_features:'best'}
        params associated with class `SequentialFeatureSelector` in
        the form {forward: bool, verbose: bool,k_features: no_of_features}

            - If forward parameter is True, then forward feature selection
              is used else, backward feature elimination method is used.

    x_train: dataframe
        dataframe for train data

    y_train: Series or np.array
        target data

    Attributes
    ----------
    _k_features: list
        columns that are selected

    Examples
    --------
    >>> from ta_lib.data_processing.feature_selection import FeatureSelector
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=4,
    ...                            n_informative=2,
    ...                            random_state=0, shuffle=False)
    >>> fs = FeatureSelector(Ridge(alpha=0.5),selection_type="regularization")
    >>> fs.fit(X,y)
    FeatureSelector(...)
    >>> transformed_df = fs.transform(X)
    """

    def __init__(
        self,
        estimator,
        selection_type="recursion",
        selection_params={"forward": False, "verbose": False, "k_features": "best"},
    ):
        """Initialize Estimator."""
        self.estimator = estimator
        self.selection_type = selection_type
        self.selection_params = selection_params

    def fit(self, X, y):
        """Build feature selection estimator from X, y."""
        check_array(X)
        self._check_params(X, y)
        self._feat, self._k_features = self.select_features(
            X, y, self.estimator, self.selection_type, self.selection_params
        )
        return self

    def transform(self, X):
        """Select features for X."""
        check_array(X)
        X = _check_X(X)
        if not (0 <= len(self._k_features) <= X.shape[1]):
            raise ValueError("Cannot transform Data")
        df_out = self._feat.transform(X)
        # x_train = pd.DataFrame(df_out, columns=self._k_features)
        return df_out

    def fit_transform(self, X, y):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def _check_params(self, X, y):
        if self.estimator is None:
            raise ValueError("Estimator is mandatory.")
        if y is None:
            raise ValueError("Target is required.")
        if not (
            self.selection_params["k_features"] == "best"
            or 0 <= self.selection_params["k_features"] <= X.shape[1]
        ):
            raise ValueError("Pass valid parameters")

    def _more_tags(self):
        return {
            "requires_y": True,
        }

    def select_features(
        self, x_train, y_train, estimator, selection_type, selection_params
    ):
        """Select features from x_train.

        SelectFromModel takes estimator(Lasso/Ridge) as an input and
        after fitting we can select the features whose coefs are greater
        than 0.
        SequentialFeatureSelector from mlxtend does a forward or
        backward feature selection based on the estimator we choose

        Parameters
        ----------
        x_train: dataframe
            train data to fit the estimator

        y_train: Series or np.array
            target variable

        estimator : object
            The base estimator from which the transformer is built.

            - If selection_type is recursion, any estimator can be used.
            - If selection_type is regularization, Lasso/Ridge estimator
              should be passed.

        selection_type : {"recursion","regularization"} str, default = "recursion".
            Method used to select features

            - If recursion, Forward or backward selection is used
              and selection_params has to be passed.
            - If regularization, Lasso or Ridge regression is used.

        selection_params : dict, default =
        {'forward':False,verbose:False,k_features:'best'}
            params associated with class `SequentialFeatureSelector` in the
            form {forward: bool, verbose: bool,k_features: no_of_features}

            - If forward parameter is True, then forward feature selection
              is used else, backward feature elimination method is used.

        Returns
        -------
        fitted estimator: estimator
            estimator fit on train data to get features.
            This is used to transform the test data
        selected_features_list: list
            list of features selected

        """
        x_train = _check_X(x_train)
        if selection_type == "regularization":
            fe_sel_ = SelectFromModel(estimator)
            fe_sel_.fit(x_train, y_train)
            selected_feat = x_train.columns[(fe_sel_.get_support())]
            # get_support returns list of Bool values where a column is important or not
            return fe_sel_, selected_feat
        else:
            try:
                from mlxtend.feature_selection import (  # noqa
                    SequentialFeatureSelector as sfs,
                )
            except ImportError as e:
                raise ImportError(
                    "{} using recursion requires {} from {}. "
                    "You can install with `pip install {}`".format(
                        "select_features",
                        "SequentialFeatureSelector",
                        "mlxtend",
                        "mlxtend",
                    )
                ) from e
            fe_sel_ = sfs(estimator, **selection_params)
            fe_sel_.fit(x_train, y_train)
            return fe_sel_, fe_sel_.k_feature_names_


class FeatureSelectorStatistic(SelectKBest):
    """Base class for feature selection using the statistics.

    Parameters
    ----------
    statistic: {"f_score", "chi_square",
                "corr_coef", "corr_ratio",
                "cramer_v", "f_score",
                "mutual_value", "woe_iv"} str, default = "f_score".

        The statistic which will be used to select the top k columns.

    k: int, optional, default=2
        Number of top features to select.
    x_train: dataframe
        dataframe for train data

    y_train: Series or np.array
        target data

    Examples
    --------
    >>> from ta_lib.data_processing.feature_selection import FeatureSelectorStatistic
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=4,
    ...                            n_informative=2,
    ...                            random_state=0, shuffle=False)
    >>> fs = FeatureSelectorStatistic()
    >>> fs.fit(X,y)
    FeatureSelectorStatistic(...)
    >>> feature_scores = fs.scores_
    >>> transformed_df = fs.transform(X)
    """

    def __init__(self, statistic="f_score", k=2):
        """Initialize Estimator."""
        super().__init__()
        statistic_options = {
            "f_score": f_score,
            "chi_square": chi_square,
            "corr_coef": corr_coef,
            "corr_ratio": corr_ratio,
            "cramer_v": cramer_v,
            "mutual_value": mutual_value,
            "woe_iv": woe_iv,
        }
        self.statistic = statistic
        self.k = k
        self.score_func = statistic_options[self.statistic]

    def _more_tags(self):
        return {
            "requires_y": True,
        }

    def fit(self, X, y, bin_type=None, nbins=None):
        # overwrites the fit() in _BaseFilter (parent class of SelectKBest class)
        # in order to avoid X, y = check_X_y(X, y, ['csr', 'csc'], multi_output=True)
        # so that categorical columns can be passed.
        """Run score function on (X, y) and get the appropriate features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        bin_type : {"cut", "qcut"} str. Type of binning required, while discretizing
                  numeric columns, when statistic = "woe_iv" (optional).

        nbins : No. of bins required, while discretizing numeric columns,
               when statistic = "woe_iv" (optional).

        Returns
        -------
        self : object
        """
        if not callable(self.score_func):
            raise TypeError(
                "The score function should be a callable, %s (%s) "
                "was passed." % (self.score_func, type(self.score_func))
            )

        self._check_params(X, y)
        if bin_type:
            if self.statistic != "woe_iv":
                raise Exception(
                    "'bin_type' parameter is applicable "
                    "only when 'woe_iv' statistic is used"
                )
            if nbins:
                score_func_ret = self.score_func(X, y, bin_type, nbins)
            else:
                score_func_ret = self.score_func(X, y, bin_type)
        elif nbins:
            if self.statistic != "woe_iv":
                raise Exception(
                    "'nbins' parameter is applicable "
                    "only when 'woe_iv' statistic is used"
                )
            if bin_type:
                score_func_ret = self.score_func(X, y, bin_type, nbins)
            else:
                score_func_ret = self.score_func(X, y, "cut", nbins)
        else:
            score_func_ret = self.score_func(X, y)
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None
        self.scores_ = np.asarray(self.scores_)
        return self
