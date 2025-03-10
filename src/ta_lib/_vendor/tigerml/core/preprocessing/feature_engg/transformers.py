import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .feature_eng import (
    feature_engg_woe_binning_transform,
    feature_engg_woe_binning_wrapper,
    get_num_cols,
    supervised_binning_fit,
    supervised_binning_transform,
    unsupervised_binning,
)


def check_X(X, columns=None):
    """Change array to DataFrame."""
    if isinstance(X, (pd.DataFrame)):
        return X
    elif isinstance(X, (np.ndarray)):
        if columns is None:
            return pd.DataFrame(X, columns=["s_" + str(i) for i in range(X.shape[1])])
        else:
            return pd.DataFrame(X, columns=columns)
    elif isinstance(X, pd.Series):
        return X.to_frame()
    elif isinstance(X, (list, tuple)):
        X = np.array(X)
        return pd.DataFrame(X, columns=["s_" + str(i) for i in range(X.shape[1])])
    elif hasattr(X, ("__array__")):
        X = X.__array__()
        return pd.DataFrame(X, columns=["s_" + str(i) for i in range(X.shape[1])])
    return X


def check_y(y):
    """Change anything to array."""
    if hasattr(y, ("__array__")):
        y = y.__array__()
        return np.array(y, dtype=float)
    elif isinstance(y, (list, tuple)):
        return np.array(y)
    else:
        raise ValueError("Expected array")


# Custom Transformer for WoeBinning
class WoeBinningTransformer(BaseEstimator, TransformerMixin):
    """WOE based binning for continuous Xs in binary classification models.

    Parameters
    ----------
    Columns

    """

    def __init__(self, cols=None, encode="ordinal"):
        """Initialize Estimator."""
        self.cols = cols
        self.encode = encode

    def fit(self, X, y):
        """Build WOE based bin limits from the training set (X, y)."""
        valid_encode = ("onehot", "ordinal")
        if self.encode not in valid_encode:
            raise ValueError(
                "Valid options for 'encode' are {}. "
                "Got encode={!r} instead.".format(valid_encode, self.encode)
            )
        cols = None
        if hasattr(X, "columns"):
            cols = X.columns.tolist()
        self._validate_data(X, y, force_all_finite=False)
        X = check_array(X, force_all_finite=False)
        # X, y = check_X_y(X, y, force_all_finite=False)
        X = check_X(X, cols)
        y = check_y(y)
        self.check_params(X, y)
        if isinstance(y, pd.DataFrame):
            y = y["target"]
        if len(np.unique(y.__array__())) <= 2:
            if not (
                np.sum(np.unique(y.__array__()) == [0, 1])
                or np.sum(np.unique(y.__array__()) == [1])
                or np.sum(np.unique(y.__array__()) == [0])
            ):
                lb = preprocessing.LabelBinarizer()
                y = lb.fit_transform(y)[:, 0]
        else:
            ValueError("Pass binary variable")

        self.woe_binner_ = feature_engg_woe_binning_wrapper(X, y, self.cols)
        tmp = pd.DataFrame()
        if "onehot" in self.encode:
            for col in self.woe_binner_.keys():
                category = self.woe_binner_.get(col).woe_df["Category"].tolist()
                dummy_series = pd.Series(category, dtype="str")
                tmp = pd.concat([tmp, dummy_series], axis=1)
            tmp.columns = [f"{i}_woebin" for i in list(self.woe_binner_.keys())]
            self.encode_cols_ = tmp.columns
            self._encoder = OneHotEncoder()
            if len(tmp) > 0:
                for i in tmp.columns:
                    tmp[i] = tmp[i].fillna(tmp[i].iloc[0])
            self._encoder.fit(tmp)
        else:
            self.encode_cols_ = [f"{i}_woebin" for i in list(self.woe_binner_.keys())]
        self._is_fitted = True
        return self

    def transform(self, X):
        """Build binning by using fitted bin limits on X."""
        cols = None
        if hasattr(X, "columns"):
            cols = X.columns.tolist()
        check_is_fitted(self, "_is_fitted")
        Xt = check_array(X, copy=True, force_all_finite=False)
        Xt = check_X(Xt, cols)
        assert isinstance(Xt, (pd.DataFrame, pd.Series))
        cols = self.cols or get_num_cols(Xt)
        if len(cols) == 0:
            raise ValueError("Expects atleast one numeric column")
        Xt = Xt[cols].copy()
        columns_transformed = self.woe_binner_.keys()
        existing = set(list(columns_transformed)).intersection(set(list(Xt.columns)))
        if len(existing) != len(list(self.woe_binner_.keys())):
            raise ValueError("Inconsistent fit and transform")
        Xt = feature_engg_woe_binning_transform(Xt, self.woe_binner_, existing)
        if "onehot" in self.encode:
            Xt = self._encoder.transform(Xt[self.encode_cols_].astype("str")).toarray()
            Xt = pd.DataFrame(Xt, columns=self.get_feature_names())
            return Xt.values
        else:
            for i in self.encode_cols_:
                Xt[i] = Xt[i].astype("category").cat.codes
            return Xt[self.encode_cols_].values

    def fit_transform(self, X, y):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def check_params(self, X, y):
        """Checks if params are present."""
        if y is None:
            raise ValueError("Expected y or Y as second argument for method")
        if y.__array__().dtype == "object":
            raise TypeError("Expected y as Binary")

    def _more_tags(self):
        return {
            "requires_y": True,
            "binary_only": True,
            "requires_fit": True,
            "requires_positive_X": True,
        }

    def get_feature_names(self):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of str of shape (n_features,)
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : ndarray of shape (n_output_features,)
            Array of feature names.
        """
        if "onehot" in self.encode:
            return self._encoder.get_feature_names_out(self.encode_cols_)
        else:
            return self.encode_cols_


class UnsupervisedTransformer(BaseEstimator, TransformerMixin):
    """Unsupervised binning using pd.cut.

    Parameters
    ----------
    cols: `list`
    n: int, default is None
    size_of_bin: int, default is None
    labels: list, default is None
    quantile: int, default is
    value: int, default is None
    size: int, default is None
    left: bool, default is True
    """

    def __init__(
        self,
        cols=None,
        n=None,
        size_of_bin=None,
        labels=None,
        quantile=None,
        value=None,
        size=None,
        left=True,
    ):
        """Initialize Transformer."""
        self.n = n
        self.size_of_bin = size_of_bin
        self.labels = labels
        self.quantile = quantile
        self.value = value
        self.size = size
        self.left = left
        self.cols = cols

    def fit(self, X, y=None):
        """Fit transformer."""
        check_array(X, force_all_finite=False)
        self._validate_data(X, dtype="numeric", force_all_finite=False)
        X = check_X(X)
        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform data."""
        check_array(X, force_all_finite=False)
        check_is_fitted(self, "_is_fitted")
        X = check_X(X, self.cols)
        assert isinstance(X, (pd.DataFrame, pd.Series))
        cols = self.cols or get_num_cols(X)
        X = X[cols].copy()
        for jj in cols:
            X.loc[:, jj], bins = unsupervised_binning(
                X[jj],
                self.n,
                self.size_of_bin,
                self.labels,
                self.quantile,
                self.value,
                self.size,
                self.left,
            )
        # self._features = X.columns
        return X.values

    def _more_tags(self):
        return {"allow_nan": True, "requires_fit": False, "stateless": True}


class SupervisedTransformer(BaseEstimator, TransformerMixin):
    """Unsupervised binning using pd.cut.

    Parameters
    ----------
    cols: `list`
    n: int, default is None
    size_of_bin: int, default is None
    labels: list, default is None
    quantile: int, default is
    value: int, default is None
    size: int, default is None
    left: bool, default is True
    """

    def __init__(self, cols=None, prefix="SPE"):
        """Initialize Transformer."""
        self.cols = cols
        self.prefix = prefix

    def fit(self, X, y):
        """Fit transformer."""
        cols = None
        if hasattr(X, "columns"):
            cols = X.columns.to_list()
        X = check_array(X, force_all_finite=False)
        self._validate_data(X, y, dtype="numeric", force_all_finite=False)
        X, y = check_X_y(X, y)
        y = check_y(y)
        X = check_X(X, cols)
        self.check_params(X, y)
        cols = self.cols or get_num_cols(X)
        if len(cols) == 0:
            raise ValueError("Expects atleast one numeric column")
        X = X[cols]
        self._dict_bin = {}
        for col_ in cols:
            discretizer = supervised_binning_fit(X[col_], y)
            self._dict_bin.update({col_: discretizer})
        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform data."""
        cols = None
        if hasattr(X, "columns"):
            cols = X.columns.tolist()
        X = check_array(X)
        check_is_fitted(self, "_is_fitted")
        X = check_X(X, cols)
        assert isinstance(X, (pd.DataFrame, pd.Series))
        cols = self.cols or get_num_cols(X)
        if len(cols) == 0:
            raise ValueError("Expects atleast one numeric column")
        X = X[cols].copy()
        existing = set(list(X.columns)).intersection(set(list(self._dict_bin.keys())))
        if len(existing) != len(list(self._dict_bin.keys())):
            raise ValueError("Inconsistent fit and transform")
        for jj in existing:
            X.loc[:, jj] = supervised_binning_transform(X[jj], self._dict_bin.get(jj))
        return X.values

    def fit_transform(self, X, y):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {
            "allow_nan": False,
            "requires_fit": True,
            "requires_y": True,
        }

    def check_params(self, X, y):
        """Checks if params are present."""
        if y is None:
            raise ValueError("Expected y or Y as second argument for method")
        if y.__array__().dtype == "object":
            raise TypeError("Expected y as Binary")
