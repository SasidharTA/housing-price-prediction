# from bunch import Bunch
import logging
import numpy as np
import pandas as pd
import tigerml.core.dataframe as td
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.utils.validation import check_array, check_is_fitted
from tigerml.core.utils import (
    DictObject,
    flatten_list,
    get_cat_cols,
    get_num_cols,
)

from .scripts.Mice_Impute import MiceImputer

_LOGGER = logging.getLogger(__name__)


def _check_X(X, columns=None):
    """Change array to DataFrame."""
    if isinstance(X, (pd.DataFrame, td.DataFrame)):
        return X
    elif isinstance(X, (np.ndarray)):
        if columns is None:
            return pd.DataFrame(X)
        else:
            return pd.DataFrame(X, columns=columns)
    elif isinstance(X, (pd.Series, td.Series)):
        return pd.DataFrame(X, columns=X.name)
    elif isinstance(X, (list, tuple)):
        X = np.array(X)
        return pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    elif hasattr(X, ("__array__")):
        data = X.__array__()
        return pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    return X


class Imputer(_BaseImputer, TransformerMixin):
    """Imputer class.

    Parameters
    ----------
    num_impute_method: str, default median
        numerical imputation method, {'mean', 'median', 'mode'}
    cat_impute_method: str, default mode
        categorical imputation method, {'mode', 'constant'}
    constant_value: str, default missing
        categorical constant to be used when the cat_impute_method is constant

    Attributes
    ----------
    imputed_data: pd.DataFrame
        imputed data frame
    imputation_summary: dict
        imputations summary with no of missing and
        method used for imputation for each column

    Examples
    --------
    >>> import numpy as np
    >>> from tigerml.core.preprocessing.imputer import Imputer
    >>> imp = Imputer()
    >>> imp.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    Imputer()
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp.transform(X))
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]
    """

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    REGRESSION = "regression"

    METHODS = DictObject(
        {
            MEAN: "mean",
            MEDIAN: "median",
            MODE: "mode",
            CONSTANT: "constant",
            REGRESSION: "regression",
        }
    )

    def __init__(
        self,
        missing_values=np.nan,
        add_indicator=False,
        num_impute_method="median",
        cat_impute_method="mode",
        constant_value="missing",
    ):
        if (
            num_impute_method not in self.METHODS
            or cat_impute_method not in self.METHODS
        ):
            raise Exception(
                "Supported imputation methods: {}".format(self.METHODS.keys())
            )
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.num_impute_method = num_impute_method
        self.cat_impute_method = cat_impute_method
        self.constant_value = constant_value

    @property
    def num_cols(self):
        """Returns list of numeric columns."""
        return list(set(get_num_cols(self.data_)) - set(self.drop_cols_))

    @property
    def cat_cols(self):
        """Returns list of categorical columns."""
        return list(set(get_cat_cols(self.data_)) - set(self.drop_cols_))

    def add_imputation_rule(
        self,
        cols,
        impute_method,
        constant_value="missing",
        ind_cols=[],
        ind_impute_method=None,
    ):
        """Adds imputation rule."""
        cols_already_applied = [
            col
            for col in cols
            if any([col in rule["cols"] for rule in self.imputation_rules_])
        ]
        if cols_already_applied:
            raise Exception(
                "Imputation rule already applied " "for {}".format(cols_already_applied)
            )
        if [col for col in cols if col not in self.data_]:
            raise Exception(
                "Columns not present in data - {}".format(
                    [col for col in cols if col not in self.data_]
                )
            )
        if impute_method not in self.METHODS:
            raise Exception(
                "Supported imputation methods: {}".format(self.METHODS.keys())
            )
        self.imputation_rules_.append(
            {
                "cols": cols,
                "impute_method": impute_method,
                "ind_cols": ind_cols,
                "ind_impute_method": ind_impute_method,
                "constant_value": constant_value,
            }
        )
        return self

    def fit(self, data, y=None):
        """Fits imputer."""
        cols = None
        if hasattr(data, "columns"):
            cols = data.columns.tolist()
        if isinstance(data, pd.DataFrame) or isinstance(data, td.DataFrame):
            data_without_datetime = data.select_dtypes(exclude=["datetime64"])
        else:
            data_without_datetime = data
        check_array(
            data_without_datetime,
            accept_large_sparse=False,
            dtype=object,
            force_all_finite="allow-nan",
            copy=True,
        )
        data = _check_X(data, cols)
        self.features_ = data.columns.tolist()
        self.n_features_in_ = len(self.features_)
        self.drop_cols_ = []
        self.imputation_rules_ = []
        self.imputation_summary_ = {}
        self.imputer_ = {}
        self.imputed_data_ = data
        self.data_ = data
        imputed_cols = []
        from tigerml.core.utils import compute_if_dask

        for col in self.imputed_data_.columns:
            if compute_if_dask(
                self.imputed_data_[col].isna().sum() == len(self.imputed_data_)
            ):
                self.imputed_data_ = self.imputed_data_.drop(col, axis=1)
                self.drop_cols_.append(col)
                _LOGGER.info(
                    "All the values of {} are missing. "
                    "Dropping the column.".format(col)
                )
                self.imputation_summary_.update(
                    {
                        col: {
                            "no_of_missing": "All values",
                            "method": "Dropped the column",
                        }
                    }
                )

        if len(self.imputation_rules_) > 0:
            for method in self.imputation_rules_:
                if method["impute_method"] == "regression":
                    if not method["ind_impute_method"] is None:
                        impute_method = method["ind_impute_method"]
                    else:
                        impute_method = self.num_impute_method
                    if len(method["ind_cols"]) > 0:
                        df = pd.DataFrame(self.data_[method["ind_cols"]])
                    else:
                        df = self.data_[self.num_cols]
                    for col in method["cols"]:
                        self.imputer_[col] = self.impute_by_regression(
                            self.data_[col], df, impute_method
                        )
                else:
                    impute_method = method["impute_method"]
                    cols = method["cols"]
                    if all([col in self.num_cols for col in cols]):
                        for col in cols:
                            self.imputer_[col] = self.impute_num(
                                self.data_[col], impute_method=impute_method
                            )
                    elif all([col in self.cat_cols for col in cols]):
                        for col in cols:
                            self.imputer_[col] = self.impute_cat(
                                self.data_[col],
                                impute_method=impute_method,
                                missing_str=method["constant_value"],
                            )
                    elif impute_method == "mode":
                        for col in cols:
                            if col in self.num_cols:
                                self.imputer_[col] = self.impute_num(
                                    self.data_[col], impute_method=impute_method
                                )
                            else:
                                self.imputer_[col] = self.impute_cat(
                                    self.data_[col],
                                    impute_method=impute_method,
                                    missing_str=method["constant_value"],
                                )
                    else:
                        raise Exception(
                            "numeric and categorical columns cannot "
                            "be imputed together with {}".format(impute_method)
                        )

        imputed_cols += list(
            set(flatten_list([rule["cols"] for rule in self.imputation_rules_]))
        )
        for col in set(self.num_cols) - set(imputed_cols):
            self.imputation_rules_.append(
                {
                    "cols": [col],
                    "impute_method": self.num_impute_method,
                    "ind_cols": None,
                    "ind_impute_method": None,
                    "constant_value": None,
                }
            )
            self.imputer_[col] = self.impute_num(
                self.data_[col], impute_method=self.num_impute_method
            )
        for col in set(self.cat_cols) - set(imputed_cols):
            self.imputation_rules_.append(
                {
                    "cols": [col],
                    "impute_method": self.cat_impute_method,
                    "ind_cols": None,
                    "ind_impute_method": None,
                    "constant_value": self.constant_value,
                }
            )
            self.imputer_[col] = self.impute_cat(
                self.data_[col],
                impute_method=self.cat_impute_method,
                missing_str=self.constant_value,
            )
        self.is_fitted_ = True

        non_imputed_cols = set(self.imputed_data_.columns) - set(self.imputer_.keys())
        if len(non_imputed_cols) > 0:
            for col in non_imputed_cols:
                no_of_missing = self.imputed_data_[col].isnull().sum().sum()
                _LOGGER.info(
                    "Column '{}' is skipped for imputation as its "
                    "not recognised as either numerical or categorical "
                    "data type.".format(col)
                )
                self.imputation_summary_.update(
                    {
                        col: {
                            "no_of_missing": no_of_missing,
                            "imputed with": "Not Imputed",
                        }
                    }
                )
        return self

    def transform(self, data):
        """Returns imputed data values."""
        cols = None
        if hasattr(data, "columns"):
            cols = data.columns.tolist()
        check_is_fitted(self, "is_fitted_")
        if isinstance(data, pd.DataFrame) or isinstance(data, td.DataFrame):
            data_without_datetime = data.select_dtypes(exclude=["datetime64"])
        else:
            data_without_datetime = data
        check_array(
            data_without_datetime,
            accept_large_sparse=False,
            dtype=object,
            force_all_finite="allow-nan",
        )
        data = _check_X(data, cols)
        if data.shape[1] != self.data_.shape[1]:
            raise ValueError(
                "The number of features {} in transform is different "
                "from the number of features {} in fit.".format(
                    data.shape[1], self.data_.shape[1]
                )
            )
        data_ = data.copy()
        imputed_data_ = data.copy()
        cols = list(self.imputer_.keys())
        for col in cols:
            if col not in self.drop_cols_:
                no_of_missing = data_[col].isnull().sum().sum()
                impute_method = [
                    rule for rule in self.imputation_rules_ if col in rule["cols"]
                ][0]["impute_method"]
                imputed_data_[col] = self.imputer_[col].transform(
                    data_[col].to_numpy().reshape(-1, 1)
                )
                self.imputation_summary_.update(
                    {
                        col: {
                            "no_of_missing": no_of_missing,
                            "imputed with": impute_method,
                        }
                    }
                )
        return imputed_data_.values

    def _more_tags(self):
        """Returns tag to allow nan for string and 2d array dtypes."""
        return {"allow_nan": True, "X_types": ["2darray", "string"]}

    def get_imputed_data(self):
        """Returns imputed data."""
        return self.imputed_data_

    def get_feature_names(self):
        """Returns list of feature names."""
        return self.features_

    def get_imputation_method(self, col_name):
        """Returns imputation method."""
        if [rule for rule in self.imputation_rules_ if col_name in rule["cols"]]:
            return [
                rule for rule in self.imputation_rules_ if col_name in rule["cols"]
            ][0]["impute_method"]
        else:
            return None

    @staticmethod
    def impute_num(feature, impute_method="mean", constant_value=None):
        """Imputing numerical data.

        This method will impute the missing values in a continuous variable.
        This transformation will impute in inplace.

        Parameters
        ----------
          feature : str
            Name of the numeric variable to be encoded.
          impute_method : int
            Default is 'mean'. Other options include - 'mode' and 'median'.df[]

        Returns
        -------
           fitted object : SimpleImputer fitted python object
        """
        if isinstance(feature, (pd.DataFrame, pd.Series, td.DataFrame, td.Series)):
            feature = feature.to_numpy()
        elif isinstance(feature, (np.ndarray)):
            pass
        else:
            raise ValueError("Pass either numpy or pd or td data")
        if impute_method == "median":
            imp = SimpleImputer(strategy="median")
            imp.fit(feature.reshape(-1, 1))
        elif impute_method == "mode":
            imp = SimpleImputer(strategy="most_frequent")
            imp.fit(feature.reshape(-1, 1))
        elif impute_method == "mean":
            imp = SimpleImputer(strategy="mean")
            imp.fit(feature.reshape(-1, 1))
        else:
            raise Exception(
                'Supported values for "impute_method" are - mean, median, mode'
            )
        return imp

    @staticmethod
    def impute_cat(feature, impute_method="constant", missing_str="missing"):
        """Impute Categorical data.

        This method will impute the missing values in a categorical variable.
        This transformation will impute in inplace.

        Parameters
        ----------
          feature : str
            Name of the category variable to be encoded.
          impute_method : str
            Default is 'constant'. Other options include - 'mode'.
          missing_str : str
            Default is 'missing'. Value that you
            want impute in the missing values.

        Returns
        -------
           fitted object : SimpleImputer fitted python object
        """
        assert str(feature.dtype) == "category" or not np.issubdtype(
            feature.dtype, np.number
        ), (
            "feature is not Categorical, imputeMissingValCat "
            "can impute only Categorical variables"
        )

        if isinstance(feature, (pd.DataFrame, pd.Series, td.DataFrame, td.Series)):
            feature = feature.to_numpy()
        elif isinstance(feature, (np.ndarray)):
            pass
        else:
            raise ValueError("Pass either numpy or pd or td data")

        if impute_method == "constant":
            imp = SimpleImputer(strategy="constant", fill_value=missing_str)
            imp.fit(feature.reshape(-1, 1))
        elif impute_method == "mode":
            imp = SimpleImputer(strategy="most_frequent")
            imp.fit(feature.reshape(-1, 1))
        else:
            raise Exception('Incorrect input. Should be either "constant" or "mode".')
        return imp

    @staticmethod
    def impute_by_regression(target, df, impute_method="mean"):
        """Impute by regression.

        This method will impute the missing values of the target variable
        by fitting linear/logistic equation.
        Unlike MICE, this method will create single equation for the
        target variable.
        This equations are either linear or logistic equations.

        Parameters
        ----------
          target : str
            variable for which imputation has to be performed.
          group : list
            independent variables with which missing values have to be estimated.
          impute_method : str
            Default is 'Mean'. Strategy on how to impute missing
            values in independent variables of the equation.

        Return
        ------
           fitted object : SimpleImputer fitted python object
        """
        if target.name in df.columns:
            df = df[~target.name]
        reg_imp = MiceImputer(seed_strategy=impute_method, target=target.name, group=[])
        reg_imp.fit(pd.concat([df, target], axis=0))
        return reg_imp
