import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2_contingency, pearsonr
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.feature_selection._mutual_info import _compute_mi_cd
from sklearn.metrics.cluster import mutual_info_score
from sklearn.preprocessing import scale
from sklearn.utils import check_array, check_random_state
from tigerml.core.utils.pandas import get_bool_cols, get_cat_cols, get_num_cols
from tigerml.core.utils.stats import correlation_ratio, woe_info_value


def _check_X(X):
    """Change array to DataFrame."""
    check_array(X, dtype=None)
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X)
    elif isinstance(X, pd.Series):
        return pd.DataFrame(X, columns=X.name)
    return X


def _check_y(y):
    """Determine the type of target variable.

    Parameters
    ----------
    y: {np.array, pd.Series or pd.Dataframe} of shape (n_samples, 1).
        Target series.

    Returns
    -------
    y_type: str
        Datatype of the target variable.
    """
    if isinstance(y, np.ndarray):
        target_df = pd.DataFrame({"target": y})
    elif isinstance(y, pd.Series):
        target_df = pd.DataFrame({"target": y})
    elif isinstance(y, pd.DataFrame):
        target_df = y
    else:
        raise ValueError(
            "Target variable should be one of np.array. pd.Series or pd.DataFrame"
        )

    if len(get_num_cols(target_df)) >= 1:
        if target_df["target"].nunique() == 2:
            y_type = "boolean"
        else:
            y_type = "continuous"
    elif len(get_bool_cols(target_df)) >= 1:
        y_type = "boolean"
    elif len(get_cat_cols(target_df)) >= 1:
        y_type = "categorical"
    else:
        raise ValueError(
            "Unsupported y datatype. Unable to detect Y type "
            "to be continuous or boolean or categorical"
        )

    return y_type


def _validate_params(y_type, statistic, cat_cols=[], bool_cols=[], num_cols=[]):
    """Validate inputs."""
    cat_len = len(cat_cols + bool_cols)

    if statistic == "corr_coef":
        if y_type != "continuous":
            raise ValueError("y must be numeric")
        if len(num_cols) == 0:
            raise ValueError("At least one X column must be numeric")
    if statistic in ["chi_square", "cramer_v"]:
        if y_type == "continuous":
            raise ValueError("y must be boolean/categorical")
        if cat_len == 0:
            raise ValueError("X must have at least one numeric column")
    if statistic == "woe_iv":
        if y_type == "continuous":
            raise ValueError("y must be boolean/categorical")
    if statistic in ["corr_ratio"]:
        if y_type == "continuous":
            if cat_len == 0:
                raise ValueError("X must have at least one non-numeric column")
        else:
            if len(num_cols) == 0:
                raise ValueError("X must have at least one numeric column")


def corr_coef(X, y):
    """Compute Pearson Correlation Coefficient for Y vs. X.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors for which correlation coefficient
        is calculated sequentially.
    y : {np.array, pd.Series or pd.Dataframe} of shape (n_samples, 1)

    Returns
    -------
    corr_coef_val : array, shape = (n_features,)
        correlation coefficient statistics of each feature.
    """
    X = _check_X(X)
    num_cols = get_num_cols(X)
    y_type = _check_y(y)
    _validate_params(y_type, "corr_coef", num_cols=num_cols)

    corr_coef_val = []
    for col in X.columns:
        if col in num_cols:
            stat_ = pearsonr(X[col], y)[0]
            corr_coef_val.append(abs(stat_))
        else:
            corr_coef_val.append(np.nan)
    return np.array(corr_coef_val)


def mutual_value(X, y):
    """Compute mutual values for Y vs. X.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will be tested sequentially.
    y : {np.array, pd.Series or pd.Dataframe} of shape (n_samples, 1)

    Returns
    -------
    mutual_info : array, shape = (n_features,)
        mutual info statistics of each feature.
    """
    X = _check_X(X)
    cat_cols = get_cat_cols(X)
    bool_cols = get_bool_cols(X)
    num_cols = get_num_cols(X)
    y_type = _check_y(y)
    _validate_params(
        y_type,
        "mutual_value",
        cat_cols=cat_cols,
        bool_cols=bool_cols,
        num_cols=num_cols,
    )
    random_state = 42
    if y_type == "continuous":
        num_col_scores = {}
        if num_cols:
            num_col_scores = dict(
                zip(
                    num_cols,
                    mutual_info_regression(
                        X=X[num_cols], y=y, random_state=random_state
                    ),
                )
            )
        cat_col_scores = {}
        rng = check_random_state(random_state)
        scaled_y = scale(y, with_mean=False)
        scaled_y += (
            1e-10 * np.maximum(1, np.mean(np.abs(scaled_y))) * rng.randn(X.shape[0])
        )
        for col in cat_cols + bool_cols:
            cat_col_scores[col] = _compute_mi_cd(
                c=scaled_y, d=X[col].values, n_neighbors=3
            )
        all_col_scores = {**num_col_scores, **cat_col_scores}
        mutual_info = np.array([all_col_scores[col] for col in X.columns])
    elif y_type in ["categorical", "boolean"]:
        num_col_scores = {}
        if num_cols:
            num_col_scores = dict(
                zip(
                    num_cols,
                    mutual_info_classif(X=X[num_cols], y=y, random_state=random_state),
                )
            )
        cat_col_scores = {}
        for col in cat_cols + bool_cols:
            cat_col_scores[col] = mutual_info_score(X[col].values, y)
        all_col_scores = {**num_col_scores, **cat_col_scores}
        mutual_info = np.array([all_col_scores[col] for col in X.columns])
    return mutual_info


def woe_iv(X, y, bin_type="cut", nbins=10):
    """Compute IV (Information Value) from WOE (Weight Of Evidence) for Y vs. X.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will be tested sequentially.
    y : {np.array, pd.Series or pd.Dataframe} of shape (n_samples, 1)
    bin_type : {"cut", "qcut"} str (default = "cut")
    nbins : No. of bins required, while discretizing numeric columns (default = 10).

    Returns
    -------
    woe_iv : array, shape = (n_features,)
        f_score statistics of each feature.
    """
    X = _check_X(X)
    cat_cols = get_cat_cols(X)
    bool_cols = get_bool_cols(X)
    num_cols = get_num_cols(X)
    y_type = _check_y(y)
    y = pd.Series(y)
    _validate_params(
        y_type, "woe_iv", cat_cols=cat_cols, bool_cols=bool_cols, num_cols=num_cols
    )

    woe_iv_val = []
    for col in X.columns:
        # We can use the woe for numerical columns by using the following split
        # if (col in num_cols) & (X[col].nunique() > 5):
        #     X[col] = pd.qcut(X[col], 5, duplicates="drop")
        if col in bool_cols + cat_cols:
            stat_ = woe_info_value(target_series=y, idv_series=X[col])
        else:
            if len(X[col].unique()) > nbins:
                if bin_type == "cut":
                    binned_col = pd.cut(X[col], nbins).astype(str)
                else:
                    binned_col = pd.qcut(X[col], nbins).astype(str)
            else:
                binned_col = X[col].astype(str)
            stat_ = woe_info_value(target_series=y, idv_series=binned_col)
        woe_iv_val.append(stat_)
    return np.array(woe_iv_val)


def f_score(X, y):
    """Compute f_scores statistic for Y vs. X.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will be tested sequentially.
    y : {np.array, pd.Series or pd.Dataframe} of shape (n_samples, 1)

    Returns
    -------
    f_score : array, shape = (n_features,)
        f_score statistics of each feature.
    pval : array, shape = (n_features,)
        p-values of each feature.
    """
    X = _check_X(X)
    cat_cols = get_cat_cols(X)
    bool_cols = get_bool_cols(X)
    y_type = _check_y(y)
    _validate_params(y_type, "f_score", cat_cols=cat_cols, bool_cols=bool_cols)

    if y_type == "continuous":
        f_score, p_value = f_regression(X=X, y=pd.Series(y))
        return f_score, p_value
    elif y_type in ["boolean", "categorical"]:
        f_score, p_value = f_classif(X=X, y=pd.Series(y))
        return f_score, p_value


def chi_square(X, y):
    """Compute Chi-Square statistic for Y vs. X.

    Compute chi-squared stats between each non-negative feature and class.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will be tested sequentially.
    y : {np.array, pd.Series or pd.Dataframe} of shape (n_samples, 1)

    Returns
    -------
    chi2 : array, shape = (n_features,)
        chi2 statistics of each feature.
    pval : array, shape = (n_features,)
        p-values of each feature.
    """
    X = _check_X(X)
    cat_cols = get_cat_cols(X)
    bool_cols = get_bool_cols(X)
    y_type = _check_y(y)
    _validate_params(y_type, "chi_square", cat_cols=cat_cols, bool_cols=bool_cols)
    cols = cat_cols + bool_cols
    chi_val = []
    p_value = []
    for col in X.columns:
        # We can use the chi_square for numerical columns by using the following split
        # if (col in num_cols) & (X[col].nunique() > 5):
        #     X[col] = pd.qcut(X[col], 5, duplicates="drop")
        if col in cols:
            crosstab = pd.crosstab(X[col], pd.Series(y))
            chi2, p, dof, ex = chi2_contingency(crosstab)
            chi_val.append(chi2)
            p_value.append(p)
        else:
            chi_val.append(np.nan)
            p_value.append(np.nan)
    chi_val = np.array(chi_val)
    p_value = np.array(p_value)
    return chi_val, p_value


def cramer_v(X, y):
    """Compute cramer v for Y vs. X.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will be tested sequentially.
    y : {np.array, pd.Series or pd.Dataframe} of shape (n_samples, 1)

    Returns
    -------
    cramer_v : array, shape = (n_features,)
        cramers v statistics of each feature.
    pval : array, shape = (n_features,)
        p-values of each feature.
    """
    X = _check_X(X)
    cat_cols = get_cat_cols(X)
    bool_cols = get_bool_cols(X)
    y_type = _check_y(y)
    _validate_params(y_type, "cramer_v", cat_cols=cat_cols, bool_cols=bool_cols)

    cols = bool_cols + cat_cols
    cramer_val = []
    p_value = []
    for col in X.columns:
        # We can use the cramer_v for numerical columns by using the following split
        # if (col in num_cols) & (X[col].nunique() > 5):
        #     X[col] = pd.qcut(X[col], 5, duplicates="drop")
        if col in cols:
            crosstab = pd.crosstab(X[col], pd.Series(y))
            chi2, p, dof, ex = chi2_contingency(crosstab)
            n = crosstab.sum().sum()
            phi2 = chi2 / n
            r, k = crosstab.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            stat_ = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
            cramer_val.append(stat_)
            p_value.append(p)
        else:
            cramer_val.append(np.nan)
            p_value.append(np.nan)
    cramer_val = np.array(cramer_val)
    p_value = np.array(p_value)
    return cramer_val, p_value


def corr_ratio(X, y):
    """Compute correlation ratio for Y vs. X.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will be tested sequentially.
    y : {np.array, pd.Series or pd.Dataframe} of shape (n_samples, 1)

    Returns
    -------
    corr_ratio : array, shape = (n_features,)
        cramers v statistics of each feature.
    """
    X = _check_X(X)
    cat_cols = get_cat_cols(X)
    bool_cols = get_bool_cols(X)
    num_cols = get_bool_cols(X)
    y_type = _check_y(y)
    _validate_params(
        y_type, "corr_ratio", cat_cols=cat_cols, bool_cols=bool_cols, num_cols=num_cols
    )
    corr_ratio = []
    for col in X.columns:
        if y_type in ["categorical", "boolean"]:
            if col in num_cols:
                stat_ = correlation_ratio(pd.Series(y), X[col])
                corr_ratio.append(abs(stat_))
            else:
                corr_ratio.append(np.nan)
        elif y_type == "continuous":
            cols = cat_cols + bool_cols
            if col in cols:
                stat_ = correlation_ratio(X[col], pd.Series(y))
                corr_ratio.append(abs(stat_))
            else:
                corr_ratio.append(np.nan)
    return np.array(corr_ratio)
