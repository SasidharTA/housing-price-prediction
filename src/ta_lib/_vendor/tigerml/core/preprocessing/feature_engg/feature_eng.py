"""Supporting module for various binning methods."""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import PowerTransformer
from tigerml.core.preprocessing.scripts.MDLP import MDLP_Discretizer
from tigerml.core.utils.pandas import get_num_cols

type(enable_iterative_imputer)  # this line is added to deal with flake8 error


def feature_engg_power_fit(data: DataFrame, target: Series, method: str, cols=None):
    """Apply power transform for all numerical columns.

    Parameters
    ----------
    data : pd.dataframe
    target : np.array
    method : str,
        box-cox or yeo-johnson

    Returns
    -------
    data: pd.dataframe
        dataframe with transformed columns
    """
    if type(data) != pd.core.frame.DataFrame:
        raise Exception("data must be a valid pandas data frame")

    if method == "box-cox":
        if data[data <= 0].sum().sum() > 0:
            raise Exception("box-cox transformation is only for positive data.")

    num_cols = cols or get_num_cols(data)
    if len(num_cols) < 1:
        raise Exception("No numerical columns present for power transformation.")
    power_dict = {}
    for col in num_cols:
        power_ = PowerTransformer().fit(data[col], target)
        power_dict.update({col: power_})
    return power_dict


def feature_engg_power_transform(data, power_dict):
    """Transform numeric variables using power transform.

    Parameters
    ----------
    data : pd.DataFrame
    power_dict : dict

    Returns
    -------
    data: pd.dataframe
    """
    for col in power_dict.keys():
        data[f"{col}_power_transform"] = power_dict.get(col).transform(data[col])

    return data


def feature_engg_woe_binning(x_train, y_train, col_bin):
    """Bin numeric variables using WOE(weight of evidence) binning method.

    Parameters
    ----------
    x_train : Training dataset
    y_train : Target vairble of training dataset. (Binary/Dichotomous)
    col_bin : Numeric variable to bucket.

    Returns
    -------
    x_train: binned dataframe
    woe_binner: fitted object woe binner
    """
    try:
        from tigerml.core.preprocessing.feature_engg.xverse.transformer import (
            WOE,
        )
    except ImportError:
        raise Exception(
            "Could not import WOE from xverse. Please make sure xverse is installed."
        )
    woe_binned = WOE(treat_missing="mode").fit(x_train[[col_bin]], y_train)
    return woe_binned


def feature_engg_woe_binning_wrapper(data, target, cols=None):
    """Binning numeric variables using WOE(weight of evidence) binning method to fit.

    Parameters
    ----------
    data : Training dataset
    target : Target vairble of training dataset. (Binary/Dichotomous)
    cols : numerical columns to be binned.

    Returns
    -------
    data : binned dataframe
    woe_binner: dict fitted object woe binner
    """
    if type(data) != pd.core.frame.DataFrame:
        raise ValueError("data must be a valid pandas data frame")

    num_cols = cols or get_num_cols(data)

    if len(num_cols) < 1:
        raise ValueError("Expects atleast one numeric column")

    # woe
    woe_binner = {}
    for col in num_cols:
        try:
            woe_binner_ = feature_engg_woe_binning(data, target, col)
            woe_binner.update({col: woe_binner_})
        except Exception as e:
            print(e)
    return woe_binner


def feature_engg_woe_binning_transform(data, woe_binner, existing):
    """Binning numeric variables using WOE(weight of evidence) binning method to transform.

    Parameters
    ----------
    data : Training dataset
    woe_binner : fitted woe binner for each column

    Returns
    -------
    Binned data
    """
    main_df = data.copy()
    for col in existing:
        data = main_df
        data["WOE"] = woe_binner.get(col).transform(data[[col]])
        df_bin = (
            woe_binner.get(col)
            .woe_df[["WOE", "Category"]]
            .rename(columns={"Category": col + "_woebin"})
        )
        data = data.merge(df_bin, on="WOE").drop("WOE", axis=1)
        main_df[f"{col}_woebin"] = data[f"{col}_woebin"].astype("str")
    # data = data[[i for i in data.columns if '_woebin' in i]]
    return main_df


def unsupervised_binning(
    series: Series,
    n=None,
    size_of_bin=None,
    labels=None,
    quantile=None,
    value=None,
    size=None,
    left=True,
):
    """Bin values in discrete intervals.

    Takes series as input and returns binned series.

    Parameters
    ----------
    series: one-dimensional labeled array capable of holding data of any type
    n: None(default); integer; n number of equal-sized buckets
    size_of_bin: None(default); integer; number of data points in each bucket
    labels: None(default); string array; used as
           label in resulting bins, its length must be same as
           number of resulting bins, if None return only integers indicator of bins
    quantile: None(default); array of quantiles;
              buckets cut based on quantiles provided
    value: None(default); array of numbers(int/float);
           buckets produced based on values provided
    size: None(default); number(integer/float); size of each interval/buckets,
          size = (upper bound-lower bound) of bucket
    left: True(default); boolean; if True buckets'
          size counting will start from left, else from right

    Returns
    -------
    binned: binned series
    bins: bin intervals for each binned series
    """
    if type(series) != pd.core.series.Series:
        raise Exception("data must be a valid pandas series")

    if n:
        pass
    elif size_of_bin:
        pass
    elif quantile:
        pass
    elif value:
        pass
    elif size:
        pass
    else:
        raise Exception(
            "one of the n, size_of_bin, quantile, value, size should be non null"
        )
    # Require numpy and pandas
    # Input is pandas series
    if labels is None:
        labels = False
    if n:
        # n number of bins
        binned, bins = pd.qcut(
            series, n, labels=labels, duplicates="drop", retbins=True
        )
    elif size_of_bin:
        # size of a bin is provided
        # Improvement needed
        n = int(np.ceil(len(series) / size_of_bin))
        binned, bins = pd.qcut(
            series, n, labels=labels, duplicates="drop", retbins=True
        )
    elif quantile:
        # Quantile specified on which intervals has to be created
        binned, bins = pd.qcut(
            series, quantile, labels=labels, duplicates="drop", retbins=True
        )
    elif value:
        # Actual value specified on which intervals has to be created
        binned, bins = pd.cut(series, value, labels=labels, retbins=True)
    elif size:
        # Size of interval is specified
        if left:
            # Left TRUE : intervals will be created starting from left
            # Array initialized with minimum value
            value = [-float("Inf")]
            # First cut point will be one size above the minimum value
            temp = series.min() + size
            while temp < series.max():
                # Iteratively appending cut points with specified size
                value.append(temp)
                temp = temp + size
            # Right side is bounded with highest value
            # Right-most interval will not necessarily have same size
            value.append(float("Inf"))
        else:
            # Left FALSE: intervals will be created starting from right
            # Array initialized with maximum value
            value = [float("Inf")]
            # First cut point will be one size below the maximum value
            temp = series.max() - size
            while temp > series.min():
                # Iteratively appending cut points with specified size
                value.append(temp)
                temp = temp - size
            # Left side is bounded with lowest value
            # Left-most interval will not necessarily have same size
            value.append(-float("Inf"))
            # Value array is created in descending order, it is needed to be reversed
            value.reverse()
        binned, bins = pd.cut(series, value, labels=labels, retbins=True)
    else:
        raise Exception("Input not proper")
    if not (labels):
        binned = binned.astype("float")
    bins = [-np.Inf] + list(bins)[1:-1] + [np.inf]
    bins = pd.DataFrame(bins)
    return binned, bins


def supervised_binning_fit(feature, target, prefix="SPE"):
    """Supervised binning method fit.

    Parameters
    ----------
    feature : str
        Name of the category variable to be encoded.
    prefix : str
        Default is 'SPE'. The prefex will be appended to encoded variable.
        Ex: 'SPE_VariableName_FactorName'

    Returns
    -------
    tuple : (cuts, dataframe)
        Cuts and modified dataframe will be returned.
    """
    discretizer = MDLP_Discretizer(features=np.array([0]))
    discretizer.fit(np.array(feature).reshape(-1, 1), target)
    return discretizer


def supervised_binning_transform(feature, discretizer):
    """Supervise binning method.

    Parameters
    ----------
    feature : str
        Name of the category variable to be encoded.
    prefix : str
        Default is 'SPE'. The prefex will be appended to encoded variable.
        Ex: 'SPE_VariableName_FactorName'

    Returns
    -------
      tuple : (cuts, dataframe)
        Cuts and modified dataframe will be returned.
    """
    binned_feature = discretizer.transform(np.array(feature).reshape(-1, 1))
    return binned_feature
