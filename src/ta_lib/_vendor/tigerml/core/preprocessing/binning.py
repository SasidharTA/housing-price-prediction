import numpy as np
import pandas as pd

from .scripts.MDLP import MDLP_Discretizer


def supervised_binning(feature, target, prefix="SPE"):
    """Supervised binning.

    This method finds optimal bins for the given feature such that the
    information gain is maximized w.r.t to target variable. This
    transformation can be used only for classification task.

    Parameters
    ----------
      feature : str
        Name of the category variable to be encoded.
      prefix : str
        Default is 'SPE'. The prefex will be appended to
        encoded variable. Ex: 'SPE_VariableName_FactorName'
    Returns
    -------
      tuple : (cuts, dataframe)
        Cuts and modified dataframe will be returned.

    """
    binned_feature_name = prefix + "_" + feature
    discretizer = MDLP_Discretizer(features=np.array([0]))
    discretizer.fit(feature, target)
    binned_feature = discretizer.transform(np.array(feature).reshape(-1, 1))
    binned_feature.rename(binned_feature_name)
    return binned_feature


def binning(
    series,
    n=None,
    size_of_bin=None,
    labels=None,
    quantile=None,
    value=None,
    size=None,
    left=True,
    target=None,
):
    """Function to bin values in discrete intervals.

    Takes series as input and returns binned series.

    Parameters
    ----------
    n           None(default); integer;
                    n number of equal-sized buckets
    labels      None(default); string array;
                    Used as label in resulting bins, its length must
                    be same as number of resulting bins, if None return
                    only integers indicator of bins
    quantile    None(default); array of quantiles;
                    Buckets cut based on quantiles provided
    value       None(default); array of numbers(int/float);
                    Buckets produced based on values provided
    size        None(default); number(integer/float);
                    Size of each interval/buckets,
                    size = (upper bound-lower bound) of bucket
    left        True(default); boolean;
                    If True buckets' size counting will start from
                    left, else from right
    size_of_bin  None(default); integer;
                    Number of data points in each bucket
    target      Target variable for supervised binning

    """
    # FIXME: Handle Key Error Runtime Exception
    try:
        if n:
            # n number of bins
            binned = pd.qcut(series, n, labels=labels)
        elif size_of_bin:
            # size of a bin is provided
            # ################################# Improvement needed
            n = int(np.ceil(len(series) / size_of_bin))
            binned = pd.qcut(series, n, labels=labels)
        elif quantile:
            # Quantile specified on which intervals has to be created
            binned = pd.qcut(series, quantile, labels=labels)
        elif value:
            # Actual value specified on which intervals has to be created
            binned = pd.cut(series, value, labels=labels)
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
                # Value array is created in descending order, it is needed
                # to be reversed
                value.reverse()
            binned = pd.cut(series, value, labels=labels)
        elif target:
            assert isinstance(target, pd.Series), "target should be a pandas Series"
            assert not np.issubdtype(target.dtype, np.number), (
                "target is a number and not Categorical, "
                "supervised binning can only be applied for classification tasks"
            )
            assert len(series) == len(
                target
            ), "input series and target should be of the same length"
            binned = supervised_binning(series, target)
        else:
            print("Input not proper")
            return None
        return binned
    except ValueError as e:
        print(e)


def grouping():
    pass
