import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO)


def _get_outlier_IQR_bounds(data: pd.DataFrame, columns: List) -> pd.DataFrame:
    """Calculate the lower and upper bounds of columns using the Interquartile Range (IQR) method.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    columns : List[str]
        The list of column names on which outlier detection should be performed.

    Returns
    -------
    pd.DataFrame
        DataFrame with bounds of all required columns.
    """
    bounds = data[columns].quantile([0.25, 0.75])
    IQR = (
        bounds.diff()
        .reset_index()
        .drop("index", axis=1)
        .drop(0, axis=0)
        .rename(index={1: "IQR"})
    )
    bounds = pd.concat([bounds, IQR]).T

    bounds["lower_bound"] = bounds[0.25] - (1.5 * bounds["IQR"])
    bounds["upper_bound"] = bounds[0.75] + (1.5 * bounds["IQR"])

    return bounds.T


def _get_outlier_ZSCORE_bounds(data: pd.DataFrame, columns: List) -> pd.DataFrame:
    """Calculate the lower and upper bounds of columns using the Z-score method.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    columns : List[str]
        The list of column names on which outlier detection should be performed.

    Returns
    -------
    pd.DataFrame
        DataFrame with bounds of all required columns.
    """

    bounds = data[columns].describe().loc[["mean", "std"], :].T

    bounds["lower_bound"] = bounds["mean"] - 3 * bounds["std"]
    bounds["upper_bound"] = bounds["mean"] + 3 * bounds["std"]

    return bounds.T


def _cap_outliers(
    df_treat: pd.DataFrame,
    columns: List,
    outlier_direction: int,
    bounds: pd.DataFrame,
) -> pd.DataFrame:
    """Cap outliers in the input DataFrame by applying capping treatment.

    Parameters
    ----------
    df_treat : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        The list of column names on which outlier detection should be performed.
    outlier_direction : int
        The direction in which outlier treatment should be performed.
        Acceptable inputs are 1, 0, and -1.
        If 0, performs outlier detection for both upper and lower bounds.
        If 1, performs outlier detection for the upper bound only.
        If -1, performs outlier detection for the lower bound only.
    bounds : pd.DataFrame
        DataFrame with bounds of all columns.

    Raises
    ------
    ValueError
        If an invalid outlier detection direction is provided. Available options are 1, 0, -1.

    Returns
    -------
    pd.DataFrame
        DataFrame with capped outlier samples.
    """
    for col in columns:
        if outlier_direction == 0:
            df_treat[col] = np.where(
                df_treat[col] > bounds.loc["upper_bound", col],
                bounds.loc["upper_bound", col],
                np.where(
                    df_treat[col] < bounds.loc["lower_bound", col],
                    bounds.loc["lower_bound", col],
                    df_treat[col],
                ),
            )
        elif outlier_direction == 1:
            df_treat[col] = np.where(
                df_treat[col] > bounds.loc["upper_bound", col],
                bounds.loc["upper_bound", col],
                df_treat[col],
            )
        elif outlier_direction == -1:
            df_treat[col] = np.where(
                df_treat[col] < bounds.loc["lower_bound", col],
                bounds.loc["lower_bound", col],
                df_treat[col],
            )
        else:
            raise ValueError(
                "Invalid outlier detection direction . Available options are 1,0,-1 ."
            )

    return df_treat


def _identify_outliers(
    data: pd.DataFrame,
    columns: List,
    outlier_direction: int,
    bounds: pd.DataFrame,
    bool_val: bool,
) -> pd.DataFrame:
    """Identify outliers in the input DataFrame based on the provided bounds.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    columns : List
        The list of columns on which outlier detection should be performed.
    outlier_direction : int
        The direction in which outlier treatment should be performed.
        Acceptable inputs are 1, 0, and -1.
        If 0, performs outlier detection for both upper and lower bounds.
        If 1, performs outlier detection for the upper bound only.
        If -1, performs outlier detection for the lower bound only.
    bounds : pd.DataFrame
        DataFrame with bounds of all columns.
    bool_val : bool
        Pass False if resulting data needed is outlier-free data, else pass True.
        Acceptable inputs are True or False.

    Raises
    ------
    ValueError
        If an invalid outlier detection direction is provided. Available options are 1, 0, -1.

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier samples.
    """

    if outlier_direction == 0:
        outliers = data[columns].apply(
            lambda x: (x < bounds.loc["lower_bound", x.name])
            | (x > bounds.loc["upper_bound", x.name])
        )
    elif outlier_direction == 1:
        outliers = data[columns].apply(
            lambda x: (x > bounds.loc["upper_bound", x.name])
        )
    elif outlier_direction == -1:
        outliers = data[columns].apply(
            lambda x: (x < bounds.loc["lower_bound", x.name])
        )
    else:
        raise ValueError(
            "Invalid outlier detection direction. Available options are 1, 0, -1."
        )

    outliers = outliers.any(axis=1)

    outliers_rem = data.copy()
    outliers_rem["_BOOL"] = outliers
    out = outliers_rem[outliers_rem["_BOOL"] == bool_val].drop(["_BOOL"], axis=1)
    return out


def outlier_treatment(
    data: pd.DataFrame,
    columns: List,
    outlier_direction: int = 0,
    method: str = "IQR",
    treatment: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform outlier detection or treatment on a dataframe by removing or capping the outliers.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe.
    columns : List
        The columns on which outlier treatment is to be performed.
    outlier_direction : int, optional
        The direction in which outlier treatment is to be performed.

        - 0: Performs outlier treatment for both upper and lower bounds.
        - 1: Performs outlier treatment for the upper bound only.
        - -1: Performs outlier treatment for the lower bound only.

        Defaults to 0.
    method : str, optional
        The method used for outlier detection.
        Acceptable inputs are "IQR" and "ZSCORE".
        Defaults to "IQR".
    treatment : str, optional
        The outlier treatment method. Available options are:

        - None: No treatment is applied. The function returns a dataframe with outliers.
        - "remove": Removes the outliers from the dataframe.
        - "cap": Caps the outliers within the specified bounds.

        Defaults to None.

    Returns
    -------
    pd.DataFrame
        The dataframe with outliers or the dataframe after outlier treatment, depending on the combination of parameters.

    Examples
    --------
    >>> import outlier_treatment
    >>> import pandas as pd

    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 500], 'C': [100, 200, 300, 400, 500]})
    >>> columns = ['A', 'B']

    >>> # Perform outlier detection and removal
    >>> result = outlier_treatment(data, columns, outlier_direction=0, method="IQR", treatment="remove")

    >>> print(result)
    """

    if method == "IQR":
        bounds = _get_outlier_IQR_bounds(data, columns)

    elif method == "ZSCORE":
        bounds = _get_outlier_ZSCORE_bounds(data, columns)

    else:
        raise ValueError(
            "Invalid Outlier detection method. Acceptable inputs are IQR or ZSCORE."
        )

    if treatment is None:
        outliers = _identify_outliers(data, columns, outlier_direction, bounds, True)

    elif treatment == "remove":
        outliers = _identify_outliers(data, columns, outlier_direction, bounds, False)

    elif treatment == "cap":
        outliers = _cap_outliers(data.copy(), columns, outlier_direction, bounds)

    else:
        raise ValueError(
            "Invalid outlier treatment method. Available options are 'remove', 'cap', or None."
        )

    return outliers


def _check_impute_method(method, numerical_cols):
    if isinstance(method, str):
        method = {method: numerical_cols}
    elif not isinstance(method, dict):
        raise ValueError(
            "Invalid imputation method. The method parameter should be a string or a dictionary."
        )
    return method


def missing_value_impute(
    data: pd.DataFrame,
    method: Union[str, Dict[str, list]],
    value: Union[int, float, dict, pd.Series, pd.DataFrame] = 0,
) -> pd.DataFrame:
    """
    Replace missing values in the DataFrame based on the specified imputation strategies.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    method : Union[str, Dict[str, list]]
        The imputation method or a dictionary mapping imputation methods to the corresponding columns.
        If a single method is provided, it will be applied to all numerical columns in the DataFrame.
        If a dictionary is provided, each key represents an imputation method,
        and the corresponding value is a list of column names to which the method should be applied.
        Acceptable inputs for method are: "mean", "median", "most_frequent", "constant", "ffill", "bfill", and "custom".
    value : Union[int, float, dict, pd.Series, pd.DataFrame], optional
        Value to use to fill holes (e.g. 0) when the imputation strategy is "constant."
        If a dictionary is provided, the keys should be column names, and the values should be the imputation values for each column.
        Values not in the dictionary will not be filled. This value cannot be a list.
        Default value is 0.

    Returns
    -------
    pd.DataFrame
        The DataFrame with imputed values.

    Raises
    ------
    ValueError
        If an invalid imputation method is provided.
        If the method parameter is neither a string nor a dictionary.

    Examples
    --------
    ### Example 1: Apply Different Imputation Methods to Different Columns
    >>> import pandas as pd

    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({'A': [1, None, 3, 4, None], 'B': [10, 20, None, 40, 50], 'C': [200, None, 300, None, 500], 'D': ['X', 'Y', None, 'Z', 'X']})
    >>> method = {
    ...     "mean": ['A', 'B'],
    ...     "ffill": ['C'],
    ...     "most_frequent": ['D']
    ... }

    >>> # Perform missing value imputation
    >>> result = missing_value_impute(data, method)

    >>> print(result)


    ### Example 2: Apply Mean Imputation to All Numerical Columns
    >>> import pandas as pd

    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({'A': [1, 2, None, 4, 5], 'B': [10, 20, None, 40, 50], 'C': [None, 200, 300, None, 500]})
    >>> method = "mean"  # Apply mean imputation to all numerical columns

    >>> # Perform missing value imputation with mean
    >>> result = missing_value_impute(data, method)

    >>> print(result)


    ### Example 3: Apply Constant Imputation to All Numerical Columns with a Dictionary Value

    >>> import pandas as pd

    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({'A': [1, 2, None, 4, 5], 'B': [10, 20, None, 40, 50], 'C': [None, 200, 300, None, 500]})
    >>> method = "constant"  # Apply mean imputation to all numerical columns
    >>> value = {'A': 100, 'B': 200}  # Custom imputation values for columns A and B

    >>> # Perform missing value imputation with a dictionary value
    >>> result = missing_value_impute(data, method, value=value)

    >>> print(result)


    ### Example 4: Apply Constant Imputation to Specific Columns with a Series Value
    >>> import pandas as pd

    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({'A': [1, None, 3, 4, None], 'B': [10, 20, None, 40, 50], 'C': [None, 200, 300, None, 500]})
    >>> method = {"custom": ['A'],"constant":['B','C']}  # Apply custom imputation to columns A and B
    >>> value = pd.Series([10, 20, 30, 40, 50])  # Custom imputation values as a Series

    >>> # Perform missing value imputation with a Series value
    >>> result = missing_value_impute(data, method, value=value)

    >>> print(result)


    ### Example 5: Apply Constant Imputation to All Columns with a DataFrame Value
    >>> import pandas as pd

    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({'A': [1, None, 3, 4, None], 'B': [10, 20, None, 40, 50], 'C': [None, 200, 300, None, 500]})
    >>> method = "constant"  # Apply constant imputation to all columns
    >>> value = pd.DataFrame({'A': [10, 20, 30, 40, 50], 'B': [100, 200, 300, 400, 500], 'C': [1000, 2000, 3000, 4000, 5000]})  # Custom imputation values as a DataFrame

    >>> # Perform missing value imputation with a DataFrame value
    >>> result = missing_value_impute(data, method, value=value)

    >>> print(result)
    """

    df = data.copy()
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    method = _check_impute_method(method, numerical_cols)
    method_list = [
        "mean",
        "median",
        "most_frequent",
        "constant",
        "ffill",
        "bfill",
        "custom",
    ]

    for strategy, cols_to_impute in method.items():
        if strategy not in method_list:
            raise ValueError(
                "Invalid imputation method. Available methods are {}".format(
                    method_list
                )
            )
        if strategy in ["ffill", "bfill"]:
            df[cols_to_impute] = df[cols_to_impute].fillna(method=strategy)
        elif strategy == "custom":
            df[cols_to_impute] = df[cols_to_impute].interpolate(
                axis=0, limit_direction="both"
            )
        elif strategy == "constant":
            df[cols_to_impute] = df[cols_to_impute].fillna(value=value)
        else:
            imputer = SimpleImputer(strategy=strategy)
            df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    return df
