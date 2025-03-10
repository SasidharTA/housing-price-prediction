import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List


def calculate_vif(data: pd.DataFrame, idv_cols: List, dv_col: str) -> pd.DataFrame:
    """Calculate Variance Inflation Factor (VIF) for each independent variable and their correlation with the dependent variable.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame on which VIF and correlation analysis is to be performed.
    idv_cols : List[str]
        List of independent variables on which VIF analysis is to be done.
    dv_col : str
        Dependent variable on which correlation values are to be calculated.

    Returns
    -------
    pd.DataFrame
        DataFrame with VIF and correlation values for each independent variable.
    """

    # Create an empty DataFrame to store the VIF values
    vif_df = pd.DataFrame()

    # Add the names of independent variables to the DataFrame
    vif_df["independent_variable"] = idv_cols

    # Create a subset of the input DataFrame with only the independent variables and add a constant column to it
    data_subset = data[idv_cols]
    data_subset = data_subset.assign(const=1)

    # Create a list of column indices for the independent variables and calculate their VIF values
    var_indices = list(range(data_subset.shape[1]))
    vif_values = [
        variance_inflation_factor(data_subset.iloc[:, var_indices].values, i)
        for i in range(data_subset.iloc[:, var_indices].shape[1])
    ]

    # Remove the VIF value of the constant column and add the VIF values to the DataFrame
    vif_values = vif_values[:-1]
    vif_df["VIF"] = vif_values

    # Calculate the correlation matrix between independent variables and the dependent variable
    corr_mat = data[[*idv_cols, dv_col]].corr()
    corr_values = corr_mat[dv_col]
    corr_values = corr_values.drop(index=dv_col).reset_index()

    # Merge the VIF and correlation DataFrames and return the result
    return (
        vif_df.merge(
            corr_values, how="inner", left_on="independent_variable", right_on="index"
        )
        .drop("index", axis=1)
        .round(2)
    )


def compute_idv_dv_correlation(
    data: pd.DataFrame, idv_cols: List[str], dv_col: List[str]
) -> pd.DataFrame:
    """Compute the correlation between independent variables and dependent variables.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe containing independent and dependent variables.
    idv_cols : List[str]
        A list of independent variables.
    dv_col : List[str]
        A list of dependent variables.

    Returns
    -------
    pd.DataFrame
        A correlation matrix containing the correlation coefficients between the independent and dependent variables.
    """

    # Compute the correlation matrix for the independent and dependent variables
    corr_mat = data[idv_cols + dv_col].corr()

    # Drop the correlations between the dependent variables
    corr_mat = corr_mat[dv_col].drop(index=dv_col)

    return corr_mat
