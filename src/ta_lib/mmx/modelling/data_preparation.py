import numpy as np
import pandas as pd
from typing import Tuple


def prepare_train_test_data(
    data: pd.DataFrame,
    train_test_col_name: str,
    date_col: str,
    group_col: str,
    idv_cols: list,
    dv_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.array, pd.DataFrame, np.array]:
    """
    Prepare the train and test data.

    Parameters
    ----------
    data : pd.DataFrame
        Transformed dataset used to create models.
    train_test_col_name : str
        Column name for the train test flag.
    date_col : str
        Column name for the date variable.
    group_col : str
        Column name for the group variable.
    idv_cols : List
        List of columns representing the independent variables.
    dv_col : str
        Column name for the dependent variable.

    Returns
    -------
    Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        np.array,
        pd.DataFrame,
        np.array
    ]:
    Returns train and test data along with date and group dataframes.
    """

    # prepare training and test data
    train_flag = data[train_test_col_name] == "train"
    test_flag = data[train_test_col_name] == "test"

    train_date_level_df = data[train_flag][[date_col, group_col, train_test_col_name]]
    test_date_level_df = data[test_flag][[date_col, group_col, train_test_col_name]]

    X_train = data.filter(idv_cols)[train_flag]
    y_train = data[train_flag][dv_col].values
    X_test = data.filter(idv_cols)[test_flag]
    y_test = data[test_flag][dv_col].values

    return train_date_level_df, test_date_level_df, X_train, y_train, X_test, y_test
