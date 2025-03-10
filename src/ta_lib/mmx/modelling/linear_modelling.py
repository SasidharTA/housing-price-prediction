import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from typing import List

from .data_preparation import prepare_train_test_data
from .evaluation_utils import get_act_vs_preds_df


def _get_coefficient_matrix_linear(
    feature_df: pd.DataFrame, idv_cols: list, group_col: str, group_values: list
) -> pd.DataFrame:
    """
    Generate a coefficient matrix from the coefficients of Lasso and Ridge models.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature coefficient dataframe, having two columns: feature names and feature coefficients.
    idv_cols : list
        List of all the independent variables (IDVs) to include in the coefficient matrix.
    group_col : str
        Name of the column representing groups.
    group_values : list
        List of group values.

    Returns
    -------
    pd.DataFrame
        The output matrix of the coefficients.
    """

    coeffs_list = feature_df[feature_df["feature names"].isin(idv_cols)][
        "feature coefficients"
    ].values

    # Repeat the array elements along the first axis according to the data
    coeffs_matrix = np.tile(coeffs_list, (len(group_values), 1))
    column_names = [
        f"beta_{col}"
        for col in feature_df[feature_df["feature names"].isin(idv_cols)][
            "feature names"
        ].values
    ]

    intercept = feature_df[feature_df["feature names"] == "Intercept"][
        "feature coefficients"
    ].values[0]
    e_intercept = np.exp(intercept)

    output_matrix = pd.DataFrame(coeffs_matrix, columns=column_names)
    output_matrix["Intercept"] = intercept
    output_matrix["e_intercept"] = e_intercept
    output_matrix.insert(0, group_col, group_values)

    return output_matrix


def train_linear_model(
    data: pd.DataFrame,
    idv_cols: List[str],
    dv_col: str,
    train_test_col_name: str,
    date_col: str,
    group_col: str,
    model_type: str = "ridge",
    model_args: dict = {},
):
    """
    Train a linear model using Lasso or Ridge regression.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame containing the independent variables and dependent variable.
    idv_cols : List[str]
        The list of column names to use as independent variables.
    dv_col : str
        The column name of the dependent variable.
    train_test_col_name : str
        The column name that indicates the train-test split.
    date_col : str
        The column name for the date variable.
    group_col : str
        The column name for the group variable (e.g., country, geographic region, etc.).
    model_type : str, optional
        The type of linear model to use: 'ridge' or 'lasso'. Defaults to 'ridge'.
    model_args : dict, optional
        Additional arguments to pass to the linear model constructor. Defaults to an empty dictionary.

    Returns
    -------
    Tuple
        A tuple containing the trained model, feature coefficients, coefficient matrix, and the actuals-vs-predictions DataFrame.

    Raises
    ------
    ValueError
        If an invalid model_type is provided.
    """
    # prepare training and test data
    (
        train_date_level_df,
        test_date_level_df,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepare_train_test_data(
        data=data,
        train_test_col_name=train_test_col_name,
        date_col=date_col,
        group_col=group_col,
        idv_cols=idv_cols,
        dv_col=dv_col,
    )

    # constrain the coefficients to be +ve
    model_args["positive"] = True

    if model_type == "lasso":
        model = Lasso(**model_args)
    elif model_type == "ridge":
        model = Ridge(**model_args)
    else:
        raise ValueError("Please enter either 'ridge' or 'lasso' as the model_type")

    model.fit(X_train, y_train)

    # Get the feature names and coefficients from the model
    feature_names = X_train.columns
    feature_coeffs = model.coef_
    intercept = model.intercept_
    feature_df = pd.DataFrame(
        {"feature names": feature_names, "feature coefficients": feature_coeffs}
    )

    # prepare the coefficient matrix
    feature_df = pd.concat(
        [
            feature_df,
            pd.DataFrame(
                data=[["Intercept", intercept]], columns=feature_df.columns.tolist()
            ),
        ],
        axis=0,
    ).reset_index(drop=True)

    group_values = data[group_col].unique().tolist()
    coeff_matrix = _get_coefficient_matrix_linear(
        feature_df=feature_df,
        idv_cols=idv_cols,
        group_col=group_col,
        group_values=group_values,
    )

    # Make predictions on the training and test data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # get actuals vs predictions dataframe
    actuals_vs_preds_df = get_act_vs_preds_df(
        train_preds,
        test_preds,
        y_train,
        y_test,
        train_date_level_df,
        test_date_level_df,
    )

    return model, feature_df, coeff_matrix, actuals_vs_preds_df
