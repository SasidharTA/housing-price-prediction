import numpy as np
import pandas as pd
from typing import List


def get_additive_attribution(
    data: pd.DataFrame,
    coeff_matrix: pd.DataFrame,
    act_vs_preds: pd.DataFrame,
    date_col: str,
    marketing_vars: List[str],
    control_vars: List[str],
) -> pd.DataFrame:
    """Calculate the additive attribution based on the coefficient matrix and input data.

    Parameters
    ----------
    data : pd.DataFrame
        Data used to train the model.
    coeff_matrix : pd.DataFrame
        Coefficient matrix for the additive attribution calculation.
    act_vs_preds : pd.DataFrame
        Actuals vs Predictions dataframe.
    date_col : str
        Name of the date column in the data.
    marketing_vars : List[str]
        List of marketing variables.
    control_vars : List[str]
        List of control variables.

    Returns
    -------
    pd.DataFrame
        DataFrame with the final contributions for each variable.
    """
    # combining all marketing and control variables
    all_idvs = marketing_vars + control_vars

    # adding a prefix of t_ to column names to indicate they are transformed variables
    contribution_df = data.filter(all_idvs).add_prefix("t_")

    # insert the date column and the group to the contribution df
    contribution_df.insert(0, date_col, data[date_col])
    group_col = coeff_matrix.columns[0]
    contribution_df.insert(1, group_col, data[group_col])

    contribution_df = (
        pd.merge(
            contribution_df,
            coeff_matrix,
            on=group_col,
            how="left",
        )
        .sort_values(by=date_col)
        .reset_index(drop=True)
    )

    beta_matrix = contribution_df.filter([f"beta_{col}" for col in all_idvs]).values
    x_matrix = contribution_df.filter([f"t_{col}" for col in all_idvs]).values

    beta_into_x = np.multiply(beta_matrix, x_matrix)

    column_names = [f"rc_{col}" for col in all_idvs]
    intermediate_df = pd.DataFrame(beta_into_x, columns=column_names)

    contribution_df = pd.concat([contribution_df, intermediate_df], axis=1)

    act_vs_preds_subset = act_vs_preds[[date_col, group_col, "actuals", "preds"]]
    contribution_df[["actuals", "preds"]] = contribution_df.merge(
        act_vs_preds_subset, on=[date_col, group_col], how="left"
    )[["actuals", "preds"]]

    contribution_df["actuals_without_intercept"] = (
        contribution_df["actuals"] - contribution_df["Intercept"]
    )
    contribution_df["preds_without_intercept"] = (
        contribution_df["preds"] - contribution_df["Intercept"]
    )

    for col in all_idvs:
        contribution_df[f"ac_{col}"] = contribution_df[f"rc_{col}"] * (
            contribution_df["actuals_without_intercept"]
            / contribution_df["preds_without_intercept"]
        )

    # filter only actual contributions
    final_contrib_df = contribution_df.filter(
        [date_col, group_col] + ["Intercept"] + [f"ac_{col}" for col in all_idvs]
    ).rename(columns={"Intercept": "base"})

    return final_contrib_df


def calculate_roas(
    data: pd.DataFrame,
    attribution_df: pd.DataFrame,
    time_period: str,
    mapping: pd.DataFrame,
    date_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Calculate the return on ad spent (ROAS) for each marketing variable.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the marketing data.
    attribution_df : pd.DataFrame
        DataFrame containing the attribution data.
    time_period : str
        The time period for grouping the data. Can be either "quarter" or "year".
    mapping : pd.DataFrame
        DataFrame containing the mapping between the model and spend (attribution config file).
    date_col : str
        The name of the date column.
    group_col : str
        The name of the group column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ROAS for each marketing variable.

    Raises
    ------
    ValueError
        If the time_period argument is not "quarter" or "year".
    Exception
        If the length of the raw data and contribution data is not the same.
    """

    # check if time_period is valid
    if time_period not in ["quarter", "year"]:
        raise ValueError("time_period must be 'quarter' or 'year'")

    if data.shape[0] != attribution_df.shape[0]:
        raise Exception("raw data and contribution data shape is not equal.")

    # add "ac_" prefix to mkt_cols
    model_cols = mapping["variable_name"].tolist()
    spend_cols = mapping["variable_spend"].tolist()

    ac_cols = [f"ac_{col}" for col in model_cols]
    # create roas column names
    roas_cols = mapping["variable_description"].tolist()

    # convert date_col to datetime
    data[date_col] = pd.to_datetime(data[date_col])
    attribution_df[date_col] = pd.to_datetime(attribution_df[date_col])

    # calculate total attribution and total spent
    total_attribution = attribution_df.loc[:, ac_cols].sum()
    total_spent = data.loc[:, spend_cols].sum()

    # calculate overall ROAS
    overall_roas = total_attribution.values / total_spent.values

    # create dataframe for overall ROAS
    overall_roas_df = pd.DataFrame(
        data=overall_roas.reshape(1, -1),
        columns=roas_cols,
        index=["Overall"],
    )

    # group data by time_period and sum mkt_cols
    data_agg = (
        data.sort_values(by=[date_col, group_col])
        .assign(
            quarter=lambda x: x[date_col].dt.to_period("Q"),
            year=lambda x: x[date_col].dt.to_period("Y"),
        )
        .groupby(time_period)[spend_cols]
        .sum()
    )

    # group attribution data by time_period and sum ac_cols
    attr_df_agg = (
        attribution_df.sort_values(by=[date_col, group_col])
        .assign(
            quarter=lambda x: x[date_col].dt.to_period("Q"),
            year=lambda x: x[date_col].dt.to_period("Y"),
        )
        .groupby(time_period)[ac_cols]
        .sum()
    )

    # calculate ROAS for each time_period
    roas_df = pd.DataFrame(
        np.divide(attr_df_agg.values, data_agg.values),
        columns=roas_cols,
        index=data_agg.index,
    )

    # concatenate overall ROAS dataframe and ROAS dataframe for each time_period
    final_roas_df = pd.concat([roas_df, overall_roas_df], axis=0)

    return final_roas_df


def calculate_efficiency(
    data: pd.DataFrame,
    attribution_df: pd.DataFrame,
    time_period: str,
    mapping: pd.DataFrame,
    date_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Calculate the efficiency of each marketing variable.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the marketing data.
    attribution_df : pd.DataFrame
        DataFrame containing the attribution data.
    time_period : str
        The time period for grouping the data. Can be either "quarter" or "year".
    mapping : pd.DataFrame
        DataFrame containing the mapping between model and spend.
    date_col : str
        The name of the date column.
    group_col : str
        The name of the group column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the efficiency for each marketing variable.

    Raises
    ------
    ValueError
        If the time_period argument is not "quarter" or "year".
    Exception
        If the length of the raw data and contribution data is not the same.
    """

    # check if time_period is valid
    if time_period not in ["quarter", "year"]:
        raise ValueError("time_period must be 'quarter' or 'year'")

    if data.shape[0] != attribution_df.shape[0]:
        raise Exception("raw data and contribution data shape is not equal.")

    # add "ac_" prefix to mkt_cols
    mkt_cols = mapping["variable_name"].tolist()
    mkt_sp_cols = mapping["variable_spend"].tolist()
    ac_cols = [f"ac_{col}" for col in mkt_cols]
    # create efficiency column names
    effi_cols = mapping["variable_description"].tolist()

    # convert date_col to datetime
    data[date_col] = pd.to_datetime(data[date_col])
    attribution_df[date_col] = pd.to_datetime(attribution_df[date_col])

    # group data by time_period and sum mkt_cols
    data_agg = (
        data.sort_values(by=[date_col, group_col])
        .assign(
            quarter=lambda x: x[date_col].dt.to_period("Q"),
            year=lambda x: x[date_col].dt.to_period("Y"),
        )
        .groupby(time_period)[mkt_sp_cols]
        .sum()
    )

    data_agg = data_agg.div(data_agg.sum(axis=1), axis=0)

    # group attribution data by time_period and sum ac_cols
    attr_df_agg = (
        attribution_df.sort_values(by=[date_col, group_col])
        .assign(
            quarter=lambda x: x[date_col].dt.to_period("Q"),
            year=lambda x: x[date_col].dt.to_period("Y"),
        )
        .groupby(time_period)[ac_cols]
        .sum()
    )

    attr_df_agg = attr_df_agg.div(attr_df_agg.sum(axis=1), axis=0)

    # calculate ROAS for each time_period
    effi_df = pd.DataFrame(
        np.divide(attr_df_agg.values, data_agg.values),
        columns=effi_cols,
        index=data_agg.index,
    )

    return effi_df


def _combine_marketing_and_control_vars(
    transformed_data: pd.DataFrame, marketing_vars: List[str], control_vars: List[str]
) -> pd.DataFrame:
    """
    Combine marketing and control variables with transformed data.

    Parameters
    ----------
    transformed_data : pd.DataFrame
        Transformed data used to train the model.
    marketing_vars : List[str]
        List of marketing variables.
    control_vars : List[str]
        List of control variables.

    Returns
    -------
    pd.DataFrame
        DataFrame with combined marketing and control variables.
    """
    all_idvs = marketing_vars + control_vars
    contribution_df = transformed_data.filter(all_idvs).add_prefix("t_")
    return contribution_df


def _merge_data_with_coefficients(
    contribution_df: pd.DataFrame,
    transformed_data: pd.DataFrame,
    coeff_matrix: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """
    Merge contribution data with coefficient matrix.

    Parameters
    ----------
    contribution_df : pd.DataFrame
        DataFrame with combined marketing and control variables.
    transformed_data : pd.DataFrame
        Transformed data used to train the model.
    coeff_matrix : pd.DataFrame
        DataFrame containing the feature coefficients generated by training a model.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    contribution_df.insert(0, date_col, transformed_data[date_col])
    group_col = coeff_matrix.columns[0]
    contribution_df.insert(1, group_col, transformed_data[group_col])
    return (
        pd.merge(contribution_df, coeff_matrix, on=group_col, how="left")
        .sort_values(by=date_col)
        .reset_index(drop=True)
    )


def _calculate_beta_into_x(
    contribution_df: pd.DataFrame, marketing_vars: List[str], control_vars: List[str]
) -> pd.DataFrame:
    """
    Calculate beta into x.

    Parameters
    ----------
    contribution_df : pd.DataFrame
        Merged DataFrame.
    marketing_vars : List[str]
        List of marketing variables.
    control_vars : List[str]
        List of control variables.

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated values.
    """
    beta_matrix = contribution_df.filter(
        [f"beta_{col}" for col in marketing_vars + control_vars]
    ).values
    x_matrix = contribution_df.filter(
        [f"t_{col}" for col in marketing_vars + control_vars]
    ).values
    e_beta_into_x = np.exp(np.multiply(beta_matrix, x_matrix))
    column_names = [f"e_{col}_into_beta" for col in marketing_vars + control_vars]
    intermediate_df = pd.DataFrame(e_beta_into_x, columns=column_names)
    return pd.concat([contribution_df, intermediate_df], axis=1)


def _calculate_y_values(
    contribution_df: pd.DataFrame,
    act_vs_preds: pd.DataFrame,
    control_vars: List[str],
    date_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Calculate y, y_control, and y_mkt values.

    Parameters
    ----------
    contribution_df : pd.DataFrame
        DataFrame with calculated values.
    act_vs_preds : pd.DataFrame
        DataFrame containing actual vs predicted values.
    control_vars : List[str]
        List of control variables.
    date_col : str
        Name of the date column.
    group_col : str
        Name of the group column.

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated values.
    """
    act_vs_preds_subset = act_vs_preds[[date_col, group_col, "preds", "actuals"]]
    contribution_df[["preds", "actuals"]] = np.exp(
        contribution_df.merge(
            act_vs_preds_subset, on=[date_col, group_col], how="left"
        )[["preds", "actuals"]]
    )
    contribution_df["y_control"] = (
        np.prod(
            contribution_df.filter([f"e_{col}_into_beta" for col in control_vars]),
            axis=1,
        )
        * contribution_df["e_intercept"]
    )
    contribution_df["y_mkt"] = contribution_df["preds"] - contribution_df["y_control"]
    return contribution_df


def _calculate_raw_contributions(
    contribution_df: pd.DataFrame, marketing_vars: List[str], control_vars: List[str]
) -> pd.DataFrame:
    """
    Calculate raw contributions for marketing and control variables.

    Parameters
    ----------
    contribution_df : pd.DataFrame
        DataFrame with calculated values.
    marketing_vars : List[str]
        List of marketing variables.
    control_vars : List[str]
        List of control variables.

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated values.
    """
    for col in control_vars:
        contribution_df[f"rc_{col}"] = contribution_df["y_control"] * (
            1 - 1 / contribution_df[f"e_{col}_into_beta"]
        )

    contribution_df["sum_control"] = contribution_df.filter(
        [f"rc_{col}" for col in control_vars]
    ).sum(axis=1)

    for col in marketing_vars:
        contribution_df[f"rc_{col}"] = contribution_df["y_mkt"] * (
            1 - 1 / contribution_df[f"e_{col}_into_beta"]
        )

    contribution_df["sum_mkt"] = contribution_df.filter(
        [f"rc_{col}" for col in marketing_vars]
    ).sum(axis=1)
    return contribution_df


def _calculate_actual_contributions(
    contribution_df: pd.DataFrame,
    marketing_vars: List[str],
    control_vars: List[str],
    date_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Calculate actual contributions for marketing and control variables.

    Parameters
    ----------
    contribution_df : pd.DataFrame
        DataFrame with calculated values.
    marketing_vars : List[str]
        List of marketing variables.
    control_vars : List[str]
        List of control variables.
    date_col : str
        Name of the date column.
    group_col : str
        Name of the group column.

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated values.
    """
    for col in control_vars:
        contribution_df[f"ac_{col}"] = (
            contribution_df[f"rc_{col}"]
            * contribution_df["y_control"]
            / contribution_df["sum_control"]
        )

    for col in marketing_vars:
        contribution_df[f"ac_{col}"] = (
            contribution_df[f"rc_{col}"]
            * contribution_df["y_mkt"]
            / contribution_df["sum_mkt"]
        )

    contribution_df["actuals_without_intercept"] = (
        contribution_df["actuals"] - contribution_df["e_intercept"]
    )

    for col in marketing_vars + control_vars:
        contribution_df[f"ac_{col}"] = (
            contribution_df[f"ac_{col}"]
            * contribution_df["actuals_without_intercept"]
            / contribution_df["preds"]
        )

    final_contrib_df = contribution_df.filter(
        [date_col, group_col]
        + ["e_intercept"]
        + [f"ac_{col}" for col in marketing_vars + control_vars]
    ).rename(columns={"e_intercept": "base"})
    return final_contrib_df


def get_multiplicative_attribution(
    data: pd.DataFrame,
    coeff_matrix: pd.DataFrame,
    act_vs_preds: pd.DataFrame,
    date_col: str,
    marketing_vars: List[str],
    control_vars: List[str],
) -> pd.DataFrame:
    """
    Compute multiplicative attribution.

    Parameters
    ----------
    data : pd.DataFrame
        Data used to train the model.
    coeff_matrix : pd.DataFrame
        DataFrame containing the feature coefficients generated by training a model.
    act_vs_preds : pd.DataFrame
        DataFrame containing actual vs predicted values. This dataframe can be obtained when training a model or by using the coefficient dataframe.
    date_col : str
        Name of the date column.
    marketing_vars : List[str]
        List of marketing variables.
    control_vars : List[str]
        List of control variables.

    Returns
    -------
    pd.DataFrame
        DataFrame with the final contributions.
    """
    group_col = coeff_matrix.columns[0]
    contribution_df = _combine_marketing_and_control_vars(
        data, marketing_vars, control_vars
    )
    contribution_df = _merge_data_with_coefficients(
        contribution_df, data, coeff_matrix, date_col
    )
    contribution_df = _calculate_beta_into_x(
        contribution_df, marketing_vars, control_vars
    )
    contribution_df = _calculate_y_values(
        contribution_df, act_vs_preds, control_vars, date_col, group_col
    )
    contribution_df = _calculate_raw_contributions(
        contribution_df, marketing_vars, control_vars
    )
    final_contrib_df = _calculate_actual_contributions(
        contribution_df, marketing_vars, control_vars, date_col, group_col
    )
    return final_contrib_df
