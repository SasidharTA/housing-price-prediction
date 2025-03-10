import pandas as pd


def generate_spends_vs_activity_data(
    data: pd.DataFrame, mapping: pd.DataFrame, date_col: str, group_col: str = None
) -> pd.DataFrame:
    """Create a long-format DataFrame of spends and activity for variables mentioned in the mapping file using master data.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    mapping : pd.DataFrame
        Mapping DataFrame.
    date_col : str
        Name of the column in the data containing the dates.
    group_col : str, optional
        Name of the column in the data containing the group variable. If not present, the resulting DataFrame will not have group_col. Defaults to None.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with spends and activity.
    """
    # extract relevant columns and convert to weekly level
    tp_cols = mapping["variable_name"].tolist()
    if group_col is not None:
        master_data_subset = data[[date_col, group_col, *tp_cols]]
    else:
        master_data_subset = data[[date_col, *tp_cols]]
    # scale variables that vary by group
    nat_tp = mapping[mapping["aggregation_type"] == "Mean"]["variable_name"].tolist()
    local_tp = list(set(tp_cols) - set(nat_tp))

    if group_col is not None:
        master_data_nat = (
            master_data_subset[[date_col, group_col, *nat_tp]]
            .groupby([date_col, group_col])
            .mean()
            .reset_index()
        )
        master_data_loc = (
            master_data_subset[[date_col, group_col, *local_tp]]
            .groupby([date_col, group_col])
            .sum()
            .reset_index()
        )
        master_data_subset = pd.merge(
            master_data_nat, master_data_loc, on=[date_col, group_col]
        )
    else:
        master_data_nat = (
            master_data_subset[[date_col, *nat_tp]]
            .groupby([date_col])
            .mean()
            .reset_index()
        )
        master_data_loc = (
            master_data_subset[[date_col, *local_tp]]
            .groupby([date_col])
            .sum()
            .reset_index()
        )
        master_data_subset = pd.merge(master_data_nat, master_data_loc, on=[date_col])

    # convert to long format and add relevant columns
    if group_col is not None:
        df = pd.melt(
            master_data_subset, id_vars=[date_col, group_col], var_name="variable_name"
        )
    else:
        df = pd.melt(master_data_subset, id_vars=[date_col], var_name="variable_name")
    df = df.merge(
        mapping[
            [
                "variable_name",
                "variable_activity_root",
                "variable_description",
                "variable_category",
            ]
        ],
        on="variable_name",
    )

    # create pivot and export data
    if group_col is not None:
        df_pivot = df.pivot_table(
            index=[
                date_col,
                group_col,
                "variable_activity_root",
                "variable_description",
            ],
            columns="variable_category",
            values="value",
        ).reset_index()
        df_pivot = pd.melt(
            df_pivot,
            id_vars=[
                date_col,
                group_col,
                "variable_activity_root",
                "variable_description",
                "Spend",
            ],
            var_name="activity_type",
            value_name="Activity",
        )
        df_pivot.sort_values(
            ["variable_activity_root", date_col, group_col], inplace=True
        )
    else:
        df_pivot = df.pivot_table(
            index=[date_col, "variable_activity_root", "variable_description"],
            columns="variable_category",
            values="value",
        ).reset_index()
        df_pivot = pd.melt(
            df_pivot,
            id_vars=[
                date_col,
                "variable_activity_root",
                "variable_description",
                "Spend",
            ],
            var_name="activity_type",
            value_name="Activity",
        )
        print(df_pivot)
        df_pivot.sort_values(["variable_activity_root", date_col], inplace=True)

    df_pivot.rename(columns={date_col: date_col}, inplace=True)
    return df_pivot.reset_index(drop=True)


def generate_quarterly_spends_data(
    data: pd.DataFrame,
    mapping: pd.DataFrame,
    group_col: str,
    year_col: str,
    quarter_col: str,
) -> pd.DataFrame:
    """
    Generate quarterly spend data based on the provided input data and mapping.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing spend data.
    mapping : pd.DataFrame
        The mapping DataFrame containing variable mappings.
    group_col : str
        The column name specifying the group.
    year_col : str
        The column name specifying the year.
    quarter_col : str
        The column name specifying the quarter.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing quarterly spend data.

    """
    tp_cols = mapping["variable_name"].tolist()

    temp = data.copy()

    if temp[year_col].dtypes == "object" and temp[quarter_col].dtypes == "object":
        temp["YEAR_QTR"] = temp[year_col] + temp[quarter_col]
    elif temp[year_col].dtypes != "object" and temp[quarter_col].dtypes == "object":
        temp["YEAR_QTR"] = str(temp[year_col]) + temp[quarter_col]
    elif temp[year_col].dtypes == "object" and temp[quarter_col].dtypes != "object":
        temp["YEAR_QTR"] = temp[year_col] + str(temp[quarter_col])
    elif temp[year_col].dtypes != "object" and temp[quarter_col].dtypes != "object":
        temp["YEAR_QTR"] = str(temp[year_col]) + str(temp[quarter_col])

    if group_col is not None:
        master_data_subset = temp[["YEAR_QTR", group_col, *tp_cols]]
    else:
        master_data_subset = temp[["YEAR_QTR", *tp_cols]]

    # scale variables that vary by group
    nat_tp = mapping[mapping["aggregation_type"] == "Mean"]["variable_name"].tolist()
    local_tp = list(set(tp_cols) - set(nat_tp))

    if group_col is not None:
        master_data_nat = (
            master_data_subset[["YEAR_QTR", group_col, *nat_tp]]
            .groupby(["YEAR_QTR", group_col])
            .mean()
            .reset_index()
        )
        master_data_loc = (
            master_data_subset[["YEAR_QTR", group_col, *local_tp]]
            .groupby(["YEAR_QTR", group_col])
            .sum()
            .reset_index()
        )
        master_data_subset = pd.merge(
            master_data_nat, master_data_loc, on=["YEAR_QTR", group_col]
        )
    else:
        master_data_nat = (
            master_data_subset[["YEAR_QTR", *nat_tp]]
            .groupby(["YEAR_QTR"])
            .mean()
            .reset_index()
        )
        master_data_loc = (
            master_data_subset[["YEAR_QTR", *local_tp]]
            .groupby(["YEAR_QTR"])
            .sum()
            .reset_index()
        )
        master_data_subset = pd.merge(master_data_nat, master_data_loc, on=["YEAR_QTR"])

    if group_col is not None:
        df = pd.melt(
            master_data_subset,
            id_vars=["YEAR_QTR", group_col],
            var_name="variable_name",
        )
    else:
        df = pd.melt(master_data_subset, id_vars=["YEAR_QTR"], var_name="variable_name")

    df = df.merge(
        mapping[["variable_name", "variable_description"]], on="variable_name"
    )

    return df
