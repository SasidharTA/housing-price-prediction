# import random
# import re
# from datetime import timedelta
import numpy as np
import pandas as pd

# import pandas_flavor as pf
# import math
# from IPython.display import display


def set_baseline_value(
    model_df,
    model_coef,
    var_col="model_coefficient_name",
    coef_col="model_coefficient_value",
    intercept_name="(Intercept)",
    baseline_value=None,
):
    """Set baseline value.

    Parameters
    ----------
    model_df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables' coefficient estimate
    intercept_name : str
        Name of intercept variable
    baseline_value : pd.DataFrame
        Dataframe consisting of IDVs with baseline value


    Method
        - If action is MEAN, then baseline value is set to the mean value
        - If action is MODE, then baseline value is set to the mode value
        - If action is MEDIAN, then baseline value is set to the median value
        - If action is MAX, then baseline value is set to the max value
        - If action is MIN, then baseline value is set to the min value
        - If action is As is, then it is taken as is. Since the contribution will be 0 we exclude it while calculating attribution
        - Variables not specified in the baseline file values are unchanged and no attribution is found.

    Returns
    -------
    pd.DataFrame
    """
    # Get baseline value

    # Initialization

    base_var = [intercept_name] + baseline_value["Parameters"].to_list()
    baseline_df = model_df.loc[:, model_df.columns.isin(model_coef[var_col])].copy()

    # Set baseline value
    temp_list = list(
        np.intersect1d(baseline_value["Parameters"].to_list(), baseline_df.columns)
    )
    for i in temp_list:

        action = baseline_value.loc[
            baseline_value["Parameters"] == i, "Action"
        ].to_list()[0]

        if action.isnumeric():
            value = baseline_value.loc[
                baseline_value["Parameters"] == i, "Action"
            ].to_list()[0]

            baseline_df[i] = value

        else:
            action = action.upper()

            if action == "MEAN":
                baseline_df[i] = baseline_df[i].mean()
            elif action == "AS IS":
                baseline_df[i] = model_df[i]
            elif action == "MAX":
                baseline_df[i] = model_df[i].max()
            elif action == "MIN":
                baseline_df[i] = model_df[i].min()
            elif action == "MEDIAN":
                baseline_df[i] = model_df[i].median()
            elif action == "MODE":
                baseline_df[i] = model_df[i].mode()
            elif action == "":
                baseline_df[i] = 0
            else:
                # Do nothing
                print(
                    "Enter a valid Action: Max, Min, Mean, Median, Mode, As is, a numeric or leave blank"
                )

    baseline_df = baseline_df.astype(float)
    return base_var, baseline_df


def _predict(
    pred_df,
    model_coef,
    intercept_df,
    model_type,
    gv,
    var_col="model_coefficient_name",
    coef_col="model_coefficient_value",
    intercept_name="(Intercept)",
):
    """Predict Sales.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Dataframe consisting of IDVs
    model_coef : pd.DataFrame
        Dataframe consisting of model coefficient and its value
    intercept_df : pd.DataFrame
        Dataframe consisting of random intercepts and their coefficients
    model_type : str
        Type of model
    gv : str
        Grouping variable
    var_col : str
        Column containing model variable
    coef_col : str
        Column containing model variables and their coefficient estimate
    intercept_name : str
        Name of intercept variable

    Returns
    -------
    ndarray
    """

    idv_cols = [col for col in model_coef[var_col] if col != intercept_name]
    idv_coef = model_coef.loc[model_coef[var_col].isin(idv_cols), coef_col]
    idv_df = pred_df.loc[:, idv_cols]
    intercept = model_coef.loc[~model_coef[var_col].isin(idv_cols), coef_col].to_list()[
        0
    ]

    prediction = []
    if model_type == "Fixed_effect":
        for i in pred_df[gv].unique():
            preds = idv_df.values.dot(idv_coef.values) + intercept
            prediction.append(preds)
    else:
        for i in pred_df[gv].unique():
            preds = (
                idv_df.values.dot(idv_coef.values)
                + intercept
                + intercept_df.loc[intercept_df["group_var"] == i, "beta"].to_list()[0]
            )
            prediction.append(preds)
    return prediction


def get_var_contribution_variants(
    dist_df, var_col_name, value_col_name, group, gv, model_type, time_column
):
    """Get variable contribution by different variants.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Dataframe consisting of IDVs
    var_col_name : str
        Column name for melted IDV columns
    value_col_name : str
        Column name for IDV values
    group : str
        Group value
    gv : str
        Grouping variable name
    model_type : str
        Type of model
    time_column : str
        Time/Date variable name

    Returns
    -------
    tuple of pd.DataFrame
    """
    numeric_cols = dist_df.select_dtypes(include="number").columns.to_list()

    # Gayatri++
    # Datewise Aggregation
    dt_dist_df = dist_df.copy()
    dt_dist_df[time_column] = dt_dist_df[time_column].dt.date.astype(str)
    dt_dist_df = dt_dist_df.groupby(by=[time_column], as_index=False).agg(np.sum)

    pct_dist_df = dt_dist_df.copy()
    pct_dist_df[numeric_cols] = pct_dist_df[numeric_cols].div(
        pct_dist_df["Predicted_sales" + "_" + group], axis=0
    )

    dist_df_1 = pd.merge(
        dt_dist_df.melt(
            id_vars=[time_column], var_name=var_col_name, value_name=value_col_name
        ),
        pct_dist_df.melt(
            id_vars=[time_column],
            var_name=var_col_name,
            value_name="pct_" + value_col_name,
        ),
        how="left",
        on=[time_column, var_col_name],
    )
    dist_df_1[gv] = group

    # Quarterly Aggegration
    qtr_dist_df = dist_df.copy()
    qtr_dist_df["Quarter"] = (
        qtr_dist_df["Date"].dt.year.astype(str)
        + "-"
        + "Q"
        + qtr_dist_df["Date"].dt.quarter.astype(str)
    )
    qtr_dist_df = qtr_dist_df.drop(columns=[time_column])
    qtr_dist_df = qtr_dist_df.groupby(by=["Quarter"], as_index=False).agg(np.sum)

    pct_qtr_dist_df = qtr_dist_df.copy()
    pct_qtr_dist_df[numeric_cols] = pct_qtr_dist_df[numeric_cols].div(
        pct_qtr_dist_df["Predicted_sales" + "_" + group], axis=0
    )
    dist_df_2 = pd.merge(
        qtr_dist_df.melt(
            id_vars=["Quarter"], var_name=var_col_name, value_name=value_col_name
        ),
        pct_qtr_dist_df.melt(
            id_vars=["Quarter"],
            var_name=var_col_name,
            value_name="pct_" + value_col_name,
        ),
        how="left",
        on=["Quarter", var_col_name],
    )
    dist_df_2[gv] = group

    # Yearly Aggegration
    yr_dist_df = dist_df.copy()
    yr_dist_df["Year"] = yr_dist_df[time_column].dt.year.astype(str)
    yr_dist_df = yr_dist_df.drop(columns=[time_column])
    yr_dist_df = yr_dist_df.groupby(by=["Year"], as_index=False).agg(np.sum)

    pct_yr_dist_df = yr_dist_df.copy()
    pct_yr_dist_df[numeric_cols] = pct_yr_dist_df[numeric_cols].div(
        pct_yr_dist_df["Predicted_sales" + "_" + group], axis=0
    )
    dist_df_3 = pd.merge(
        yr_dist_df.melt(
            id_vars=["Year"], var_name=var_col_name, value_name=value_col_name
        ),
        pct_yr_dist_df.melt(
            id_vars=["Year"], var_name=var_col_name, value_name="pct_" + value_col_name
        ),
        how="left",
        on=["Year", var_col_name],
    )
    dist_df_3[gv] = group

    return dist_df_1, dist_df_2, dist_df_3


def get_attribution( # noqa
    model_data,
    model_coef,
    gv,
    model_type,
    time_column,
    rand_intercept_df,
    attrbution_config,
    attribution_type,
):
    """Get variable contribution with or without baseline defined.

    Parameters
    ----------
    model_data : pd.DataFrame
        Dataframe consisting of IDVs of the selected Product. The following columns must be present alongside the IDV columns.

        - PPG_Item_No - Product column in str format
        - Date – Time scale like Date, Week, Month in mm/dd/yyyy format

    model_coef : pd.DataFrame
        Dataframe consisting of model variables and their coefficient estimate

        - Data frame with model coefficients of the selected product. Intercept to be given as ‘(Intercept)’

    gv : str
        Group variable name

        - Name of the column which has the ppg names (Product column) here it is given as: ‘PPG_Item_No’

    model_type : str
        Type of the model

        - Model type to given as ‘Fixed_effect’ or ‘Mixed_effect’

    time_column : str
        Time/Date variable name

        - Column containing Date: ‘Date’.

    rand_intercept_df : pd.DataFrame
        Dataframe consisting of random intercepts and their coefficients

    attrbution_config : pd.DataFrame
        Dataframe consisting of IDVs with baseline value/base variables

        - with_baseline_defined.csv - prefilled by the user with baseline values for the respective columns in IDV file.
        - without_baseline_defined.csv – prefilled by the user with variables that are to be considered as “Base” variables.

    attribution_type : Integer 1 or 0
        0 - Without baseline value
        1 - With baseline value

    Returns
    -------
    tuple of pd.DataFrame
    """

    # Dataframe consisting of IDV Columns with baseline value/base variables
    attrbution_config["Action"] = attrbution_config["Action"].astype(str)

    baseline_value = attrbution_config[["Parameters", "Action"]]

    # Gayatri++
    if model_type == "Mixed_effect":
        model_coef = model_coef.loc[model_coef["column"] != "intercept_" + gv]

    # Column name with coeffients name
    var_col = model_coef.columns[0]

    # Column name with coeffients value
    coef_col = model_coef.columns[1]

    # Name of the intercept variable
    intercept_name = [i for i in model_coef[var_col] if "ntercept" in i][0]

    # Column with product names
    # model_cols = model_data.dropna(axis=1, how="all").columns.tolist()[0]
    # Base Variable with all coefficients name removing intercept
    x = attrbution_config["Parameters"].to_list()
    base_var = x

    if attribution_type == 0:

        # Predict Sales
        model_df = model_data.copy()

        prediction = np.exp(
            _predict(
                model_df,
                model_coef,
                rand_intercept_df,
                model_type,
                gv,
                var_col=var_col,
                coef_col=coef_col,
                intercept_name=intercept_name,
            )
        )
        # rishu++
        for m, j in zip(model_df[gv].unique(), range(0, len(prediction))):
            k = "Predicted_sales_" + m
            model_df[k] = prediction[j]

        # Get base and impact variables

        base_var = [i for i in base_var if i in model_coef[var_col].to_list()]

        base_var = [intercept_name] + base_var

        impact_var = [i for i in model_coef[var_col] if i not in base_var]

        dt_unit_dist_df = pd.DataFrame()
        qtr_unit_dist_df = pd.DataFrame()
        yr_unit_dist_df = pd.DataFrame()

        # Get base and impact variables
        model_df[intercept_name] = 1

        # Gayatri ++
        model_df2 = model_df.copy()
        for i in model_df[gv].unique():
            model_df = model_df2[model_df2[gv] == i]
            tmp_model_coef = model_coef[model_coef[var_col].isin(base_var)]
            tmp_model_df = model_df.copy()

            base_val = tmp_model_df[tmp_model_coef[var_col].to_list()].values.dot(
                tmp_model_coef[coef_col].values
            )

            tmp_model_coef = model_coef[model_coef[var_col].isin(impact_var)]

            impact_val = model_df[tmp_model_coef[var_col].to_list()].values.dot(
                tmp_model_coef[coef_col].values
            )

            model_df["baseline_contribution" + "_" + i] = np.exp(base_val)
            model_df["incremental_contribution" + "_" + i] = (
                model_df["Predicted_sales" + "_" + i]
                - model_df["baseline_contribution" + "_" + i]
            )

            # Calculate raw contribution for impact variables
            row_sum = 0
            abs_sum = 0
            for j in impact_var:
                tmp_impact_val = (
                    model_df[j].values
                    * model_coef.loc[model_coef[var_col] == j, coef_col].to_list()[0]
                )
                model_df[j + "_contribution_impact" + "_" + i] = np.exp(
                    base_val + impact_val
                ) - np.exp(base_val + impact_val - tmp_impact_val)
                row_sum = row_sum + model_df[j + "_contribution_impact" + "_" + i]
                abs_sum = abs_sum + abs(model_df[j + "_contribution_impact" + "_" + i])

            y_b_s = model_df["incremental_contribution" + "_" + i] - row_sum

            impact_contribution = model_df[
                [time_column, "Predicted_sales" + "_" + i]
            ].copy()
            for j in impact_var:
                i_adj = j + "_contribution_impact" + "_" + i
                impact_contribution[i_adj] = (
                    model_df[i_adj] + (abs(model_df[i_adj]) / abs_sum) * y_b_s
                )

            # Calculate raw contribution for base variables
            if model_type == "Mixed_effect":
                base_rc = model_coef.loc[
                    model_coef[var_col] == intercept_name, coef_col
                ].to_list()[0]
                # Gayatri ++
                # + rand_intercept_df.loc[rand_intercept_df['group_var'] == i , 'beta'].to_list()[0])
            else:
                base_rc = model_coef.loc[
                    model_coef[var_col] == intercept_name, coef_col
                ].to_list()[0]
            impact_contribution[
                intercept_name + "_contribution_base" + "_" + i
            ] = np.exp(base_rc)
            for j in base_var[1:]:
                t = (
                    tmp_model_df[j]
                    * model_coef.loc[model_coef[var_col] == j, coef_col].to_list()[0]
                    + base_rc
                )
                impact_contribution[j + "_contribution_base" + "_" + i] = np.exp(
                    t
                ) - np.exp(base_rc)
                base_rc = t
            impact_contribution = impact_contribution.fillna(0)
            unit_dist_df = impact_contribution.copy()

            # Get variable contribution variants
            dt_unit, qtr_unit, yr_unit = get_var_contribution_variants(
                unit_dist_df,
                "model_coefficient_name",
                "units",
                i,
                gv,
                model_type,
                time_column,
            )
            dt_unit_dist_df = pd.concat([dt_unit_dist_df, dt_unit], axis=0)
            qtr_unit_dist_df = pd.concat([qtr_unit_dist_df, qtr_unit], axis=0)
            yr_unit_dist_df = pd.concat([yr_unit_dist_df, yr_unit], axis=0)

        overall_dt_dist_df = dt_unit_dist_df
        overall_qtr_dist_df = qtr_unit_dist_df
        overall_yr_dist_df = yr_unit_dist_df

        overall_dt_dist_df["model_coefficient_name"] = (
            overall_dt_dist_df["model_coefficient_name"]
            .str.split("_")
            .str[:-1]
            .str.join("_")
        )
        overall_qtr_dist_df["model_coefficient_name"] = (
            overall_qtr_dist_df["model_coefficient_name"]
            .str.split("_")
            .str[:-1]
            .str.join("_")
        )
        overall_yr_dist_df["model_coefficient_name"] = (
            overall_yr_dist_df["model_coefficient_name"]
            .str.split("_")
            .str[:-1]
            .str.join("_")
        )

        # overall_dt_dist_df[model_cols] = ppg
        # overall_qtr_dist_df[model_cols] = ppg
        # overall_yr_dist_df[model_cols] = ppg

        return overall_dt_dist_df, overall_qtr_dist_df, overall_yr_dist_df

    if attribution_type == 1:

        # print("**** Attribution With Baseline Defined ****")

        model_df = model_data.copy()
        # model_coef = model_coef[model_coef["column"] != "sigma_target"]
        prediction = np.exp(
            _predict(
                model_df,
                model_coef,
                rand_intercept_df,
                model_type,
                gv,
                var_col=var_col,
                coef_col=coef_col,
                intercept_name=intercept_name,
            )
        )

        # Predict Sales
        for n, j in zip(model_df[gv].unique(), range(0, len(prediction))):
            k = "Predicted_sales_" + n
            model_df[k] = prediction[j]

        # Predict Baseline Sales
        tmp_model_df = model_df.copy()

        base_var, baseline_df = set_baseline_value(
            tmp_model_df, model_coef, var_col, coef_col, intercept_name, baseline_value
        )

        baseline_df[gv] = model_df[gv]

        count = 0
        for i in model_df[gv].unique():
            baseline_df["Predicted_baseline_sales" + "_" + i] = np.exp(
                _predict(
                    baseline_df,
                    model_coef,
                    rand_intercept_df,
                    model_type,
                    gv,
                    var_col=var_col,
                    coef_col=coef_col,
                    intercept_name=intercept_name,
                )
            )[count]

            model_df["Predicted_baseline_sales" + "_" + i] = baseline_df[
                "Predicted_baseline_sales" + "_" + i
            ]
            model_df["incremental_predicted" + "_" + i] = (
                model_df["Predicted_sales" + "_" + i]
                - model_df["Predicted_baseline_sales" + "_" + i]
            )

            # Calculate raw contribution
            model_df[intercept_name] = 1
            baseline_df[intercept_name] = 1
            pred_xb = _predict(
                model_df,
                model_coef,
                rand_intercept_df,
                model_type,
                gv,
                var_col=var_col,
                coef_col=coef_col,
                intercept_name=intercept_name,
            )[count]

            count = count + 1
            rc_sum = 0
            abs_sum = 0

            # Remove Attribution for columns if not defined in config.csv and if Action is "As is"
            temp_attribution_config = baseline_value.copy()

            temp_attribution_config["Action"] = temp_attribution_config[
                "Action"
            ].str.upper()
            asis_list = temp_attribution_config[
                temp_attribution_config["Action"] == "AS IS"
            ]["Parameters"].to_list()

            model_coef_updated = model_coef.copy()

            for j in model_coef[var_col]:
                if j != intercept_name:
                    if j not in attrbution_config["Parameters"].to_list():

                        model_coef_updated.drop(
                            model_coef[model_coef[var_col] == j].index, inplace=True
                        )
                    elif j in asis_list:

                        model_coef_updated.drop(
                            model_coef[model_coef[var_col] == j].index, inplace=True
                        )

            for j in model_coef_updated[var_col]:
                rc = 0

                rc = np.exp(pred_xb) - np.exp(
                    pred_xb
                    - (
                        model_df[j]
                        * model_coef_updated.loc[
                            model_coef_updated[var_col] == j, coef_col
                        ].to_list()[0]
                    )
                    + (
                        baseline_df[j]
                        * model_coef_updated.loc[
                            model_coef_updated[var_col] == j, coef_col
                        ].to_list()[0]
                    )
                )

                model_df[j + "_" + "rc"] = rc
                rc_sum = rc_sum + rc
                abs_sum = abs_sum + abs(rc)

            # Calculate actual contribution
            y_b_s = model_df["incremental_predicted" + "_" + i] - rc_sum
            unit_dist_df = model_df[
                [
                    "Date",
                    "Predicted_sales" + "_" + i,
                    "Predicted_baseline_sales" + "_" + i,
                ]
            ].copy()

            for j in model_coef_updated[var_col]:
                rc = model_df[j + "_" + "rc"]
                ac = rc + (abs(rc) / abs_sum) * y_b_s
                i_adj = (
                    j + "_contribution_inc_base"
                    if j in base_var
                    else j + "_contribution_impact"
                )
                unit_dist_df[i_adj] = ac
            unit_dist_df = unit_dist_df.fillna(0)

            # Get variable contribution variants
            (
                dt_unit_dist_df,
                qtr_unit_dist_df,
                yr_unit_dist_df,
            ) = get_var_contribution_variants(
                unit_dist_df,
                "model_coefficient_name",
                "units",
                i,
                gv,
                model_type,
                time_column,
            )

        overall_dt_dist_df = dt_unit_dist_df

        overall_qtr_dist_df = qtr_unit_dist_df

        overall_yr_dist_df = yr_unit_dist_df

        # overall_dt_dist_df[model_cols] = ppg
        # overall_qtr_dist_df[model_cols] = ppg
        # overall_yr_dist_df[model_cols] = ppg

        return overall_dt_dist_df, overall_qtr_dist_df, overall_yr_dist_df
