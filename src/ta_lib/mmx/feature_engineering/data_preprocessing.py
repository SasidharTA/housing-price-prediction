import logging
import numpy as np
import pandas as pd
from math import exp, log
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import add_trend
from typing import List

logging.basicConfig(level=logging.INFO)


def add_fiscal_calendar(
    data: pd.DataFrame, date_col: List[str], calendar: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Add fiscal calendar identifiers to the DataFrame.

    If `calendar` is provided, it appends the fiscal calendar to the DataFrame.
    Otherwise, it uses the `date_col` to extract identifiers.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    date_col : List[str]
        A list with two elements: the first element is the name of the date column
        in the input DataFrame, and the second element is the name of the date column
        in the fiscal calendar.
    calendar : pd.DataFrame, optional
        The fiscal calendar DataFrame. Defaults to None.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame.
    """

    # Make a copy of the input dataframe
    temp = data.copy()

    # Merge input dataframe with fiscal calendar dataframe
    if calendar is not None:
        temp = pd.merge(
            temp, calendar, how="left", left_on=date_col[0], right_on=date_col[1]
        )
        return temp

    # Extract year, month, week, and quarter from the date column of the input dataframe
    temp["Year"] = temp[date_col[0]].dt.strftime("%Y")
    temp["Month"] = temp[date_col[0]].dt.strftime("%B")
    temp["Week"] = temp[date_col[0]].dt.strftime("%W")
    temp["Quarter"] = pd.PeriodIndex(temp[date_col[0]], freq="Q")
    temp["Quarter"] = temp["Quarter"].astype(str)

    return temp


def add_holidays(
    data: pd.DataFrame,
    holidays: pd.DataFrame,
    data_date_col: str,
    holiday_date_col: str,
    granularity: str,
    day_move: int = 6,
) -> pd.DataFrame:
    """Add an 'Is_Holiday' column to the dataframe based on specified holidays.

    If the granularity is set to the week level, the 'Is_Holiday' column will have 0 or 1 values.
    For other granularities, the 'Is_Holiday' column will contain the sum of holidays for the week or month.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame with the date column in date format.
    holidays : pd.DataFrame
        The DataFrame with holidays in date format.
    data_date_col : str
        The name of the column in the data containing the dates.
    holiday_date_col : str
        The name of the column in the holidays data containing the dates.
    granularity : str
        The level at which the data is considered. Acceptable inputs are 'Day', 'Week', and 'Month'.
    day_move : int, optional
        Used when the granularity is set to the week level. Specifies the weekday to which the dates in holidays
        should be adjusted. Takes input from 0 to 6, where 0 represents Monday. Defaults to 6.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an additional 'Is_Holiday' column indicating the presence of holidays.
    """
    data_copy = data.copy()
    holidays_copy = holidays.copy()

    if granularity == "Week":
        holidays_copy = move_nearest_day(
            holidays_copy, date_col=holiday_date_col, day=day_move
        )
    elif granularity == "Month":
        holidays_copy = move_month_end(holidays_copy, date_col=holiday_date_col)

    holidays_copy["Is_Holiday"] = 1
    holidays_agg = holidays_copy.groupby(holiday_date_col).sum().reset_index()

    result = pd.merge(
        data_copy,
        holidays_agg,
        how="left",
        left_on=data_date_col,
        right_on=holiday_date_col,
    )
    result["Is_Holiday"] = result["Is_Holiday"].fillna(0)

    return result


def move_nearest_day(
    data: pd.DataFrame, date_col: str, target_day: int = 6
) -> pd.DataFrame:
    """Shifts dates in a Pandas DataFrame to the nearest target day.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe.
    date_col : str
        The name of the date column in the dataframe.
    target_day : int, optional
        The target weekday to which the dates should be shifted (default is 6, which represents Saturday).
        This value should be an integer between 0 (Monday) and 6 (Sunday).

    Returns
    -------
    pd.DataFrame
        A new dataframe with the dates shifted to the nearest target day.
    """
    result_df = data.copy()
    result_df[date_col] = result_df[date_col] + pd.offsets.Week(n=0, weekday=target_day)
    return result_df


def move_month_end(data: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Move the dates in the DataFrame to the end of the month.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    date_col : str
        The column name on which the transformation should be applied.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the dates moved to the end of the month.
    """
    result = data.copy()

    result[date_col] = result[date_col] + pd.offsets.MonthEnd(0)

    return result


def add_linear_trend(
    data: pd.DataFrame,
    date_col: str,
    granularity: str = "Week",
    trend_yr_month: List = [],
) -> pd.DataFrame:
    """
    Add a linear trend column "trend" based on the date column, where the trend value increases as the date increases.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe.
    date_col : str
        The date identifier column.
    granularity : str, optional
        The level at which the data is considered. Acceptable inputs are "Day", "Week", and "Month". Defaults to "Week".
    trend_yr_month : List, optional
        Required input if the granularity is "Month". The first value in the list is the year column, and the second value is the month column.
        If empty, it will be generated based on the date_col.

    Returns
    -------
    pd.DataFrame
        The input dataframe with the "trend" column added.
    """

    if granularity == "Day":
        temp = pd.date_range(
            start=data[date_col].min(), end=data[date_col].max(), freq="D"
        )
        temp = add_trend(pd.DataFrame({date_col: temp}), trend="t")
        return data.merge(temp, how="left", on=date_col)

    elif granularity == "Week":
        temp = pd.date_range(
            start=data[date_col].min(), end=data[date_col].max(), freq="W"
        )
        temp = add_trend(pd.DataFrame({date_col: temp}), trend="t")
        return data.merge(temp, how="left", on=date_col)

    elif granularity == "Month":
        temp = pd.date_range(
            start=data[date_col].min(), end=data[date_col].max(), freq="W"
        )
        temp = pd.DataFrame({date_col: temp})

        if len(trend_yr_month) == 0:
            temp["X_YEAR"] = temp[date_col].dt.strftime("%Y").astype(int)
            temp["X_MONTH"] = temp[date_col].dt.strftime("%m").astype(int)
        else:
            temp["X_YEAR"] = temp[trend_yr_month[0]]
            temp["X_MONTH"] = temp[trend_yr_month[1]]
        min_year = temp["X_YEAR"].min()

        temp["trend"] = (temp["X_YEAR"] - min_year) + temp["X_MONTH"]

        temp.drop(["X_YEAR", "X_MONTH"], axis=1, inplace=True)

        return data.merge(temp, how="left", on=date_col)
    else:
        raise ValueError(
            "Invalid granularity. Available options are 'Day','Week' and 'Month'."
        )


def _add_granularity(
    data: pd.DataFrame, granularity: str, date_col: str, granularity_col: str = ""
) -> pd.DataFrame:
    """
    Add granularity column to the DataFrame based on the specified granularity level.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    granularity : str
        The granularity level to add. Supported values are "week" and "month".
    date_col : str
        The name of the date column in the DataFrame.
    granularity_col : str, optional
        The name of an existing column to use as the granularity column. If not provided, a new column will be created based on the date column. Defaults to "".

    Returns
    -------
    pd.DataFrame
        The DataFrame with the added granularity column.
    """
    temp = data.copy()
    if len(granularity_col) == 0:
        temp[f"X_{granularity}"] = (
            temp[date_col].dt.week if granularity == "week" else temp[date_col].dt.month
        )
    else:
        temp[f"X_{granularity}"] = temp[granularity_col]
    return temp


def _add_week_avg(temp: pd.DataFrame, dv_col: str, granularity: str) -> pd.DataFrame:
    """
    Add the weekly or monthly average column to the DataFrame.

    Parameters
    ----------
    temp : pd.DataFrame
        The input DataFrame.
    dv_col : str
        The name of the dependent variable column.
    granularity : str
        The granularity level used to calculate the average. Supported values are "week" and "month".

    Returns
    -------
    pd.DataFrame
        The DataFrame with the added average column.

    """
    data_w = temp.groupby(f"X_{granularity}", as_index=False).agg({dv_col: "mean"})
    avg = temp[dv_col].mean()
    data_w[dv_col] = data_w[dv_col] / avg
    s_col_name = "s_index" + ("_weekly" if granularity == "week" else "_monthly")
    data_w.rename(columns={dv_col: s_col_name}, inplace=True)
    temp = temp.merge(data_w, on=f"X_{granularity}", how="left")
    temp.drop(f"X_{granularity}", axis=1, inplace=True)
    return temp


def _add_custom(
    temp: pd.DataFrame, dv_col: str, granularity: str, threshold: List
) -> pd.DataFrame:
    """
    Add a custom column to the DataFrame based on specified thresholds.

    Parameters
    ----------
    temp : pd.DataFrame
        The input DataFrame.
    dv_col : str
        The name of the dependent variable column.
    granularity : str
        The granularity level used to calculate the custom column. Supported values are "week" and "month".
    threshold : List
        A list of two elements representing the lower and upper threshold values.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the added custom column.

    Raises
    ------
    ValueError
        If an incorrect number of arguments is provided for the threshold. It should contain exactly two elements.
    """
    temp = _add_week_avg(temp, dv_col, granularity)
    s_col_name = "s_index" + ("_weekly" if granularity == "week" else "_monthly")
    if len(threshold) != 2:
        raise ValueError(
            "Wrong number of arguments for threshold. Required elements are 2 for lower and upper threshold respectively."
        )
    temp[s_col_name] = np.where(
        temp[s_col_name] > threshold[1],
        1,
        np.where(temp[s_col_name] < threshold[0], -1, 0),
    )
    return temp


def _add_decompose(
    data: pd.DataFrame, dv_col: str, date_col: str, decompose_method: str, period: int
) -> pd.DataFrame:
    """
    Add a decomposition column to the DataFrame based on the specified decomposition method.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    dv_col : str
        The name of the dependent variable column.
    date_col : str
        The name of the date column in the DataFrame.
    decompose_method : str
        The decomposition method to use. Supported values are "additive" and "multiplicative".
    period : int
        The period length for seasonal decomposition.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the added decomposition column.
    """
    # Ensure date_col column is of datetime type
    data[date_col] = pd.to_datetime(data[date_col])

    temp1 = seasonal_decompose(data[dv_col], model=decompose_method, period=period)
    temp1 = pd.DataFrame(temp1.seasonal).reset_index()
    return data.merge(temp1, on=date_col, how="left")


def get_seasonality_column(
    data: pd.DataFrame,
    dv_col: str,
    date_col: str,
    decompose_method: str,
    granularity: str = "week",
    granularity_col: str = "",
    seasonal_method: str = "week_avg",
    threshold: List = None,
    period: int = 52,
) -> pd.DataFrame:
    """
    Add a seasonality column to the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
    dv_col : str
        The name of the dependent variable.
    date_col : str
        The name of the date variable.
    decompose_method : str
        The model to use when seasonal_method is set to "decompose". Acceptable inputs are "additive" and "multiplicative".
    granularity : str, optional
        The granularity of the data in the date_col column. Acceptable inputs are "week" and "month". Defaults to "week".
    granularity_col : str, optional
        The column that identifies the granularity. For example, if the granularity is week, enter the column name that contains the week number. If empty, it will be generated based on date_col.
    seasonal_method : str, optional
        The method used to create the seasonality column. Acceptable inputs are "week_avg", "decompose", and "custom". Defaults to "week_avg".
    threshold : List, optional
        A list with 2 elements containing the lower and upper thresholds, respectively. Required when seasonal_method is set to "custom". Defaults to None.
    period : int, optional
        The period length for the decompose method. Defaults to 52.

    Returns
    -------
    pd.DataFrame
        The dataset with the seasonality column.
    """
    temp = data.copy()

    if seasonal_method == "week_avg":
        temp = _add_granularity(temp, granularity, date_col, granularity_col)
        temp = _add_week_avg(temp, dv_col, granularity)

    elif seasonal_method == "custom":
        temp = _add_granularity(temp, granularity, date_col, granularity_col)
        temp = _add_custom(temp, dv_col, granularity, threshold)

    elif seasonal_method == "decompose":
        temp = _add_decompose(temp, dv_col, date_col, decompose_method, period)

    else:
        raise ValueError(
            "Wrong seasonal method. Acceptable inputs are {“week_avg” ,“decompose” ,“custom”}."
        )

    return temp


def _s_curve_values(alpha: float, beta: float, x: float) -> float:
    """
    Compute the S-Curve value for x given the alpha and beta parameters.

    Parameters
    ----------
    alpha : float
        The alpha parameter of the S-Curve.
    beta : float
        The beta parameter of the S-Curve.
    x : float
        The input value to be transformed.

    Returns
    -------
    float
        The S-Curve transformed value of x.

    """

    return alpha * (1 - np.exp(-1 * beta * x))


def get_scurve_transform(
    data: pd.DataFrame,
    date_col: str,
    alpha: List,
    beta: List,
    group_cols: List = None,
    columns: List = None,
) -> pd.DataFrame:
    """Apply the S-curve transform to the specified columns in the input DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    date_col : str
        The name of the column in the data containing the dates.
    alpha : List
        A list of alpha values to be used for creating the transformed columns.
    beta : List
        A list of beta values to be used for creating the transformed columns.
    group_cols : List, optional
        A list of column names by which data has to be grouped. Defaults to None.
    columns : List, optional
        A list of column names for which the S-curve transform has to be created.
        By default, it takes the list of all columns excluding group_cols and date_col.
        Defaults to None.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the S-curved columns.
    """
    if columns is None:
        if group_cols is not None:
            columns = [e for e in data.columns if e not in (*group_cols, date_col)]
        else:
            columns = [e for e in data.columns if e not in date_col]

    if group_cols is not None:
        data_res = data.sort_values([*group_cols, date_col])
        data_res["id"] = data_res.groupby(group_cols).ngroup()
    else:
        data_res = data.sort_values([date_col])
        data_res["id"] = 0

    ids = data_res["id"].unique()

    data_curve = pd.DataFrame()
    for id in ids:
        # Dataframe with columns to curve and group var filtering
        temp_data = data_res[data_res["id"] == id][columns]

        # Dataframe with curveed columns for single group var
        tmp_curve = pd.DataFrame()
        for idx_al, al in enumerate(alpha):
            for idx_beta, be in enumerate(beta):
                ad_dr = pd.DataFrame()
                for i in temp_data.columns:
                    ad_dr[i] = _s_curve_values(alpha=al, beta=be, x=temp_data[i])
                ad_dr = ad_dr.add_suffix(f"_alpha{idx_al}_beta{idx_beta}")
                tmp_curve = pd.concat([tmp_curve, ad_dr], axis=1)

        if group_cols is not None:
            tmp_curve = pd.concat(
                [
                    data_res[data_res["id"] == id][[*group_cols, date_col]].reset_index(
                        drop=True
                    ),
                    tmp_curve.reset_index(drop=True),
                ],
                axis=1,
            )
        else:
            tmp_curve = pd.concat(
                [
                    data_res[data_res["id"] == id][[date_col]].reset_index(drop=True),
                    tmp_curve.reset_index(drop=True),
                ],
                axis=1,
            )
        data_curve = pd.concat([data_curve, tmp_curve], ignore_index=True)

    alpha_map = [[f"alpha{idx}", i] for idx, i in enumerate(alpha)]
    beta_map = [[f"beta{idx}", i] for idx, i in enumerate(beta)]
    for i in beta_map:
        alpha_map.append(i)
    map_df = pd.DataFrame(alpha_map, columns=["Name", "Value"])

    return data_curve, map_df


def _apply_adstock(x: List, max_memory: int, decay: float) -> pd.Series:
    """
    Create adstock transformation for a given array with specified cutoff and decay.

    Parameters
    ----------
    x : List[float]
        The input array.
    max_memory : int
        The cutoff for the adstock transformation.
    decay : float
        The decay factor for the feature.

    Returns
    -------
    pd.Series
        The adstocked column.
    """
    # code reference from https://github.com/sibylhe/mmm_stan/blob/main/mmm_stan.py

    adstocked_x = []

    if max_memory != 0:
        x = np.append(np.zeros(max_memory - 1), x)

        weights = np.zeros(max_memory)
        for j in range(max_memory):
            weight = decay**j
            weights[max_memory - 1 - j] = weight

        for i in range(max_memory - 1, len(x)):
            x_array = x[i - max_memory + 1 : i + 1]
            xi = sum(x_array * weights)
            adstocked_x.append(xi)

    else:
        for i in x:
            if len(adstocked_x) == 0:
                adstocked_x.append(i)
            else:
                adstocked_x.append(i + decay * adstocked_x[-1])

    return pd.Series(adstocked_x, copy=False)


def create_adstock(
    data: pd.DataFrame,
    date_col: str,
    half_lives: List,
    max_memory: int = 0,
    group_cols: List = None,
    suffix: str = "",
    columns: List = None,
) -> pd.DataFrame:
    """
    Create Adstock transformation for specified columns in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    date_col : str
        The name of the column in the data containing the dates.
    half_lives : List
        A list of different half-lives for the Adstock transformation.
    max_memory : int, optional
        The number of values after which the decay effect will stop.
        If set to 0, the decay will not stop. Defaults to 0.
    group_cols : List[str], optional
        A list of column names by which data has to be grouped. Defaults to None.
    suffix : str, optional
        A character string (generally unit of half-lives) to be appended at the end of Adstock column names. Defaults to "".
    columns : List[str], optional
        A list of column names for which Adstock transformation has to be created.
        By default, it takes the list of all columns excluding group_cols and date_col. Defaults to None.

    Returns
    -------
    pd.DataFrame
        DataFrame with Adstocked columns.

    """

    if columns is None:
        if group_cols is not None:
            columns = [e for e in data.columns if e not in (*group_cols, date_col)]
        else:
            columns = [e for e in data.columns if e not in date_col]

    if group_cols is not None:
        data_res = data.sort_values([*group_cols, date_col])
        data_res["id"] = data_res.groupby(group_cols).ngroup()
    else:
        data_res = data.sort_values([date_col])
        data_res["id"] = 0

    ids = data_res["id"].unique()

    data_adstock = pd.DataFrame()
    for id in ids:
        # Dataframe with columns to adstock and group var filtering
        temp_data = data_res[data_res["id"] == id][columns]

        # Dataframe with adstocked columns for single group var
        tmp_adstock = pd.DataFrame()
        for n in half_lives:
            decay_rate = exp(log(0.5) / n)
            ad_dr = pd.DataFrame()
            for i in temp_data.columns:
                ad_dr[i] = _apply_adstock(
                    temp_data[i], max_memory=max_memory, decay=decay_rate
                )
            ad_dr = ad_dr.add_suffix(f"_{n}").add_suffix(suffix)
            tmp_adstock = pd.concat([tmp_adstock, ad_dr], axis=1)

        if group_cols is not None:
            tmp_adstock = pd.concat(
                [
                    data_res[data_res["id"] == id][[*group_cols, date_col]].reset_index(
                        drop=True
                    ),
                    tmp_adstock.reset_index(drop=True),
                ],
                axis=1,
            )
        else:
            tmp_adstock = pd.concat(
                [
                    data_res[data_res["id"] == id][[date_col]].reset_index(drop=True),
                    tmp_adstock.reset_index(drop=True),
                ],
                axis=1,
            )
        data_adstock = pd.concat([data_adstock, tmp_adstock], ignore_index=True)

    return data_adstock


def create_lag(
    data: pd.DataFrame,
    date_col: str,
    lags: List,
    group_cols: List = None,
    suffix: str = "",
    columns: List = None,
) -> pd.DataFrame:
    """
    Create lagged columns for specified columns in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    date_col : str
        The name of the column in the data containing the dates.
    lags : List[int]
        A list of different lags to create.
    group_cols : List[str], optional
        A list of column names by which data has to be grouped. Defaults to None.
    suffix : str, optional
        A character string (generally unit of half-lives) to be appended at the end of lagged column names. Defaults to "".
    columns : List[str], optional
        A list of column names for which lagged columns are to be created.
        By default, it takes the list of all columns excluding group_cols and date_col. Defaults to None.

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged columns.

    """

    if columns is None:
        if group_cols is not None:
            columns = [e for e in data.columns if e not in (*group_cols, date_col)]
        else:
            columns = [e for e in data.columns if e not in date_col]

    if group_cols is not None:
        data_res = data.sort_values([*group_cols, date_col])
        data_res["id"] = data_res.groupby(group_cols).ngroup()
    else:
        data_res = data.sort_values([date_col])
        data_res["id"] = 0

    ids = data_res["id"].unique()

    data_lag = pd.DataFrame()
    for id in ids:
        # Dataframe with columns to lag and group var filtering
        temp_data = data_res[data_res["id"] == id][columns]

        # Dataframe with lag columns for single group var
        tmp_lag = pd.DataFrame()
        for n in lags:
            ad_dr = pd.DataFrame()
            for i in temp_data.columns:
                ad_dr[i] = temp_data[i].shift(n).fillna(method="bfill")
            ad_dr = ad_dr.add_suffix(f"_{n}").add_suffix(suffix)
            tmp_lag = pd.concat([tmp_lag, ad_dr], axis=1)

        if group_cols is not None:
            tmp_lag = pd.concat(
                [
                    data_res[data_res["id"] == id][[*group_cols, date_col]].reset_index(
                        drop=True
                    ),
                    tmp_lag.reset_index(drop=True),
                ],
                axis=1,
            )
        else:
            tmp_lag = pd.concat(
                [
                    data_res[data_res["id"] == id][[date_col]].reset_index(drop=True),
                    tmp_lag.reset_index(drop=True),
                ],
                axis=1,
            )
        data_lag = pd.concat([data_lag, tmp_lag], ignore_index=True)

    return data_lag


def add_rolling_average(
    data: pd.DataFrame,
    columns: List,
    date_col: str,
    rolling_window: int,
    remove_rows: str = "keep",
) -> pd.DataFrame:
    """
    Add a moving average to the specified columns of the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    columns : List[str]
        The columns on which the moving average is to be performed.
    date_col : str
        The name of the column in the DataFrame containing the dates.
    rolling_window : int
        The size of the moving window.
    remove_rows : str, optional
        The method for handling rows with NaN values. Acceptable inputs are 'keep', 'drop', and 'append'.
        If 'keep', NaN values are kept in the final DataFrame.
        If 'drop', rows with NaN values are dropped from the final DataFrame.
        If 'append', NaN values at the start of the DataFrame are removed, and the original data is appended.
        Defaults to 'keep'.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the date_col and transformed columns.

    Raises
    ------
    ValueError
        If an invalid remove_rows option is provided. Acceptable inputs are 'keep', 'drop', and 'append'.

    """
    df = data.copy()
    df.set_index(date_col, inplace=True)

    if remove_rows == "keep":
        return df[columns].rolling(rolling_window).mean().reset_index()
    elif remove_rows == "drop":
        return df[columns].rolling(rolling_window).mean().reset_index().dropna()
    elif remove_rows == "append":
        temp_start = df[columns].head(rolling_window - 1).reset_index()
        return pd.concat(
            [
                temp_start,
                df[columns].rolling(rolling_window).mean().reset_index().dropna(),
            ]
        )
    else:
        raise ValueError(
            "Invalid remove_rows option . Accepatable inputs are keep ,drop and append"
        )
