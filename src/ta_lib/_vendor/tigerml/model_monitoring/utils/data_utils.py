import logging
import numpy as np
import pandas as pd
import pdb
import time
from tigerml.core.dataframe.dataframe import measure_time

_LOGGER = logging.getLogger(__name__)


def get_data_type(series_data, max_levels=0.05):
    """
    Fn to get Data type.

    Parameter
    ---------
    series_data: pd.Series
        Pandas series

    Returns
    -------
    str ('numerical' or 'categorical' or 'boolean)
    """
    if isinstance(max_levels, float) and (0 < max_levels < 1):
        max_levels = int(round(max_levels * len(series_data), 0))
    _LOGGER.info("Calculating max_levels value with respect to the current series_data")

    is_boolean = series_data.dtype == "bool" or set(series_data.unique()) <= {0, 1}
    if is_boolean:
        _LOGGER.info("Column is of boolean type")

    is_categorical = (
        series_data.dtype == "object" or series_data.nunique() <= max_levels
    ) and (is_boolean is False)
    if is_categorical:
        _LOGGER.info("Column is of categorical type")

    if not is_categorical and not is_boolean:
        _LOGGER.info("Column is of numerical type")

    if is_boolean:
        return "boolean"
    elif is_categorical:
        return "categorical"
    else:
        return "numerical"


def sort(list):
    """
    Fn to get Data type.

    Parameter
    ---------
    list: list

    Returns
    -------
    numerical or categorical
    """
    strings = [x for x in list if isinstance(x, str)]
    numbers = [x for x in list if not isinstance(x, str)]
    _LOGGER.info("Sorted data list")
    return sorted(numbers) + sorted(strings)


def compare_num_stats(base_df, curr_df, features=None):
    """
    Fn to get numerical summary.

    Parameter
    ---------
    base_df: pd.DataFrame
        Base Data
    current_df: pd.DataFrame
        Current Data
    features: list
        List of Features

    Return
    ------
    pd.DataFrame
    """
    if features is None:
        base_features = base_df.columns.tolist()
        curr_features = curr_df.columns.tolist()
        features = list(set(base_features).union(set(curr_features)))

    def num_desc(df):
        desc = df.describe(percentiles=[0.5]).T
        desc = desc.rename(columns={"50%": "median"})
        desc = desc.reset_index().rename(columns={"index": "variable"})
        return desc

    base_desc = num_desc(base_df[features])
    curr_desc = num_desc(curr_df[features])

    num_summary = pd.merge(
        base_desc, curr_desc, on="variable", suffixes=("_base", "_curr")
    )[
        [
            "variable",
            "count_base",
            "count_curr",
            "mean_base",
            "mean_curr",
            "std_base",
            "std_curr",
            "min_base",
            "min_curr",
            "median_base",
            "median_curr",
            "max_base",
            "max_curr",
        ]
    ]

    _LOGGER.info(
        "Got a concatenated dictionary of base and current numerical features nameshdg"
    )
    return num_summary


def compare_cat_stats(base_df, curr_df, features=None):
    """
    Categorical Summary.

    Compare summary stats of base data vs current data for categorical
    features.

    Parameter
    ---------
    base_df: pd.DataFrame
        Base Data
    current_df: pd.DataFrame
        Current Data
    features: list
        List of Features

    Return
    ------
    pd.DataFrame
    """
    if features is None:
        base_features = base_df.columns.tolist()
        curr_features = curr_df.columns.tolist()
        features = list(set(base_features).union(set(curr_features)))

    def cat_desc(df):
        desc = (
            df.astype("category")
            .describe()
            .T.rename(columns={"top": "mode", "freq": "mode_freq"})
        )
        desc["mode_freq_pct"] = desc["mode_freq"] / desc["count"]
        desc = desc.reset_index().rename(columns={"index": "variable"})
        return desc

    base_desc = cat_desc(base_df[features])
    curr_desc = cat_desc(curr_df[features])

    cat_summary = pd.merge(
        base_desc, curr_desc, on="variable", suffixes=("_base", "_curr")
    )[
        [
            "variable",
            "count_base",
            "count_curr",
            "unique_base",
            "unique_curr",
            "mode_base",
            "mode_curr",
            "mode_freq_base",
            "mode_freq_curr",
            "mode_freq_pct_base",
            "mode_freq_pct_curr",
        ]
    ]

    _LOGGER.info(
        "Got a concatenated dictionary of base and current categorical features names"
    )
    return cat_summary


def compare_bool_stats(base_df, curr_df, features=None):
    """
    Boolean Summary.

    Compare summary stats of base data vs current data for boolean
    features.

    Parameter
    ---------
    base_df: pd.DataFrame
        Base Data
    current_df: pd.DataFrame
        Current Data
    features: list
        List of Features

    Return
    ------
    pd.DataFrame
    """
    bool_summary = None

    if features is None:
        base_features = base_df.columns.tolist()
        curr_features = curr_df.columns.tolist()
        features = list(set(base_features).union(set(curr_features)))

    def bool_desc(df):
        desc = df[features].apply(lambda x: x.value_counts()).T
        res = desc.div(desc.sum(axis=0), axis=1)
        res = res.rename(columns={1: "Perc_1s", 0: "Perc_0s"})
        desc = desc.rename(columns={1: "count_1s", 0: "count_0s"})
        desc = pd.concat([desc, res], axis=1)
        desc = desc.reset_index().rename(columns={"index": "variable"})
        return desc

    base_desc = bool_desc(base_df)
    curr_desc = bool_desc(curr_df)

    bool_summary = pd.merge(
        base_desc, curr_desc, on="variable", suffixes=("_base", "_curr")
    )[
        [
            "variable",
            "count_0s_base",
            "count_0s_curr",
            "count_1s_base",
            "count_1s_curr",
            "Perc_0s_base",
            "Perc_0s_curr",
            "Perc_1s_base",
            "Perc_1s_curr",
        ]
    ]

    _LOGGER.info(
        "Got a concatenated dictionary of base and current boolean features names"
    )
    bool_summary = bool_summary.drop(
        columns=["Perc_0s_base", "Perc_0s_curr", "Perc_1s_base", "Perc_1s_curr"]
    )
    return bool_summary


def setanalyse(base, curr):
    """
    Set analysis by categorical levels in base and current feature.

    Parameter
    ---------
    base: list()
        Name of columns present in base data
    curr: pd.DataFrame
        Name of columns present in current data

    Return
    ------
    dict()
    """
    base = pd.Series(base)
    curr = pd.Series(curr)
    set_base, set_curr = set(base.unique()), set(curr.unique())
    intersection = set_base.intersection(set_curr)
    diff_base_curr = set_base - set_curr
    diff_curr_base = set_curr - set_base
    set_analysis = {
        "n_base": len(set_base),
        "n_curr": len(set_curr),
        "n_common": len(intersection),
        "n_base-curr": len(diff_base_curr),
        "n_curr-base": len(diff_curr_base),
        "base-curr": list(diff_base_curr),
        "curr-base": list(diff_curr_base),
        "setdiff": False if set_base == set_curr else True,
    }
    _LOGGER.info(
        "Set analysis by categorical levels in base and current feature completed"
    )
    return set_analysis


def setanalyse_by_features(base_df, curr_df, features=None, diff_only=False):
    """
    Feature-wise set analysis by categorical levels in base and current data.

    If diff_only is False, all features will be returned.
    Otherwise, returns only the difference cases.

    Parameter
    ---------
    base_df: pd.DataFrame
        Base Data
    current_df: pd.DataFrame
        Current Data
    features: list
        List of Features
    diff_only: bool
        Specify True for difference of base and curr data, else False

    Return
    ------
    pd.DataFrame
    """
    if features is None:
        features = base_df.columns.tolist()
    sa_list = []
    for col in features:
        set_analysis = setanalyse(base_df[col], curr_df[col])
        sa_list.append(set_analysis)
    comp_df = pd.DataFrame.from_records(sa_list, index=features)
    comp_df = comp_df.reset_index().rename(columns={"index": "variable"})
    if diff_only:
        comp_df = comp_df[comp_df["setdiff"]].reset_index(drop=True)
        _LOGGER.info("Difference between base and current features returned")
    _LOGGER.info("All features returned")
    return comp_df


def concat_dfs(df_dict, names=None, axis=0, reset_index=True, drop_existing_level=True):
    """
    Concatenate a dictionary of dataframes into a single dataframe.

    Parameter
    ---------
    df_dict: default_dict()
        Base Data
    names: List
        Name of the columns of the resulting dataframe
    reset_index: bool
        Specify True to reset index
    drop_existing_level: bool
        Specify True to drop existing level

    Return
    ------
    pd.DataFrame
    """
    df = pd.concat(df_dict, names=names, axis=axis)
    if drop_existing_level:
        df = df.droplevel(-1, axis=axis)
    if reset_index:
        df = df.reset_index()
    _LOGGER.info("Concatenated a dictionary of dataframes into a single dataframe")
    return df


def get_all_segments(df, segment_by):
    all_segments = df[segment_by].drop_duplicates().values.tolist()
    all_segments = list(map(tuple, all_segments))
    _LOGGER.info("Got all segments of a dataframe segmented by {}".format(segment_by))
    return all_segments


def get_all_segment_dfs(df, segment_by, keep=False, reset_index=True):
    """Convert a dataframe to dict of dfs grouped by columns."""
    cols = df.columns.tolist()
    if isinstance(segment_by, str):
        segment_by = [segment_by]
    if not keep:
        cols = [col for col in cols if col not in segment_by]
    dict_dfs = {}
    for key, index in df.groupby(segment_by).groups.items():
        if reset_index:
            dict_dfs[key] = df.loc[index, cols].reset_index(drop=True)
        else:
            dict_dfs[key] = df.loc[index, cols]
    _LOGGER.info("Converted a dataframe to dict of dfs grouped by columns")
    return dict_dfs


def flatten_dict(
    input_dict: dict,
    key_sep: str = None,
    output_dict: dict = None,
    parent_key: str = None,
):
    """Flattening Dictionary."""
    if output_dict is None:
        output_dict = {}
    for key, value in input_dict.items():
        if key_sep is None:
            if parent_key is None:
                new_key = (key,)
            else:
                new_key = parent_key + (key,)
        else:
            if parent_key is None:
                new_key = str(key)
            else:
                new_key = parent_key + key_sep + str(key)
        if isinstance(value, dict):
            flatten_dict(value, key_sep, output_dict, new_key)
        else:
            output_dict[new_key] = value
    _LOGGER.info("Flattened input dictionary")
    return output_dict
