import logging
import numpy as np
import pandas as pd
from tigerml.core.dataframe.dataframe import measure_time
from tigerml.model_monitoring.config import threshold_options
from tigerml.model_monitoring.config.highlight_config import (
    COLOR_DICT,
    COLUMN_FORMATS,
    NUM_FORMAT_DICT,
)
from tigerml.model_monitoring.config.threshold_options import THRESHOLD_OPTIONS

_LOGGER = logging.getLogger(__name__)


def change_condition_string(cond, column):
    """
    Generate condition string as follows.

    1. lower_limit-upper_limit --> (lower_limit<=df['column']) & (df['column']<=upper_limit)
    2. <threshold --> df['column']<threshold
    3. >threshold --> df['column']>threshold
    """
    if "-" in cond:
        lower_limit = cond.split("-")[0]
        upper_limit = cond.split("-")[1]
        cond = (
            "("
            + lower_limit
            + "<=df['"
            + column
            + "']) & (df['"
            + column
            + "']<"
            + upper_limit
            + ")"
        )
    else:
        cond = "df['" + column + "']" + cond
    _LOGGER.info("Generated condition string")
    return cond


def get_conditions(thre_dict):
    """
    Restructure the thresholds dictionary to following format.

    {'column1': [ (condition_string1, color), (condition_string2, color), .... ],
     'column2': [ (condition_string1, color), (condition_string2, color), .... ],
     .
     .
     .
    }
    """
    col_cond_dict = {}
    level_dict = {
        "Low": COLOR_DICT["bg_green"],
        "Moderate": COLOR_DICT["bg_yellow"],
        "High": COLOR_DICT["bg_red"],
    }
    for key, value in thre_dict.items():
        if "High" in value["threshold"].keys():
            col_cond_dict[key] = []
            for level in value["threshold"].keys():
                col_cond_dict[key].append(
                    (
                        change_condition_string(value["threshold"][level], key),
                        level_dict[level],
                    )
                )

        else:
            if isinstance(value["threshold"], str):
                col_cond_dict[key] = [
                    (
                        change_condition_string(value["threshold"], key),
                        COLOR_DICT["bg_green"],
                    )
                ]
            else:
                for sub_key in value["threshold"].keys():
                    col_name = key + "_" + sub_key
                    col_cond_dict[col_name] = [
                        (
                            change_condition_string(
                                value["threshold"][sub_key], col_name
                            ),
                            COLOR_DICT["bg_green"],
                        )
                    ]
    _LOGGER.info("Generated column condition dictionary from thresholds dictionary")
    return col_cond_dict


def apply_numerical_formating(df, column_dict=COLUMN_FORMATS):
    """
    Applying numerical formatting to a given dataframe.

    Parameters
    ----------
    df: pandas.DataFrame or pandas.Styler
        Dataframe to applying styling on.
    column_dict: dict
        Contains column name and its corresponding numerical formatting to be applied.

    Returns
    -------
    styler: pandas.Styler
        Returns pandas style object
    """
    if isinstance(df, pd.DataFrame):
        styler = df.style
    elif isinstance(df, pd.io.formats.style.Styler):
        styler = df
    else:
        return "Not a valid input"
    format_dict = {
        k: NUM_FORMAT_DICT[v] for k, v in column_dict.items() if v in NUM_FORMAT_DICT
    }
    styler = styler.format(format_dict)
    _LOGGER.info("Created a pandas styling object for numerical data")
    return styler


def apply_conditional_bg_coloring(df, metric_list=THRESHOLD_OPTIONS.copy()):
    """
    Applying condition based background color for selected metric columns.

    Parameters
    ----------
    df: pandas.DataFrame or pandas.Styler
        Dataframe to applying styling on.
    metric_list: dict
        Contains metric name and its corresponding threshold info.

    Returns
    -------
    styler: pandas.Styler
        Returns pandas style object
    """
    if isinstance(df, pd.DataFrame):
        styler = df.style
    elif isinstance(df, pd.io.formats.style.Styler):
        styler = df
    else:
        return "Not a valid input"
    col_cond_dict = get_conditions(metric_list)
    styler_df = pd.DataFrame("", index=styler.index, columns=styler.columns)
    for key, value in col_cond_dict.items():
        if key in df.columns:
            if len(value) == 1:
                styler_df[key] = np.where(
                    eval(value[0][0]), COLOR_DICT["bg_red"], COLOR_DICT["bg_green"]
                )
            else:
                for cond in value:
                    styler_df.loc[eval(cond[0]), key] = cond[1]
    styler.apply(lambda _: styler_df, axis=None)
    _LOGGER.info(
        "Adding condition based background coloring for selected metrics columns to styler"
    )
    return styler


def table_formatter(
    df,
    metric_list=None,
    column_dict=COLUMN_FORMATS,
    format_numerical=True,
    format_bg_color=True,
):
    """
    Adding styles to the dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe to applying styling on.
    column_dict: dict
        Contains column name and its corresponding numerical formatting to be applied.
    metric_list: dict
        Contains metric name and its corresponding threshold info.
    format_numerical: bool
        If set to true apply's numerical formating on selected columns.
    format_bg_color: bool
        If set to true apply's background color on given metric columns.

    Returns
    -------
    styler: pandas.Styler
        Returns pandas style object
    """
    threshold_values = threshold_options.update_threshold_options(
        thresholds=metric_list
    )
    if not isinstance(df, pd.DataFrame):
        _LOGGER.info("Not a valid pandas dataframe")
        return df
    else:
        if format_bg_color:
            df = apply_conditional_bg_coloring(df, metric_list=threshold_values)
            _LOGGER.info("Highlighting of df done")
        if format_numerical:
            df = apply_numerical_formating(df, column_dict=column_dict)
            _LOGGER.info("Highlighting of df done")
        return df


def apply_table_formatter_to_dict(report_dict=None):
    for key, value in report_dict.items():
        if isinstance(value, dict):
            apply_table_formatter_to_dict(report_dict=value)
        elif isinstance(value, pd.DataFrame):
            report_dict[key] = table_formatter(value, format_bg_color=True)
        else:
            report_dict[key] = value
    return report_dict
