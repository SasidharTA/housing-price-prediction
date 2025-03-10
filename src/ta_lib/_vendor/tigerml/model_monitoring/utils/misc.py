import logging
import numpy as np
import pandas as pd
import pdb
from tigerml.core.dataframe.dataframe import measure_time
from tigerml.model_monitoring.utils import highlighting
from tokenize import Name

_LOGGER = logging.getLogger(__name__)


def apply_threshold(df, dict_key, summary_options):
    """
    Applies threshold to a dataframe.

    Parameters
    ----------
    dict_key: dict
        It acts a key like feature_drift_numerical or feature_drift_categorical
        to apply threshold on.
    summary_options: dict
        dictionary containing thresholds for each of the drift.

    Returns
    -------
        Sets true false in dataframe and returns it.
    """
    threshold_on = summary_options[dict_key]["threshold_on"]
    threshold_value = summary_options[dict_key]["threshold_value"]

    if type(threshold_value) == str:
        condition_string = highlighting.change_condition_string(
            cond=threshold_value, column=threshold_on
        )
        df_drifted = df.loc[eval(condition_string)]["variable"].count()
        _LOGGER.info("Applied threshold to df with string threshold value")
        return df_drifted
    else:
        level_dict = {}
        for level in threshold_value:
            condition_string = highlighting.change_condition_string(
                cond=threshold_value[level], column=threshold_on
            )
            df_drifted = df.loc[eval(condition_string)]["variable"].count()
            level_dict[level] = df_drifted
        _LOGGER.info("Applied threshold to df with dictionary threshold value")
        return level_dict


def get_applicable_metrics(drift_metrics, data_type, drift_type):
    """
    Get All applicable metrics for a particular data and drift type.

    Parameters
    ----------
    data_type: str
        Numerical/Categorical
    drift_type: str
        Numerical/Categorical

    Returns
    -------
        List: all the metrics available for corresponding data type and drift type.

    """
    metrics_applicable = [
        key
        for key, value in drift_metrics.items()
        if (data_type in value["applicable_data_type"])
        and (drift_type in value["applicable_drift_type"])
    ]
    _LOGGER.info(
        "Got all applicable metrics for data type {} and drift type {}".format(
            data_type, drift_type
        )
    )
    return metrics_applicable


def get_intervals(base, current, nbins=10):
    """
    Getinterval.

    It generates different cuts with specified number of bins and also it
    replaces lower and higher value with both base and current.

    Parameters
    ----------
    base: pd.DataFrame
        Base data is used to get cuts.
    current: pd.DataFrame
        Current data for changing boundaries.
    nbins: int
        Number of bins for which you want to calculate cuts.

    Returns
    -------
        Bins or Decile cuts.

    """
    _, base_bins = pd.qcut(base, nbins, retbins=True, duplicates="drop")
    base_bins[0] = min(base.min(), current.min())
    base_bins[-1] = max(base.max(), current.max())
    _LOGGER.info(
        "Generated different cuts with specified number of bins and replaced lower and higher value with both base and current"
    )
    return base_bins


def get_value_counts(base, current, data_type="categorical", n_bins=10, bins=None):
    """
    Get counts for base and current.

    Parameters
    ----------
    base: pd.DataFrame
        Base data/ Reference Data
    current: pd.DataFrame
        Current Data
    data_type: str
        If set to numerical will use bins else set to categorical will
        use actual categories.
    n_bins: int
        How many bins/cuts you want for data.
    bins:
        Actual bins to be used.

    Returns
    -------
        bins dataframe, count for each bin for base and current.

    """
    if data_type == "numerical":
        if bins:
            base_bins = bins
        else:
            base_bins = get_intervals(base, current, nbins=n_bins)
        base_cuts = pd.cut(base, bins=base_bins, include_lowest=True)
        current_cuts = pd.cut(current, bins=base_bins, include_lowest=True)
        base_count = base_cuts.value_counts().reset_index().iloc[:, 1]
        current_count = current_cuts.value_counts().reset_index().iloc[:, 1]
        bins_df = pd.DataFrame({"bins": base_cuts.cat.categories})
    else:
        df = pd.merge(
            base.value_counts(dropna=False).reset_index(),
            current.value_counts(dropna=False).reset_index(),
            on=["index"],
            how="outer",
        )
        df.iloc[:, [1, 2]].fillna(0, inplace=True)
        bins_df, base_count, current_count = (
            df.iloc[:, 0],
            df.iloc[:, 1],
            df.iloc[:, 2],
        )
    _LOGGER.info(
        "Got count of data in base and current df based on {} bins".format(n_bins)
    )
    return bins_df, base_count, current_count
