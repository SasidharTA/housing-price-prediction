import logging
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from scipy.special import digamma, gamma
from scipy.stats import anderson_ksamp, chi2_contingency, ks_2samp
from tigerml.core.dataframe.dataframe import measure_time
from tigerml.model_monitoring.utils.misc import get_intervals, get_value_counts

_LOGGER = logging.getLogger(__name__)


def _psi_calculation(df: pd.DataFrame):
    df.fillna({"count_base": 0, "count_current": 0}, inplace=True)
    df["perc_base"] = df["count_base"] / df["count_base"].sum()
    df["perc_current"] = df["count_current"] / df["count_current"].sum()
    perc_diff = df["perc_base"] - df["perc_current"]
    log_ratio = np.log(df["perc_base"] / df["perc_current"])
    df["psi"] = perc_diff * log_ratio
    df["psi"].fillna(0, inplace=True)
    df["psi"].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _compute_gamma_parameters(data: pd.Series) -> float:
    mean = np.mean(data)
    variance = np.var(data)
    alpha = (mean**2) / variance
    beta = (mean) / (variance)
    return alpha, beta


def psi(base, current, feature_data_type, n_bins=10):
    """
    PSI drift detector.Returns Bin level and Aggregated psi value.

    Parameters
    ----------
    base: pd.DataFrame
        Data used as reference distribution.
    current: pd.DataFrame
        Current Data for which drift needs to be calculated
    n_bins: int
        Optionally specify the number of bins
    """
    random.seed(3)
    drift_dict = defaultdict(lambda: {})
    bins_df, base_count, current_count = get_value_counts(
        base,
        current,
        feature_data_type,
        n_bins,
    )
    merged = pd.concat([bins_df, base_count, current_count], axis=1)
    merged.columns = ["bins_or_categories", "count_base", "count_current"]
    merged = _psi_calculation(merged)
    drift_dict["aggregated"] = merged["psi"].sum()
    drift_dict["bin_level"] = merged
    return drift_dict


def dsi(
    base,
    current,
    feature_data_type,
    target_data_type,
    n_feature_bins=10,
    n_target_bins=5,
):
    """
    DSI data drift detector.

    Parameters
    ----------
    base: pd.DataFrame
        Base Data used as reference distribution with two columns Feature and Target
    current: pd.DataFrame
        Current Data used as reference distribution with two columns Feature and Target
    n_feature_bins: int
        Number of feature bins for splitting
    n_target_bins: int
        Number of target bins for splitting
    """
    base_x = base["Feature"]
    current_x = current["Feature"]
    base_y = base["Target"]
    current_y = current["Target"]
    feature_data_type = feature_data_type
    target_data_type = target_data_type
    n_feature_bins = n_feature_bins
    n_target_bins = n_target_bins
    random.seed(3)
    if feature_data_type == "numerical":
        base_feature_bins = get_intervals(
            base_x,
            current_x,
            nbins=n_feature_bins,
        )
        base_feature_cuts = pd.cut(base_x, bins=base_feature_bins, include_lowest=True)
        current_feature_cuts = pd.cut(
            current_x, bins=base_feature_bins, include_lowest=True
        )
    else:
        base_feature_cuts = base_x
        current_feature_cuts = current_x
    if target_data_type == "numerical":
        base_target_bins = get_intervals(
            base_y,
            current_y,
            nbins=n_target_bins,
        )
        base_target_cuts = pd.cut(base_y, bins=base_target_bins, include_lowest=True)
        current_target_cuts = pd.cut(
            current_y, bins=base_target_bins, include_lowest=True
        )
    else:
        base_target_cuts = base_y
        current_target_cuts = current_y
    key_cols = ["feature_bin", "target_bin"]
    base_pivot = pd.concat([base_feature_cuts, base_target_cuts], axis=1)
    base_pivot.columns = key_cols
    base_pivot_count = (
        base_pivot.groupby(key_cols).size().to_frame("count_base").reset_index()
    )
    current_pivot = pd.concat([current_feature_cuts, current_target_cuts], axis=1)
    current_pivot.columns = key_cols
    current_pivot_count = (
        current_pivot.groupby(key_cols).size().to_frame("count_current").reset_index()
    )
    merged_agg = base_pivot_count.merge(current_pivot_count, how="outer", on=key_cols)
    merged_agg = _psi_calculation(merged_agg)
    merged_agg.rename(columns={"psi": "dsi"}, inplace=True)
    drift_dict = defaultdict(lambda: {})
    drift_dict["aggregated"] = merged_agg["dsi"].sum()
    drift_dict["bin_level"] = merged_agg
    return drift_dict


def chiSquare(base, current):
    """
    Chi-Squared data drift detector.

    https://github.com/SeldonIO/alibi-detect/blob/master/alibi_detect/cd/chisquare.py

    Parameters
    ----------
    base: pd.Series
        Data used as reference distribution.
    current: pd.Series
        Current Data for which drift needs to be calculated.
    """
    _, base_count, current_count = get_value_counts(base, current)
    contingency_table = np.hstack((base_count, current_count))
    dist, p_val, _, _ = chi2_contingency(contingency_table)
    drift_dict = {"stats": dist, "pvalue": p_val}
    return drift_dict


def anderson(base, current):
    """
    Anderson Data Drift.

    Parameters
    ----------
    base: pd.Series
        Data used as reference distribution.
    current: pd.Series
        Current Data for which drift needs to be calculated.
    """
    dist, _, p_val = anderson_ksamp([base, current])
    drift_dict = {"stats": dist, "pvalue": p_val}
    return drift_dict


def kl(base, current, approximation=None):
    """
    KL Divergence Drift Detector.

    Computes the Kullback-Leibler divergence between two gamma distributions.
    https://www.researchgate.net/publication/278158089_NumPy_SciPy_Recipes_for_Data_Science_Computing_the_Kullback-Leibler_Divergence_between_Generalized_Gamma_Distributions

    Parameters
    ----------
    base: pd.Series
        Data used as reference distribution.
    current: pd.Series
        Current Data for which drift needs to be calculated
    approximation: str
        approximation for converting random variable into probability distribution
    """
    random.seed(3)
    p1 = 1
    p2 = 1
    try:
        alpha_1, beta_1 = _compute_gamma_parameters(base)
        alpha_2, beta_2 = _compute_gamma_parameters(current)
        theta_1 = 1 / beta_1
        theta_2 = 1 / beta_2

        a = p1 * (theta_2**alpha_2) * gamma(alpha_2 / p2)
        b = p2 * (theta_1**alpha_1) * gamma(alpha_1 / p1)
        c = (((digamma(alpha_1 / p1)) / p1) + np.log(theta_1)) * (alpha_1 - alpha_2)
        d = gamma((alpha_1 + p2) / p1)
        e = gamma((alpha_1 / p1))
        f = (theta_1 / theta_2) ** (p2)
        g = alpha_1 / p1

        dist = np.log(a / b) + c + (d / e) * f - g
    except Exception as e:
        dist = 0
    drift_dict = {"value": dist}
    # return only the value instead of the dict to avoid generating the column name as kldivergence_value
    # we need the column name as kldivergence
    return dist


def ks(base, current):
    """
    KS drift detector.

    This test compares the underlying continuous distributions F(x) and G(x)
    of two independent samples.

    Parameters
    ----------
    base: pd.Series
        Data used as reference distribution.
    current: pd.Series
        Current Data for which drift needs to be calculated
    """
    random.seed(3)
    dist, p_val = ks_2samp(base, current)
    drift_dict = {"stats": dist, "pvalue": p_val}
    return drift_dict
