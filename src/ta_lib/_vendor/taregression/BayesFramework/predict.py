"""Predict module predicts dependent variable for the test data."""

import collections
import os
import warnings

from .logging_utils import get_logger
from .model_evaluation import calculate_metrics
from .utils import reduce_samples, sample_mean  # noqa: F401

LOG = get_logger(__name__)

warnings.filterwarnings("ignore")


def preprocess_predict_data(data_df, spec_dict, mapped_df):
    """
    Removes special character from column name and maps the group columns.

    Parameters
    ----------
    data_df : dataframe
        dataframe that needs to be processed.
    spec_dict : dictionary
        dictionary with idvs name.
    mapped_df : dataframe
        mapped group variables.

    Returns
    -------
    duplicate_data : dataframe

    """
    duplicate_data = data_df.copy()
    # duplicate_data.columns = rem_specialchar_array(duplicate_data.columns)

    for group in spec_dict["group_cols"]:
        duplicate_data = duplicate_data[duplicate_data[group].notna()]
        duplicate_data[group + "_original"] = duplicate_data[group]
        rename_dict = mapped_df.set_index(group + "_original").to_dict()[group]
        duplicate_data[group] = duplicate_data[group].replace(rename_dict)

    return duplicate_data


def calculate_value(row, fixed_effect, spec_dict, random_effect, parameters):
    """
    Predicts dv.

    Parameters
    ----------
    row : dataframe
        row of prediction data.
    fixed_effect : list
        contains fixed effect variables name.
    spec_dict : dictionary
        contains all idvs name.
    random_effect : list
        contains random effect variables name.
    parameters : dictionary
        contains intercept and slope values.

    Returns
    -------
    val : int/float
        predicted value.

    """
    val = 0
    for slope in fixed_effect:
        val = val + (row[slope] * parameters["fixed_slope_" + slope])
    for group in spec_dict["group_cols"]:
        val = val + (parameters["intercept_" + group][int(row[group])])
    for var_slope in random_effect:
        name = "slope_" + var_slope[0] + "_" + var_slope[1]
        val = val + (row[var_slope[0]] * parameters[name][int(row[var_slope[1]])])
    if "fixed_slope_global_intercept" in parameters.keys():
        val = val + parameters["fixed_slope_global_intercept"]
    return val


def predict_value(row, fixed_effect, spec_dict, random_effect, parameters):
    """
    Predicts dv for VI.

    Parameters
    ----------
    row : dataframe
        row of prediction data.
    fixed_effect : list
        contains fixed effect variables name.
    spec_dict : dictionary
        contains all idvs name.
    random_effect : list
        contains random effect variables name.
    parameters : dictionary
        contains intercept and slope values.

    Returns
    -------
    val : int/float
        predicted value.

    """
    val = 0
    for slope in fixed_effect:
        val = val + (row[slope] * parameters["fixed_slope_" + slope]["mu"])
    for group in spec_dict["group_cols"]:
        val = val + (parameters["intercept_" + group]["mu"][int(row[group])])
    for var_slope in random_effect:
        name = "slope_" + var_slope[0] + "_" + var_slope[1]
        val = val + (row[var_slope[0]] * parameters[name]["mu"][int(row[var_slope[1]])])
    if "fixed_slope_global_intercept" in parameters.keys():
        val = val + parameters["fixed_slope_global_intercept"]["mu"]
    return val


def linear_estimate(join_dist_list, SamplesTuple):
    """
    Calculates final intercept and slope values.

    Parameters
    ----------
    join_dist_list : list

    SamplesTuple : tuple
        posterior samples.

    Returns
    -------
    tempdict : dictionary
        contains intercept and slope values.

    """
    L_intercept = []
    L_slope = []

    for var in join_dist_list[:-1]:
        if "mu_" not in var and "sigma_" not in var:
            if "intercept" in var:
                L_intercept.append(var)
            elif "slope" in var:
                L_slope.append(var)
    LinearEstimates = collections.namedtuple(  # noqa: F841
        "LinearEstimates", L_intercept + L_slope
    )
    L = L_intercept + L_slope
    s = ""
    for var in L:
        s = s + '''sample_mean(getattr(SamplesTuple,"''' + var + """")),"""

    s = s[: len(s) - 1]
    s = "LinearEstimates(" + s + ")"

    tempdict = {}
    try:
        varying_intercepts_and_slopes_estimates = eval(s)  # noqa: S307
        for i in L_intercept:
            tempdict[i] = getattr(varying_intercepts_and_slopes_estimates, i)
        for i in L_slope:
            tempdict[i] = getattr(varying_intercepts_and_slopes_estimates, i)
    except Exception as e:
        LOG.warning("Error while getting final slope and intercept values")
        LOG.exception(e)

    return tempdict


def prediction(
    data_pr,
    join_dist_list,
    SamplesTuple,
    spec_dict,
    mapped_df,
    fixed_effect,
    dt,
    random_effect,
    option,
    approx_param,
):
    """
    To Predict the values of the target variable. Saves the predicted values with original dataset.

    Parameters
    ----------
    data_pr : dataframe
        dataset for prediction .
    join_dist_list : list
    SamplesTuple : tuple
        contains posterior samples.
    spec_dict : dictionary
        contains all idvs name and hyperparameter information.
    mapped_df : dataframe
        contains mapped group variables.
    fixed_effect : list
        contains fixed effect variables name.
    dt : string
        date.
    random_effect : list
        contains random effect variables name.

    Returns
    -------
    y_pred : float
        predicted values.
    metrics : dict
        A dictionary of r2_score, rmse, MAPE, MAE, WMAPE metrics.

    """

    LOG.info("Running prediction")

    duplicate_data = preprocess_predict_data(data_pr, spec_dict, mapped_df)

    if option == "VI":
        try:
            duplicate_data["y_pred"] = duplicate_data.apply(
                lambda row: predict_value(
                    row, fixed_effect, spec_dict, random_effect, approx_param
                ),
                axis=1,
            )

        except Exception as e:
            LOG.warning("Error while predicting dv.")
            LOG.exception(e)

    else:
        parameters = linear_estimate(join_dist_list, SamplesTuple)
        try:
            duplicate_data["y_pred"] = duplicate_data.apply(
                lambda row: calculate_value(
                    row, fixed_effect, spec_dict, random_effect, parameters
                ),
                axis=1,
            )

        except Exception as e:
            LOG.warning("Error while predicting dv.")
            LOG.exception(e)

    predict_data = duplicate_data
    for group in spec_dict["group_cols"]:
        duplicate_data[group] = duplicate_data[group + "_original"]
        duplicate_data.drop([group + "_original"], axis=1, inplace=True)

    y_pred = duplicate_data["y_pred"].values
    metrics = calculate_metrics(predict_data, spec_dict)

    return y_pred, metrics
