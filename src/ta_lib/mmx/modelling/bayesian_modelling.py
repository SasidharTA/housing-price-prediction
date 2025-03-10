import arviz as az
import bambi as bmb
import cloudpickle
import numpy as np
import pandas as pd
import re
from datetime import datetime
from typing import Tuple

from .data_preparation import prepare_train_test_data
from .evaluation_utils import get_act_vs_preds_df


def process_input_priors(pr_cfg_df: pd.DataFrame, data: pd.DataFrame) -> dict:
    """
    Map user input priors in an Excel file to Bambi prior format.

    Parameters
    ----------
    pr_cfg_df : pd.DataFrame
        Dataframe containing input priors.
        Typically, this is a CSV file "prior config file" converted to a dataframe.
    data : pd.DataFrame
        Training data (used to check if the input independent variables are present in the data).

    Returns
    -------
    processed_priors : dict
        Dictionary containing priors formatted as required by Bambi.
    """
    priors_mapping = {
        "normal": "Normal",
        "halfnormal": "HalfNormal",
    }

    processed_priors = {}

    for _, row in pr_cfg_df.iterrows():
        idv = row["idv"]

        # check if idv in prior config file is present in data
        if idv not in data.columns.tolist():
            raise ValueError(
                f"{idv} is not present in the data, Please the check the column names in the prior config excel file"
            )

        # if we have a prior input from the user
        if not pd.isnull(row["prior_est"]):
            input_prior_est = row["prior_est"].lower()

            # check if prior distribution is there in the mapping dictionary
            if input_prior_est not in priors_mapping:
                raise ValueError(
                    f"Please use priors from this list, {priors_mapping.keys()}. To use other priors, directly use priors in the format that bambi requires"
                )

            if input_prior_est == "normal":
                try:
                    mu = row["mu"]
                    sigma = row["sigma"]
                    processed_priors[row["idv"]] = bmb.Prior(
                        "Normal", mu=mu, sigma=sigma
                    )
                except:  # noqa
                    raise Exception(
                        f"Please enter valid mu and sigma for {input_prior_est}"
                    )
            elif input_prior_est == "halfnormal":
                try:
                    sigma = row["sigma"]
                    processed_priors[row["idv"]] = bmb.Prior("HalfNormal", sigma=sigma)
                except:  # noqa
                    raise Exception(f"Please enter valid sigma for {input_prior_est}")

    return processed_priors


def create_model_equation(pr_cfg_df, group_col, random_intercept=True) -> str:
    """
    Create a mixed model equation based on user input.

    Parameters
    ----------
    pr_cfg_df : pd.DataFrame
        Dataframe containing input priors.
        Typically, this is a CSV file "prior config file" converted to a dataframe.
    group_col : str
        Column name of the group variable for random effects.

    random_intercept : bool, optional
        True if the intercept should be fixed at all group levels.
        False if the intercept should be random at the group level.
        Defaults to True.

    Returns
    -------
    model_equation : str
        The mixed model equation.
    """

    dv = pr_cfg_df["dv"].unique()[0]

    fixed_effects = pr_cfg_df.query("is_fixed == 1")["idv"].values.tolist()
    fixed_equation = " + ".join(fixed_effects)

    # check if we have any random effect
    is_mixed_model = pr_cfg_df.query("is_random == 1")["idv"].values.tolist()

    random_idvs = [f"{int(random_intercept)}"] + pr_cfg_df.query("is_random == 1")[
        "idv"
    ].values.tolist()

    # check if we have random effects from the excel file
    # if we dont have any random effects, just have fixed
    # effects to the rhs equation, else add both fixed and
    # random effects.
    if is_mixed_model:
        random_equation = f"({' + '.join(random_idvs)} | {group_col})"
        rhs_eq = f"{fixed_equation} + {random_equation}"
    else:
        rhs_eq = fixed_equation

    model_equation = dv + " ~ " + rhs_eq
    return model_equation


def _get_bambi_predictions(pred_object):
    """
    Get predictions from a Bambi model.

    Parameters
    ----------
    pred_object : object
        The prediction object returned by the Bambi model.

    Returns
    -------
    final_preds : np.array
        Array of predictions.
    """
    final_preds = (
        az.summary(pred_object)
        .reset_index()
        .query("index.str.contains('_mean')")["mean"]
        .values
    )

    return final_preds


def save_bambi_model(
    filename: str, model_object: bmb.Model, inference_object: az.InferenceData
) -> None:
    """
    Save the model trace, model equation, and training data.

    Parameters
    ----------
    filename : str
        The filename to save the model data in. If not specified, the current date
        and time will be used as the filename.
    model_object : bmb.Model
        The Bambi model object.
    inference_object : az.InferenceData
        The model inference data to save.

    Returns
    -------
    None
    """
    if not filename:
        filename = f"{datetime.today().strftime('%d%m%Y_%H%M%S')}"
    else:
        filename = f"{filename}_{datetime.today().strftime('%d%m%y_%H%M')}"

    model_dict = {
        "inference_data": inference_object,
        "data": model_object.data,
        "backend_model": model_object.backend.model,
        "model_equation": model_object.formula,
        "family": model_object.family,
    }

    with open(filename, "wb") as f:
        cloudpickle.dump(model_dict, f)

    print(f"Model data saved in {filename}")


def load_bambi_model(filename: str = "") -> Tuple[bmb.Model, az.InferenceData]:
    """
    Load a Bambi model from a saved directory.

    Parameters
    ----------
    filename : str, optional
        Directory where the saved Bambi model exists

    Returns
    -------
    Tuple
        A tuple containing a Bambi fitted model and the fitted trace
    """

    with open(filename, "rb") as f:
        model_dict = cloudpickle.load(f)

    new_model = bmb.Model(
        data=model_dict["data"],
        family=model_dict["family"],
        formula=model_dict["model_equation"],
    )
    new_model.build()

    new_model.backend.model = model_dict["backend_model"]

    return new_model, model_dict["inference_data"]


def _get_coefficient_matrix_bayesian(
    input_df: pd.DataFrame,
    feature_cols: list,
    intercept_col: str,
    group_col: str,
    group_values: list,
) -> pd.DataFrame:
    """
    Generate a coefficient matrix from the coefficients bambi models.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing feature coefficients which is generated by the bayesian model.
    feature_cols : list
        List of feature columns which were used to train the model.
    intercept_col : str
        Column name representing the intercept.
    group_col : str
        Column name of the group variable.
    group_values : list
        List of unique values present in the group column of the data.

    Returns
    -------
    pd.DataFrame
        Coefficient matrix.
    """

    # Initialize the output matrix with zeros
    output_matrix = pd.DataFrame(
        0, index=group_values, columns=feature_cols + [intercept_col]
    )

    # Calculate the final coefficients for each feature
    for i, row in input_df.iterrows():
        feature_name = row["feature names"]
        feature_coefficient = row["feature coefficients"]

        # fixed intercept
        if feature_name == intercept_col:
            output_matrix.loc[:, intercept_col] = feature_coefficient

        # fixed slopes
        elif feature_name in feature_cols:
            output_matrix.loc[:, feature_name] = feature_coefficient

        # random intercept and slopes
        else:
            # we want to extract feature_name and group_value
            # from this string ----> feature_name|group_col[group_value]
            pattern = r"^(.*)\|.*\[(.*)\]$"
            matches = re.match(pattern, feature_name)
            if matches:
                column_name = (
                    intercept_col if matches.group(1) == "1" else matches.group(1)
                )
                group_index = matches.group(2)
                output_matrix.loc[group_index, column_name] += feature_coefficient

    output_matrix.columns = [f"beta_{col}" for col in feature_cols] + [intercept_col]
    output_matrix = output_matrix.reset_index()
    output_matrix = output_matrix.rename(columns={"index": group_col})
    output_matrix["e_intercept"] = np.exp(output_matrix[intercept_col])
    # output_matrix = output_matrix.drop(intercept_col, axis=1)

    return output_matrix


def train_bayesian_model(
    data: pd.DataFrame,
    idv_cols: list,
    train_test_col_name: str,
    date_col: str,
    group_col: str,
    model_equation: str,
    priors_config: dict = None,
    model_args: dict = {},
    random_seed: int = 2054,
    save: bool = True,
    model_filename: str = "",
) -> Tuple[bmb.Model, az.InferenceData, pd.DataFrame, pd.DataFrame]:
    """
    Train a Mixed effect bayesian model.

    Parameters
    ----------
    data : pd.DataFrame
        Full data having all columns.
    idv_cols : List
        List of column names to be used for training.
    train_test_col_name : str
        Column name having train test flag value.
    date_col : str
        Column name having date value.
    group_col : str
        Column name of group variable
    model_equation : str
        Bambi model equation
    priors_config : dict, optional
        Dictionary of priors to be passed to model. Defaults to None.
    model_args : dict, optional
        Dictionary containing the model parameters. Defaults to None.
    random_seed : int, optional
        Setting a random seed so that the model gives the same result every time we run.
    save : bool, optional
        If True, saves the model parameters, model equation, and train data in a directory.
        Defaults to True.
    model_filename : str, optional
        Folder name of the directory where the model details are stored.

    Returns
    -------
    Tuple[Model, InferenceData, pd.DataFrame, pd.DataFrame]
        Returns the Bambi model, inference data, feature coefficients, and actuals vs predicted dataframe
    """

    dv_name = model_equation.split("~")[0].strip()

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
        dv_col=dv_name,
    )

    # Add the target variable to the X_train and X_test dataframes
    X_train[dv_name] = y_train
    X_test[dv_name] = y_test

    # Define the model using Bambi
    model = bmb.Model(model_equation, data=X_train, family="gaussian")
    if priors_config is not None:
        model.set_priors(priors_config)

    model_args["include_mean"] = False
    model_args["random_seed"] = random_seed
    # Fit the model using MCMC sampling
    trace = model.fit(**model_args)

    if save is True:
        save_bambi_model(
            filename=model_filename,
            model_object=model,
            inference_object=trace,
        )

    # Get the posterior distribution of the model coefficients
    feature_df = (
        az.summary(trace)["mean"].reset_index().query("~index.str.contains('y_mean')")
    )
    feature_df = feature_df.rename(
        columns={"index": "feature names", "mean": "feature coefficients"}
    )

    group_values = data[group_col].unique().tolist()
    coeff_matrix = _get_coefficient_matrix_bayesian(
        input_df=feature_df,
        feature_cols=idv_cols,
        intercept_col="Intercept",
        group_col=group_col,
        group_values=group_values,
    )

    # Make predictions on the training and test data
    train_preds_object = model.predict(idata=trace, data=X_train, inplace=False)
    test_preds_object = model.predict(idata=trace, data=X_test, inplace=False)

    train_preds = _get_bambi_predictions(pred_object=train_preds_object)
    test_preds = _get_bambi_predictions(pred_object=test_preds_object)

    # get actuals vs predictions dataframe
    actuals_vs_preds_df = get_act_vs_preds_df(
        train_preds,
        test_preds,
        y_train,
        y_test,
        train_date_level_df,
        test_date_level_df,
    )

    return model, trace, feature_df, coeff_matrix, actuals_vs_preds_df
