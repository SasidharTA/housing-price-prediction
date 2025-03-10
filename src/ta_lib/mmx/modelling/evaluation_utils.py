import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from typing import List, Tuple

plt.style.use("ggplot")


def get_metrics(data: pd.DataFrame, train_test_col_name: str) -> pd.DataFrame:
    """Generate MAPE, RMSE, and R2 Score given data with train flag.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with "actuals" and "preds" columns and a "train_test_col_name" column.
    train_test_col_name : str
        Column name which contains train data as "train" and test data as "test".

    Returns
    -------
    pd.DataFrame
        DataFrame with MAPE, RMSE, and R2 Score grouped on train or test.

    """
    train_temp = data[data[train_test_col_name] == "train"]
    test_temp = data[data[train_test_col_name] == "test"]

    mse_train = mean_squared_error(train_temp["actuals"], train_temp["preds"])
    mse_test = mean_squared_error(test_temp["actuals"], test_temp["preds"])

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    mape_train = mean_absolute_percentage_error(
        train_temp["actuals"], train_temp["preds"]
    )
    mape_test = mean_absolute_percentage_error(test_temp["actuals"], test_temp["preds"])

    r2_train = r2_score(train_temp["actuals"], train_temp["preds"])
    r2_test = r2_score(test_temp["actuals"], test_temp["preds"])

    return pd.DataFrame(
        {
            "MAPE": [mape_train, mape_test],
            "RMSE": [rmse_train, rmse_test],
            "R2_Score": [r2_train, r2_test],
        },
        index=["Train Metrics", "Test Metrics"],
    )


def get_trace_plots(results: az.InferenceData, variables: List = []) -> None:
    """Generate and display trace plots.

    Parameters
    ----------
    results : Union[az.InferenceData, az.data.InferenceData]
        ArviZ InferenceData object containing the trace samples.
    variables : List[str], optional
        List of variable names to include in the trace plots. Default is an empty list.

    Returns
    -------
    None
        This function displays the generated trace plots.

    """
    try:
        if variables:
            az.plot_trace(results, var_names=variables, compact=True)
        else:
            az.plot_trace(results, compact=True)

        plt.show()
    except Exception as e:
        print(f"An error occurred while generating the trace plots: {e}")
        print(traceback.format_exc())


def get_confidence_intervals_plot(
    results: az.InferenceData,
    figsize: Tuple[int, int] = (6, 16),
    textsize: int = 10,
    r_hat: bool = True,
) -> None:
    """Generate and display a forest plot of confidence intervals..

    Parameters
    ----------
    results : arviz.InferenceData
        The InferenceData object containing the results of Bayesian inference.
    figsize : Tuple[int, int], optional
        The size of the figure (width, height). Default is (6, 16).
    textsize : int, optional
        The font size for text on the plot. Default is 10.
    r_hat : bool, optional
        Whether to include the R-hat statistic on the plot. Default is True.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame containing the computed statistics.
    """
    try:
        summary_df = az.summary(data=results)
        az.plot_forest(
            results, combined=True, figsize=figsize, textsize=textsize, r_hat=r_hat
        )
        plt.show()
        return summary_df
    except Exception as e:
        print(f"An error occurred while generating the confidence intervals plot: {e}")


def get_act_vs_preds_df(
    train_preds: np.array,
    test_preds: np.array,
    y_train: np.array,
    y_test: np.array,
    train_date_level_df: pd.DataFrame,
    test_date_level_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate the actuals vs predicted dataframe from predicted data.

    Parameters
    ----------
    train_preds : np.array
        Array containing training data predictions.
    test_preds : np.array
        Array containing test data predictions.
    y_train : np.array
        Array containing actual training data.
    y_test : np.array
        Array containing actual test data.
    train_date_level_df : pd.DataFrame
        Train dataframe having two columns: date and group.
    test_date_level_df : pd.DataFrame
        Test dataframe having two columns: date and group.

    Returns
    -------
    pd.DataFrame
        Actual vs. Predicted dataframe.
    """
    # get actuals vs predictions dataframe
    full_preds = np.hstack([train_preds, test_preds])
    full_actuals = np.hstack([y_train, y_test])

    actuals_vs_preds_df = pd.concat(
        [train_date_level_df, test_date_level_df], axis=0
    ).reset_index(drop=True)

    actuals_vs_preds_df["actuals"] = full_actuals
    actuals_vs_preds_df["preds"] = full_preds

    return actuals_vs_preds_df
