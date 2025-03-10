"""model_evaluation module lets you calculate and return the model metrics."""

from math import sqrt

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .logging_utils import get_logger

LOG = get_logger(__name__)


def calculate_metrics(
    duplicate_data,
    spec_dict,
):
    """
    To calculate the metrics of the result.

    Parameters
    ----------
    duplicate_data : dataframe
        dataframe that contains true and predicted values of dv.
    spec_dict : dictionary
        Contains information about dv and idvs.

    Returns
    -------
    metrics : dict
        A dictionary of r2_score, rmse, MAPE, MAE, WMAPE metrics.

    """
    try:
        LOG.info("Calculating metrics")
        y_pred = duplicate_data["y_pred"].values
        y_true = duplicate_data[spec_dict["dv"]].values

        # MAPE and WMAPE
        mask = y_true != 0
        mape = 100 * (np.fabs(y_true - y_pred) / y_true)[mask].mean()
        y_true_sum = y_true.sum()
        y_true_prod_mape = y_true[mask] * (
            100 * (np.fabs(y_true - y_pred) / y_true)[mask]
        )
        y_true_prod_mape_sum = y_true_prod_mape.sum()
        wmape = y_true_prod_mape_sum / y_true_sum

        # r2
        r2_scor = r2_score(y_true, y_pred)
        # mae
        mae = mean_absolute_error(y_true, y_pred)

        RMSE = sqrt(mean_squared_error(y_true, y_pred))
        LOG.info(f"MAPE  : {mape}")
        LOG.info(f"WMAPE  : {wmape}")
        LOG.info(f"r2_score  : {r2_scor}")
        LOG.info(f"MAE  : {mae}")
        LOG.info(f"RMSE  : {RMSE}")

        metrics = {
            "r2_score": r2_scor,
            "rmse": RMSE,
            "MAPE": mape,
            "MAE": mae,
            "WMAPE": wmape,
        }
        return metrics

    except Exception as e:
        LOG.warning("Error while calculating metrics")
        LOG.exception(e)
