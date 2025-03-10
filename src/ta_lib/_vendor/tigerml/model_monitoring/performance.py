import logging
import numpy as np
import pandas as pd
import pdb
from collections import defaultdict
from sklearn.metrics import auc
from tigerml.core.dataframe.dataframe import measure_time
from tigerml.core.scoring import SCORING_OPTIONS, scorers
from tigerml.model_monitoring.utils.data_utils import get_data_type

_LOGGER = logging.getLogger(__name__)


class Performance:
    """Calculate performance of a model."""

    def __init__(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        yhat_base: str,
        yhat_curr: str,
        y_base: str,
        y_curr: str,
    ):

        self.base_df = base_df
        self.current_df = current_df
        self.yhat_base = yhat_base
        self.yhat_curr = yhat_curr
        self.y_base = y_base
        self.y_curr = y_curr
        self.metrics = None
        self.is_probability = False
        _LOGGER.info("Initiated performance class")

    def _check_probabilities(self, report_type):
        if report_type != "numerical":
            if np.issubdtype(
                self.base_df[self.yhat_base].dtype.type,
                int,
            ) and np.issubdtype(
                self.current_df[self.yhat_curr].dtype.type,
                int,
            ):
                self.is_probability = False
            else:
                self.is_probability = True

    def _set_metrics(self):
        report_type = get_data_type(self.base_df[self.y_base])

        if report_type == "numerical":
            self.metrics = SCORING_OPTIONS.regression.copy()
        elif report_type == "boolean" or self.base_df[self.y_base].nunique() == 2:
            self.metrics = SCORING_OPTIONS.classification.copy()
        else:
            self.metrics = SCORING_OPTIONS.multi_class.copy()

        self._check_probabilities(report_type)
        probabs_metric_list = ["log_loss", "roc_auc"]
        if self.is_probability:
            self.metrics = {
                metric: self.metrics[metric] for metric in probabs_metric_list
            }

    def _compute_performance_drift(self):
        self._set_metrics()

        performance_dict = defaultdict(lambda: defaultdict(dict))
        for metric in self.metrics.keys():
            func = self.metrics[metric].get("func", [])
            default_params = self.metrics[metric].get("default_params", [])
            performance_dict["base"][metric] = func(
                self.base_df[self.y_base], self.base_df[self.yhat_base]
            )

            performance_dict["current"][metric] = func(
                self.current_df[self.y_curr], self.current_df[self.yhat_curr]
            )

        final = pd.DataFrame(performance_dict).reset_index()
        final.rename(columns={"index": "measures"}, inplace=True)
        final["index"] = final["current"] / final["base"]
        _LOGGER.info("Model performance calculated")
        return final
