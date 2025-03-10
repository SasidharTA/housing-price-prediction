from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from tigerml.core.reports import table_styles
from tigerml.core.utils._lib import DictObject

from .scorers import mape, root_mean_squared_error, wmape

TRAIN_PREFIX = "train"
TEST_PREFIX = "test"

SCORING_OPTIONS = DictObject(
    {
        "classification": {
            "accuracy": {
                "string": "accuracy",
                "func": accuracy_score,
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "f1_score": {
                "string": "f1_score",
                "func": f1_score,
                "default_params": {"average": "weighted", "zero_division": 0},
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "log_loss": {
                "string": "log_loss",
                "func": log_loss,
                "more_is_better": False,
                "need_prob": True,
            },
            "precision": {
                "string": "precision",
                "func": precision_score,
                "default_params": {"average": "weighted", "zero_division": 0},
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "recall": {
                "string": "recall",
                "func": recall_score,
                "default_params": {"average": "weighted", "zero_division": 0},
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "roc_auc": {
                "string": "roc_auc",
                "func": roc_auc_score,
                "more_is_better": True,
                "need_prob": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "balanced_accuracy": {
                "string": "balanced_accuracy",
                "func": balanced_accuracy_score,
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
        },
        "multi_class": {
            "accuracy": {
                "string": "accuracy",
                "func": accuracy_score,
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "f1_score": {
                "string": "f1_score",
                "func": f1_score,
                "default_params": {"average": "micro"},
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "log_loss": {
                "string": "log_loss",
                "func": log_loss,
                "more_is_better": False,
                "need_prob": True,
            },
            "precision": {
                "string": "precision",
                "func": precision_score,
                "default_params": {"average": "micro"},
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "recall": {
                "string": "recall",
                "func": recall_score,
                "default_params": {"average": "micro"},
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "roc_auc": {
                "string": "roc_auc",
                "func": roc_auc_score,
                "default_params": {"average": "macro", "multi_class": "ovr"},
                "more_is_better": True,
                "need_prob": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
            "balanced_accuracy": {
                "string": "balanced_accuracy",
                "func": balanced_accuracy_score,
                "more_is_better": True,
                "format": table_styles.percentage_format,
                "round": 4,
            },
        },
        "regression": {
            "Explained Variance": {
                "string": "Explained Variance",
                "func": explained_variance_score,
                "more_is_better": True,
                "format": table_styles.percentage_format,
            },
            "MAPE": {
                "string": "MAPE",
                "func": mape,
                "more_is_better": False,
                "format": table_styles.percentage_format,
            },
            "WMAPE": {
                "string": "WMAPE",
                "func": wmape,
                "more_is_better": False,
                "format": table_styles.percentage_format,
            },
            "MAE": {
                "string": "MAE",
                "func": mean_absolute_error,
                "more_is_better": False,
            },
            "RMSE": {
                "string": "RMSE",
                "func": root_mean_squared_error,
                "more_is_better": False,
            },
            "R^2": {"string": "R^2", "func": r2_score, "more_is_better": False},
        },
    }
)
