import logging
from tigerml.core.dataframe.dataframe import measure_time
from tigerml.core.utils._lib import DictObject

_LOGGER = logging.getLogger(__name__)

# for any metric which doesnt have {High,Moderate,Low} give Red Thresholds (i.e if pvalue < 0.05 , then it is in RED color)
THRESHOLD_OPTIONS = DictObject(
    {
        "chisquare": {
            "threshold": {"stats": ">5", "pvalue": "<0.05"},
            "range": {"stats": (1, 100), "pvalue": (0, 1)},
        },
        "anderson": {
            "threshold": {"stats": ">5", "pvalue": "<0.05"},
            "range": {"stats": (1, 100), "pvalue": (0, 1)},
        },
        "ks": {
            "threshold": {"stats": ">5", "pvalue": "<0.05"},
            "range": {"stats": (1, 100), "pvalue": (0, 1)},
        },
        "psi": {
            "threshold": {"High": ">=0.2", "Moderate": "0.1-0.2", "Low": "<0.1"},
            "range": (0, 1),
        },
        "dsi": {
            "threshold": {"High": ">=0.2", "Moderate": "0.1-0.2", "Low": "<0.1"},
            "range": (0, 1),
        },
        "kldivergence": {
            "threshold": {"High": ">=0.1", "Moderate": "0.05-0.1", "Low": "<0.05"},
            "range": (0, 1),
        },
    }
)


def update_threshold_options(thresholds=None):
    threshold_options = THRESHOLD_OPTIONS.copy()
    _LOGGER.info("Created glossary dict")
    if thresholds:
        # Auto Generate the "THRESHOLD_OPTIONS" from user given "thresholds"
        for metric in thresholds:
            if thresholds.get(metric, {}).get("threshold"):
                for threshold_on, threshold_value in thresholds[metric][
                    "threshold"
                ].items():
                    if (
                        threshold_options.get(metric, {})
                        .get("threshold", {})
                        .get(threshold_on)
                    ):
                        # FIXME: Validation: check if the metric's threshold_value is within the metric's range
                        threshold_options[metric]["threshold"][
                            threshold_on
                        ] = threshold_value
    _LOGGER.info("Auto Generated the 'THRESHOLD_OPTIONS' from user given 'thresholds'")

    return threshold_options
