import logging
from tigerml.core.dataframe.dataframe import measure_time
from tigerml.core.utils._lib import DictObject
from tigerml.model_monitoring.config.threshold_options import THRESHOLD_OPTIONS

_LOGGER = logging.getLogger(__name__)


def update_summary_with_thresholds(threshold_options):
    summary_options = DictObject(
        {
            "target_drift": {"threshold_on": "psi"},
            "feature_drift_numerical": {"threshold_on": "psi"},
            "feature_drift_categorical": {"threshold_on": "psi"},
            "concept_drift_numerical": {"threshold_on": "dsi"},
            "concept_drift_categorical": {"threshold_on": "dsi"},
        }
    )
    _LOGGER.info("Created SUMMARY_OPTIONS dictionary")
    # for each "threshold_on" key in no_threshold_summary_dict,
    # do: auto pickup "threshold_value" from THRESHOLD_OPTIONS
    for summary_column in summary_options.keys():
        threshold_on = summary_options[summary_column]["threshold_on"]

        # To handle ks_stats, chisquare_pvalue
        if "_" in threshold_on:
            metric_name = threshold_on.split("_")[0]
            metric_type = threshold_on.split("_")[1]
            if (
                threshold_options.get(metric_name, {})
                .get("threshold", {})
                .get(metric_type)
            ):
                # Get threshold_value from THRESHOLD_OPTIONS
                threshold_value = threshold_options[metric_name]["threshold"][
                    metric_type
                ]
        else:
            if threshold_options.get(threshold_on):
                # Get threshold_value from THRESHOLD_OPTIONS
                threshold_value = threshold_options[threshold_on]["threshold"]

        summary_options[summary_column]["threshold_value"] = threshold_value
    _LOGGER.info(
        "Generated summary options dictionary from user given THRESHOLD_OPTIONS"
    )

    return summary_options


# This is the default SUMMARY_OPTIONS which gets generated from default THRESHOLD_OPTIONS
SUMMARY_OPTIONS = update_summary_with_thresholds(
    threshold_options=THRESHOLD_OPTIONS.copy()
)
