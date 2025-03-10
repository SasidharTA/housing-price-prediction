import logging
import pandas as pd
from tigerml.core.dataframe.dataframe import measure_time
from tigerml.model_monitoring.config.threshold_options import THRESHOLD_OPTIONS

_LOGGER = logging.getLogger(__name__)


def update_glossary_df_with_thresholds(threshold_options):
    glossary_dict = {
        "psi": {
            "Name of the Metrics": "Population Stability Index",
            "Description": "PSI compares the distribution of a scoring variable (predicted probability) in scoring data set to a testing data set that was used to develop the model.",
        },
        "dsi": {
            "Name of the Metrics": "Dependency Stability Index",
            "Description": "DSI compares the distribution of a scoring variable (predicted probability) in scoring data set to a testing data set that was used to develop the model.",
        },
        # "kldivergence": {
        #     "Name of the Metrics": "Kullback–Leibler divergence",
        #     "Description": "KL_Div (also called relative entropy), is a measure of how one probability distribution is different from a second, reference probability distribution.",
        # },
        "chisquare": {
            "Name of the Metrics": "Chi Square Test",
            "Description": "Chi-square test of independence of variables in a contingency table. The null hypothesis is that both the samples come from same distribution",
        },
        "ks": {
            "Name of the Metrics": "Kolmogorov-Smirnov",
            "Description": """This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.  The alternative hypothesis
    can be either 'two-sided' (default), 'less' or 'greater'.""",
        },
        #     "anderson": {
        #         "Name of the Metrics": "Anderson-Darling Test",
        #         "Description": """The k-sample Anderson-Darling test is a modification of the
        # one-sample Anderson-Darling test. It tests the null hypothesis
        # that k-samples are drawn from the same population without having
        # to specify the distribution function of that population. The
        # critical values depend on the number of samples.""",
        #     },
    }

    _LOGGER.info("Created glossary dict")
    # for each metric(key) in no_threshold_summary_dict,
    # do: auto pickup "Thresholds" from THRESHOLD_OPTIONS
    for metric in glossary_dict.keys():
        # Check if metric present in THRESHOLD_OPTIONS and get threshold_value
        if threshold_options.get(metric):
            threshold_value = threshold_options[metric]["threshold"]
            glossary_dict[metric]["Thresholds"] = threshold_value
    _LOGGER.info("Set up of thresholds for various metrics done")

    # Convert this dict to pd.DataFrame
    # glossary_dataframe = pd.DataFrame(glossary_dict).T
    glossary_dataframe = pd.DataFrame.from_dict(glossary_dict, orient="index")
    _LOGGER.info("Glossary dataframe created")
    return glossary_dataframe


# This is the default GLOSSARY_DATAFRAME which gets generated from default THRESHOLD_OPTIONS
GLOSSARY_DATAFRAME = update_glossary_df_with_thresholds(
    threshold_options=THRESHOLD_OPTIONS.copy()
)
