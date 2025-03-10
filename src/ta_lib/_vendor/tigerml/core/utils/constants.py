import numpy as np
import pandas as pd

from ._lib import DictObject

MIN_CUTOFF_FOR_KEY_HEURISTIC = 90
NA_VALUES = [np.NaN, pd.NaT, None, "NA"]
SUMMARY_KEY_MAP = DictObject(
    {
        "variable_names": "Variable Name",
        "num_unique": "No of Unique",
        "samples": "Samples",
        "num_missing": "No of Missing",
        "perc_missing": "Per of Missing",
        "is_numeric": "Is Numeric",
        "comment": "Comment",
        "dtype": "Datatype",
        "max_value": "Max",
        "min_value": "Min",
        "duplicate_col": "Duplicates",
        "num_values": "No of Values",
        "unique": "Unique",
        "mode_value": "Mode",
        "mode_freq": "Mode Freq",
        "mean_value": "Mean",
        "std_deviation": "Standard Deviation",
        "percentile_25": "25th percentile",
        "percentile_50": "Median",
        "percentile_75": "75th percentile",
        "variable_1": "Variable 1",
        "variable_2": "Variable 2",
        "corr_coef": "Corr Coef",
        "abs_corr_coef": "Abs Corr Coef",
    }
)
