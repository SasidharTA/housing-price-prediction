# import numpy as np
# import pandas as pd
# import tigerml.core.dataframe as td


def hampel_filter(input_series, window_size=3, n_sigmas=3, impute=False):

    k = 1.4826  # scale factor for Gaussian distribution
    # new_series = input_series.copy()

    # helper lambda function
    # MAD = lambda x: td.median(td.abs(x - td.median(x)))
    def MAD(x):
        return (x - x.median()).abs().median()

    rolling_median = input_series.rolling(window=2 * window_size, center=True).median()
    rolling_mad = k * input_series.rolling(window=2 * window_size, center=True).apply(
        MAD
    )
    diff = (input_series - rolling_median).abs()

    # indices = list(np.argwhere(diff > (n_sigmas * rolling_mad)).flatten())
    indices = diff[diff > (n_sigmas * rolling_mad)].index.tolist()
    if impute:
        input_series.loc[indices] = rolling_median[indices]

    return indices
