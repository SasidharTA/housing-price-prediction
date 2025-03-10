"""Feature correlation."""
import numpy as np
import pandas as pd
import tigerml.core.dataframe as td
from tigerml.core.utils import get_num_cols, get_x_y_vars
from tigerml.core.utils.constants import SUMMARY_KEY_MAP


def compute_correlations(data, x_vars=None, y_vars=None):
    """Returns a correlation tables for x_vars and y_vars.

    x_vars and y_vars are taken from  self.data.

    Parameters
    ----------
    x_vars : list of variables for which we need to plot the correlation table.
    y_vars : list of variables for which we need to plot the correlation table.

    Returns
    -------
    Table :
        Containing the correlation of all variables with each other.
    """
    x_vars, y_vars, req_cols = get_x_y_vars(list(data.columns), x_vars, y_vars)
    df = data[req_cols]
    df = df[get_num_cols(df)]
    if not (df.empty):
        corr_df = df.corr()
        corr_df = corr_df.where(np.triu(np.ones(corr_df.shape)).astype(bool))
        c_df = corr_df.stack().reset_index()
        c_df = c_df.rename(
            columns=dict(
                zip(
                    list(c_df.columns),
                    [
                        SUMMARY_KEY_MAP.variable_1,
                        SUMMARY_KEY_MAP.variable_2,
                        SUMMARY_KEY_MAP.corr_coef,
                    ],
                )
            )
        )
        if len(x_vars) != len(req_cols) or len(y_vars) != len(req_cols):
            first_set = x_vars if len(x_vars) < len(y_vars) else y_vars
            second_set = x_vars if first_set == y_vars else y_vars
            second_col = (
                SUMMARY_KEY_MAP.variable_1
                if second_set == x_vars
                else SUMMARY_KEY_MAP.variable_2
            )
            c_df = c_df[
                (c_df[SUMMARY_KEY_MAP.variable_1].isin(first_set))
                | (c_df[SUMMARY_KEY_MAP.variable_2].isin(first_set))
            ]
            c_df_dup = c_df.rename(
                columns={
                    SUMMARY_KEY_MAP.variable_1: SUMMARY_KEY_MAP.variable_2,
                    SUMMARY_KEY_MAP.variable_2: SUMMARY_KEY_MAP.variable_1,
                }
            )
            c_df = td.concat([c_df, c_df_dup])
            c_df = c_df[c_df[second_col].isin(second_set)]
        c_df = c_df.loc[
            c_df[SUMMARY_KEY_MAP.variable_1] != c_df[SUMMARY_KEY_MAP.variable_2]
        ]
        c_df[SUMMARY_KEY_MAP.abs_corr_coef] = c_df[SUMMARY_KEY_MAP.corr_coef].abs()
        c_df.sort_values(SUMMARY_KEY_MAP.abs_corr_coef, ascending=False, inplace=True)
        c_df.reset_index(drop=True, inplace=True)
    else:
        c_df = pd.DataFrame()
    return c_df
