import random

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, seed, settings
from hypothesis.extra.pandas import columns, data_frames, range_indexes
from hypothesis.strategies import composite, floats, integers, sampled_from

from tigerml.eda.helpers import is_missing, split_sets


@composite
def tuple_fsm(draw):
    f = draw(floats(allow_nan=False, allow_infinity=False,))
    s = draw(sampled_from(["Red", "Blue", "Green"]))
    m = draw(sampled_from(["text", 5.27, 5]))
    return tuple([f, s, m])


@composite
def df_fsm(draw):
    df = draw(
        data_frames(
            index=range_indexes(min_size=0, max_size=100),
            columns=columns(["float_col", "string_col", "mixed_col"], dtype=float,),
            rows=tuple_fsm(),
        )
    )
    n = draw(integers(min_value=0, max_value=len(df)))
    # na_val = draw(sampled_from(["Red", np.nan, ""]))
    # na_val = random.choice(["Red", np.nan, "", 1, np.inf])
    # na_val = np.array(random.sample(["Red", np.nan, "", 1, np.inf, True]*n, n))
    for col in df.columns:
        row = random.sample(range(0, len(df)), n)
        for i in row:
            df.loc[i, [col]] = np.nan
    return df, n


@settings(max_examples=50, deadline=None, suppress_health_check=HealthCheck.all())
@given(test_df=df_fsm())
def test_is_missing(test_df):
    test_df, n = test_df
    data = np.array([n] * 3)
    expected_df = pd.Series(
        data, index=["float_col", "string_col", "mixed_col"], dtype="int64"
    )
    pd.testing.assert_series_equal(
        is_missing(test_df, na_values=[np.nan]).sum().astype("int64"), expected_df
    )


@composite
def tuple_with_dups(draw):
    f = draw(floats(allow_nan=True, allow_infinity=True,))
    s = draw(sampled_from(["Red", "Blue", "Green", np.nan, None]))
    m = draw(sampled_from(["text", 5.27, 5, np.nan, None]))
    return tuple([f, s, m, f, s, m])


@composite
def df_with_dups(draw):
    df = draw(
        data_frames(
            index=range_indexes(min_size=5),
            columns=columns(
                [
                    "float_col",
                    "string_col",
                    "mixed_col",
                    "float_col_2",
                    "string_col_2",
                    "mixed_col_2",
                ],
                dtype=float,
            ),
            rows=tuple_with_dups(),
        )
    )
    # df.loc[len(df)] = [1, 'Red', 'text', 1, 'Red', 'text']
    return df


@settings(max_examples=50, deadline=None)
@given(test_df=df_with_dups())
def test_split_sets(test_df):
    # expected_output = [[1, 4], [2, 5], [3, 6]]
    assert split_sets(test_df, test_df.columns, 0)
