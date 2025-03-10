from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis.extra.pandas import columns, data_frames, range_indexes
from hypothesis.strategies import (
    composite,
    data,
    floats,
    integers,
    lists,
    sampled_from,
    text,
)

import tigerml.core.dataframe as td


# -------------------------------------------------------------------------------
# -------------- creating composite strategies for testing ----------------------
@composite
def make_date_strings(draw, formatting_type, have_unambiguous_date="no"):
    ori_date = []
    str_date = []
    diff_formats = {
        1: ["str(day_val)+'-'+str(month_val)+'-'+str(year_val)", "%d-%m-%Y"],
        2: ["str(month_val)+'-'+str(day_val)+'-'+str(year_val)", "%m-%d-%Y"],
        3: ["str(year_val)+'-'+str(day_val)+'-'+str(month_val)", "%Y-%d-%m"],
        4: ["str(year_val)+'-'+str(month_val)+'-'+str(day_val)", "%Y-%m-%d"],
        5: ["str(day_val)+'-'+str(month_val)+'-'+str(year_val)[-2:]", "%d-%m-%y"],
        6: ["str(month_val)+'-'+str(day_val)+'-'+str(year_val)[-2:]", "%m-%d-%y"],
        7: ["str(year_val)[-2:]+'-'+str(day_val)+'-'+str(month_val)", "%y-%d-%m"],
        8: ["str(year_val)[-2:]+'-'+str(month_val)+'-'+str(day_val)", "%y-%m-%d"],
    }

    if have_unambiguous_date == "first":
        year_val = 1997
        month_val = 5
        day_val = 27
        ori_date += [pd.Timestamp(year=year_val, month=month_val, day=day_val)]
        str_date += [eval(diff_formats[formatting_type][0])]

    for i in range(1, 6):
        year_val = draw(integers(min_value=1995, max_value=2019))
        month_val = draw(integers(min_value=1, max_value=12))
        day_val = draw(integers(min_value=1, max_value=27))
        ori_date += [pd.Timestamp(year=year_val, month=month_val, day=day_val)]
        str_date += [eval(diff_formats[formatting_type][0])]

    if have_unambiguous_date == "last":
        year_val = 1997
        month_val = 5
        day_val = 27
        ori_date += [pd.Timestamp(year=year_val, month=month_val, day=day_val)]
        str_date += [eval(diff_formats[formatting_type][0])]

    ts_df = pd.DataFrame(
        data={
            "ori_date": ori_date,
            "str_date " + diff_formats[formatting_type][1]: str_date,
        }
    )

    return ts_df


# Function to give meaningful test_ids for all the test combinations
def id_func(param):
    id_dict = {
        1: "%d-%m-%Y",
        2: "%m-%d-%Y",
        3: "%Y-%d-%m",
        4: "%Y-%m-%d",
        5: "%d-%m-%y",
        6: "%m-%d-%y",
        7: "%y-%d-%m",
        8: "%y-%m-%d",
        "first": "unambiguous_date at first",
        "last": "unambiguous_date at last",
        "no": "no unambiguous_date",
    }
    return id_dict[param]


@pytest.mark.parametrize("format_type", [1], ids=id_func)  # 2, 3, 4, 5, 6, 7, 8],
@pytest.mark.parametrize(
    "unambiguous_date",
    ["first", "last"],
    # TODO: Fix the failing test case auto-no unambiguous_date-%d-%m-%Y
    ids=id_func,
)
@pytest.mark.parametrize("convert_datetime", ["auto", "explicit"])
@settings(max_examples=50, deadline=None)
@given(test_df_=data())
def test_convert_datetimes(test_df_, convert_datetime, format_type, unambiguous_date):

    test_df = test_df_.draw(
        make_date_strings(
            formatting_type=format_type, have_unambiguous_date=unambiguous_date
        )
    )
    col_nm = test_df.columns[1]
    tdf = td.DataFrame(test_df.copy())
    if convert_datetime == "auto":
        tdf = tdf.convert_datetimes()
    else:
        tdf = tdf.convert_datetimes(format_dict={col_nm: id_func(format_type)})
    reformatted_list = tdf[col_nm]._data.tolist()
    original_list = test_df["ori_date"].tolist()

    # col_nm = test_df.columns[1]
    # test_df['reformatted_date'] = pd.to_datetime(test_df[col_nm], format='%d-%m-%y')
    # reformatted_list = test_df['reformatted_date'].tolist()
    # original_list = test_df['ori_date'].tolist()

    assert original_list == reformatted_list
