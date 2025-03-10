from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis.extra.pandas import column, columns, data_frames, range_indexes
from hypothesis.strategies import (
    composite,
    data,
    datetimes,
    floats,
    integers,
    lists,
    sampled_from,
    tuples,
)

from tigerml.core.utils import NA_VALUES
from tigerml.eda import Analyser


# ----------------------------------------------------------------------------------
# ---------- composite strategy for testing column identifying functions -----------
@composite
def make_named_data(draw, col):
    df = draw(
        data_frames(
            index=range_indexes(min_size=5),
            columns=[
                column(
                    "Explicit_obj_col_with_nums",
                    elements=sampled_from([5, 5.3]),
                    dtype=object,
                ),
                column(
                    "Explicit_obj_col_with_mix",
                    elements=sampled_from([5, 5.3, "Text"]),
                    dtype=object,
                ),
                column("Implicit_nums_col", elements=sampled_from([5, 5.3])),
                column(
                    "Implicit_mix_col",
                    elements=sampled_from([5, 5.3, "Text"])
                    | datetimes(
                        min_value=pd.Timestamp("1/1/1995"),
                        max_value=pd.Timestamp("1/1/2005"),
                    ),
                ),
                column("Implicit_bool_col", elements=sampled_from([True, False])),
                column(
                    "Explicit_obj_col_with_bool",
                    elements=sampled_from([True, False]),
                    dtype=object,
                ),
                column(
                    "Implicit_dt_col",
                    datetimes(
                        min_value=pd.Timestamp("1/1/1995"),
                        max_value=pd.Timestamp("1/1/2005"),
                    ),
                ),
                column(
                    "Explicit_obj_col_with_dt",
                    elements=datetimes(
                        min_value=pd.Timestamp("1/1/1995"),
                        max_value=pd.Timestamp("31/12/2005"),
                    ),
                    dtype=object,
                ),
            ],
        )
    )
    return df[col]


# WIP: A decorator for printing falsifying examples
def print_the_example(func):
    def inner(test_df):
        try:
            func(test_df)
            print("Successful example", test_df)
        except AssertionError:
            print("Failing example", test_df)
        return func(test_df)

    return inner


# --------------------------------------------------------------------------------------
# ---------------------- get_numeric_columns testing ---------------------------------
@pytest.mark.parametrize(
    "selected_col",
    [
        ["Explicit_obj_col_with_nums"],
        ["Explicit_obj_col_with_mix"],
        ["Implicit_nums_col"],
        ["Implicit_mix_col"],
    ],
    ids=[
        "Explicit_obj_col_with_nums",
        "Explicit_obj_col_with_mix",
        "Implicit_nums_col",
        "Implicit_mix_col",
    ],
)
@settings(max_examples=20, deadline=None)
@given(test_df=data())
def test_get_numeric_columns(test_df, selected_col):
    test_df_ = test_df.draw(make_named_data(col=selected_col))
    an_obj = Analyser(test_df_)
    computed_list = an_obj.get_numeric_columns()
    expected_list = []
    old_col_nm = selected_col[0]
    new_col_nm = old_col_nm + "_bool"
    test_df_[new_col_nm] = test_df_.apply(
        lambda x: True
        if type(x[old_col_nm]) in [int, float, np.int64, np.float64]
        else False,
        axis=1,
    )
    if (
        (len(test_df_.groupby([new_col_nm], sort=False).count()) == 1)
        & test_df_[new_col_nm][0]
        == True  # noqa
    ):
        expected_list = [old_col_nm]
    assert computed_list == expected_list


# --------------------------------------------------------------------------------------
# ---------------------- get_dt_columns testing ---------------------------------
@pytest.mark.parametrize(
    "selected_col, exp_output",
    [
        (["Implicit_dt_col"], ["Implicit_dt_col"]),
        (["Explicit_obj_col_with_dt"], ["Explicit_obj_col_with_dt"]),
        (["Explicit_obj_col_with_nums"], []),
        (["Explicit_obj_col_with_mix"], []),
    ],
    ids=[
        "Implicit_dt_col",
        "Explicit_obj_col_with_dt",
        "Explicit_obj_col_with_nums",
        "Explicit_obj_col_with_mix",
    ],
)
@settings(max_examples=20, deadline=None)
@given(test_df=data())
def test_dt_columns(test_df, selected_col, exp_output):
    test_df_ = test_df.draw(make_named_data(col=selected_col))
    an_obj = Analyser(test_df_)
    an_obj.data = an_obj.data.convert_datetimes()
    if selected_col[0] == "Explicit_obj_col_with_dt":
        an_obj.data["Explicit_obj_col_with_dt"] = an_obj.data[
            "Explicit_obj_col_with_dt"
        ].apply(
            lambda x: pd._libs.tslib.Timestamp(
                str(x.year)
                + "-"
                + str(x.month)
                + "-"
                + str(x.day)
                + " "
                + str(x.hour)
                + ":"
                + str(x.minute)
                + ":"
                + str(x.second)
                + "."
                + str(x.microsecond)
            )
        )
    computed_list = an_obj.get_dt_columns()
    assert computed_list == exp_output


# --------------------------------------------------------------------------------------
# ---------------------- get_bool_columns testing ---------------------------------
@pytest.mark.parametrize(
    "selected_col, exp_output",
    [
        (["Implicit_bool_col"], ["Implicit_bool_col"]),
        (["Explicit_obj_col_with_bool"], ["Explicit_obj_col_with_bool"]),
    ],
    ids=["Implicit_bool_col", "Explicit_obj_col_with_bool"],
)
@settings(max_examples=20, deadline=None)
@given(test_df=data())
def test_bool_columns(test_df, selected_col, exp_output):
    test_df_ = test_df.draw(make_named_data(col=selected_col))
    an_obj = Analyser(test_df_)
    computed_list = an_obj.get_bool_columns()
    assert computed_list == exp_output


# ----------------------------------------------------------------------------
# ---------------------- get_cat_columns testing -----------------------------
@pytest.mark.parametrize(
    "selected_col, exp_output",
    [
        (["Explicit_obj_col_with_nums"], []),
        # FIXME: Following test cases are failing with removal of mixed dtypes
        # (["Explicit_obj_col_with_mix"], "evaluate"),
        # (["Implicit_mix_col"], "evaluate"),
        (["Implicit_nums_col"], []),
        (["Implicit_bool_col"], []),
        (["Explicit_obj_col_with_bool"], []),
        (["Implicit_dt_col"], []),
        (["Explicit_obj_col_with_dt"], []),
    ],
    ids=[
        "Explicit_obj_col_with_nums",
        # FIXME: Following test cases are failing with removal of mixed dtypes
        # "Explicit_obj_col_with_mix",
        # "Implicit_mix_col",
        "Implicit_nums_col",
        "Implicit_bool_col",
        "Explicit_obj_col_with_bool",
        "Implicit_dt_col",
        "Explicit_obj_col_with_dt",
    ],
)
@settings(max_examples=20, deadline=None)
@given(test_df=data())
def test_cat_columns(test_df, selected_col, exp_output):
    test_df_ = test_df.draw(make_named_data(col=selected_col))
    an_obj = Analyser(test_df_)
    an_obj.data = an_obj.data.convert_datetimes()
    if selected_col[0] == "Explicit_obj_col_with_dt":
        an_obj.data["Explicit_obj_col_with_dt"] = an_obj.data[
            "Explicit_obj_col_with_dt"
        ].apply(
            lambda x: pd._libs.tslib.Timestamp(
                str(x.year)
                + "-"
                + str(x.month)
                + "-"
                + str(x.day)
                + " "
                + str(x.hour)
                + ":"
                + str(x.minute)
                + ":"
                + str(x.second)
                + "."
                + str(x.microsecond)
            )
        )
    computed_list = an_obj.get_cat_columns()
    if not exp_output == "evaluate":
        expected_list = exp_output
    else:
        expected_list = []
        old_col_nm = selected_col[0]
        new_col_nm = old_col_nm + "_type"
        test_df_[new_col_nm] = test_df_.apply(lambda x: type(x[old_col_nm]), axis=1)
        if (len(test_df_.groupby([new_col_nm], sort=False).count()) == 1) & (
            test_df_[new_col_nm][0] in [str]
        ):
            expected_list = [old_col_nm]
        if len(test_df_.groupby([new_col_nm], sort=False).count()) > 1:
            expected_list = [old_col_nm]
        new_col_nm = old_col_nm + "_bool"
        test_df_[new_col_nm] = test_df_.apply(
            lambda x: True
            if type(x[old_col_nm]) in [int, float, np.int64, np.float64]
            else False,
            axis=1,
        )
        if (
            (len(test_df_.groupby([new_col_nm], sort=False).count()) == 1)
            & test_df_[new_col_nm][0]
            == True  # noqa
        ):
            expected_list = []
    assert computed_list == expected_list


# --------------------------------------------------------------------------------
# ----------- composite strategy for testing calculate_all_segment ---------------
@composite
def make_indexed_data(draw, segment_by):
    col_len = draw(integers(min_value=len(segment_by) + 2, max_value=10))
    row_len = draw(integers(min_value=5, max_value=100))

    def get_sampling_list(col_num, row_size):
        return [
            "Seg" + str(col_num) + "_Type" + str(i) for i in range(row_size)
        ] + NA_VALUES

    df_dict = {}
    for col in range(col_len):
        col_name = "col_" + str(col)
        df_dict[col_name] = draw(
            lists(
                elements=sampled_from(get_sampling_list(col, int(row_len / 2))),
                min_size=row_len,
                max_size=row_len,
            )
        )
    df = pd.DataFrame(df_dict)
    df["segments"] = df.apply(
        lambda x: "_".join([str(x[col_nm]) for col_nm in segment_by]), axis=1
    )
    return df


# -------------------------------------------------------------------------------
# ------------------ _calculate_all_segments testing -----------------------------
@settings(max_examples=20, deadline=None)
@given(test_df=data(), no_of_segments=integers(min_value=1, max_value=7))
def test_calculate_all_segments(test_df, no_of_segments):
    col_names = ["col_" + str(i) for i in range(no_of_segments)]
    test_df_ = test_df.draw(make_indexed_data(segment_by=col_names))
    an_obj = Analyser(test_df_, segment_by=col_names)
    computed_list = an_obj.all_segments
    c_list = []
    for i in computed_list:
        x = 0
        for j in i:
            if type(j) != str:
                x = 1
        if x == 0:
            c_list.append(i)
    computed_list = c_list
    expected_list = test_df_[col_names + ["segments"]].dropna()
    expected_list = expected_list["segments"].unique().tolist()
    assert len(computed_list) == len(expected_list)
