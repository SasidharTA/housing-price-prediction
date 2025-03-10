import holoviews
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
from hypothesis.strategies import (
    composite,
    data,
    datetimes,
    floats,
    integers,
    sampled_from,
)

import tigerml.core.dataframe as td
from tigerml.core.utils.constants import NA_VALUES
from tigerml.eda.segmented import SegmentedEDAReport


# --------------------------------------------------------------------------------------
# ----------------- creating composite strategies for testing -------------------------
# generates a dataframe with specified no. of segmenting columns and specified type
# of feature columns. The segment column can be set to have NaN values.
@composite
def make_df_with_select_cols(draw, non_seg_cols, non_empty_row="have"):
    date_list = [pd.Timestamp("1/1/1995"), pd.Timestamp("1/1/2005"), pd.NaT]
    column_list = []
    if "Num_col" in non_seg_cols:
        column_list += [
            column("Num_col", elements=floats(allow_infinity=True, allow_nan=True))
        ]
    if "Num_col2" in non_seg_cols:
        column_list += [
            column("Num_col2", elements=floats(allow_infinity=True, allow_nan=True))
        ]
    if "Num_col3" in non_seg_cols:
        column_list += [
            column("Num_col3", elements=floats(allow_infinity=True, allow_nan=True))
        ]
    if "Cat_col" in non_seg_cols:
        column_list += [
            column(
                "Cat_col", elements=sampled_from(["Red", "Blue", "Green", "NA", None])
            )
        ]
    if "Dt_col" in non_seg_cols:
        column_list += [column("Dt_col", elements=sampled_from(date_list))]
    if "Bool_col" in non_seg_cols:
        column_list += [column("Bool_col", elements=sampled_from([True, False, None]))]

    # df = draw(data_frames(index=range_indexes(min_size=n_rows, max_size=n_rows + 300),
    #                       columns=column_list))
    df = draw(data_frames(index=range_indexes(min_size=3), columns=column_list))
    if non_empty_row == "have":
        non_empty_row = True
    else:
        non_empty_row = False  # draw(sampled_from([True, False]))
    if non_empty_row:
        # last_ind = len(df)-1
        # if 'Num_col' in non_seg_cols:
        #     df['Num_col'][last_ind] = 5.2
        # if 'Cat_col' in non_seg_cols:
        #     df['Cat_col'][last_ind] = 'Red'
        # if 'Dt_col' in non_seg_cols:
        #     df['Dt_col'][last_ind] = pd.Timestamp('1/1/1995')
        # if 'Bool_col' in non_seg_cols:
        #     df['Bool_col'][last_ind] = False

        for i in range(3):
            new_row = {}
            if "Num_col" in non_seg_cols:
                new_row["Num_col"] = i + 0.2
            if "Num_col2" in non_seg_cols:
                new_row["Num_col2"] = i + 0.2
            if "Num_col3" in non_seg_cols:
                new_row["Num_col3"] = i + 0.2
            if "Cat_col" in non_seg_cols:
                new_row["Cat_col"] = "Red"
            if "Dt_col" in non_seg_cols:
                new_row["Dt_col"] = pd.Timestamp("1/1/1995")
            if "Bool_col" in non_seg_cols:
                new_row["Bool_col"] = False
            df = df.append(pd.Series(new_row), ignore_index=True)
    return df


@composite
def df_with_select_col_and_segments(
    draw,
    non_seg_cols,
    non_empty_row="have",
    have_segments=0,
    seg_has_na=False,
    shuffle_seg="no",
):

    if have_segments == 0:
        data_df_all = draw(
            make_df_with_select_cols(
                non_seg_cols=non_seg_cols, non_empty_row=non_empty_row
            )
        )

    elif have_segments == 1:
        segments = [["L1_S1"], ["L1_S2"], ["L1_S3"]]
        frames = []
        for segment in segments:
            data_df_all = draw(
                make_df_with_select_cols(
                    non_seg_cols=non_seg_cols, non_empty_row=non_empty_row
                )
            )
            frames += [
                data_df_all.join(
                    pd.DataFrame([segment] * len(data_df_all), columns=["Lev_1"])
                )
            ]
        data_df_all = pd.concat(frames, ignore_index=True)

    elif have_segments == 2:
        # segments = [['L1_S1', 'L2_S1'], ['L1_S1', 'L2_S2'],
        #             ['L1_S2', 'L2_S1'],
        #             ['L1_S3', 'L2_S1'], ['L1_S3', 'L2_S2'], ['L1_S3', 'L2_S3']]
        # segments = [['L1_S1', 'L2_S1'], ['L1_S1', 'L2_S2'],
        #             ['L1_S2', 'L2_S1'], ['L1_S2', 'L2_S2'], ['L1_S2', 'L2_S3']]
        segments = [
            ["L1_S1", "L2_S1"],
            ["L1_S2", "L2_S1"],
            ["L1_S2", "L2_S2"],
            ["L1_S2", "L2_S3"],
        ]
        frames = []
        for segment in segments:
            data_df_all = draw(
                make_df_with_select_cols(
                    non_seg_cols=non_seg_cols, non_empty_row=non_empty_row
                )
            )
            frames += [
                data_df_all.join(
                    pd.DataFrame(
                        [segment] * len(data_df_all), columns=["Lev_1", "Lev_2"]
                    )
                )
            ]
        data_df_all = pd.concat(frames, ignore_index=True)

    elif have_segments == 3:
        segments = [
            ["L1_S1", "L2_S1", "L3_S1"],
            ["L1_S1", "L2_S1", "L3_S2"],
            ["L1_S1", "L2_S2", "L3_S1"],
        ]
        frames = []
        for segment in segments:
            data_df_all = draw(
                make_df_with_select_cols(
                    non_seg_cols=non_seg_cols, non_empty_row=non_empty_row
                )
            )
            frames += [
                data_df_all.join(
                    pd.DataFrame(
                        [segment] * len(data_df_all),
                        columns=["Lev_1", "Lev_2", "Lev_3"],
                    )
                )
            ]
        data_df_all = pd.concat(frames, ignore_index=True)

    # na_val = 'NA'
    # na_val = None
    na_val = draw(sampled_from([None, "NA"]))
    generate_index = integers(min_value=0, max_value=len(data_df_all) - 1)
    if seg_has_na:
        if have_segments == 1:
            ind = draw(generate_index)
            data_df_all["Lev_1"][ind] = na_val
        elif have_segments == 2:
            ind = draw(generate_index)
            data_df_all["Lev_1"][ind] = na_val
            ind = draw(generate_index)
            data_df_all["Lev_2"][ind] = na_val
        elif have_segments == 3:
            ind = draw(generate_index)
            data_df_all["Lev_1"][ind] = na_val
            ind = draw(generate_index)
            data_df_all["Lev_2"][ind] = na_val
            ind = draw(generate_index)
            data_df_all["Lev_3"][ind] = na_val

    if shuffle_seg == "yes":
        data_df_all = data_df_all.sample(frac=1).reset_index(drop=True)
    return data_df_all


# Function to give meaningful test_ids for all the test combinations
def id_func(param):
    id_dict = {
        "1": "1_lev_segment",
        "2": "2_lev_segment",
        "3": "3_lev_segment",
        "True": "Seg_with_na",
        "False": "Seg_without_na",
        "Mixed_df": "Mixed_df",
        "Only_nums_df": "Only_nums_df",
        "Only_cat_df": "Only_cat_df",
        "Only_dt_df": "Only_dt_df",
        "Only_bool_df": "Only_bool_df",
        "T1": "Test_for_return",
        "T2": "Test_for_UniqueSegments",
        "T3": "Test_for_NanValues",
        "T4": "Test_for_PlotType",
        "Q_T": "quick=True",
        "Q_F": "quick=False",
        "yes": "Shuffled_segments",
        "no": "Sorted_segments",
        "have": "Segments_have_NonEmptyRow",
        "not_have": "Segments_not_have_NonEmptyRow",
    }
    if type(param) == list:
        if len(param) > 1:
            return "[" + ",".join(param) + "]"
        else:
            return param[0]
    return id_dict[str(param)]


# Fixture for assigning the levels of segmenting (called by @pytest.mark.parametrize in each test case)
@pytest.fixture(params=[1, 2, 3])
def select_segmenting_level(request):
    lev_dict = {1: ["Lev_1"], 2: ["Lev_1", "Lev_2"], 3: ["Lev_1", "Lev_2", "Lev_3"]}
    return lev_dict[request.param]


# Fixture for assigning the type of feature columns (called by @pytest.mark.parametrize in each test case)
@pytest.fixture(
    params=["Mixed_df", "Only_nums_df", "Only_cat_df", "Only_dt_df", "Only_bool_df"]
)
def df_type(request):
    df_type_dict = {
        "Mixed_df": ["Num_col", "Cat_col", "Dt_col", "Bool_col"],
        "Only_nums_df": ["Num_col"],
        "Only_cat_df": ["Cat_col"],
        "Only_dt_df": ["Dt_col"],
        "Only_bool_df": ["Bool_col"],
    }
    return df_type_dict[request.param]


@pytest.fixture(params=["Q_T", "Q_F"])
def quick_state(request):
    quick_state_dict = {"Q_T": True, "Q_F": False}
    return quick_state_dict[request.param]


""" Replace the first two parametrize decorators for each test cases to do exhaustive testing.
    Takes longer time.
@pytest.mark.parametrize("select_segmenting_level",
                         [1, 2, 3],
                         ids=id_func, indirect=True)
@pytest.mark.parametrize("df_type",
                         ['Mixed_df', 'Only_nums_df', 'Only_cat_df', 'Only_dt_df', 'Only_bool_df'],
                         ids=id_func, indirect=True)
"""


# --------------------------------------------------------------------------------------
# ---------------------- missing_per_segment testing ---------------------------------
@pytest.mark.parametrize("shuffle_seg", ["no", "yes"], ids=id_func)
@pytest.mark.parametrize("have_non_empty_row", ["have", "not_have"], ids=id_func)
@pytest.mark.parametrize("segment_property", [False, True], ids=id_func)
@pytest.mark.parametrize("df_type", ["Mixed_df"], ids=id_func, indirect=True)
@pytest.mark.parametrize(
    "select_segmenting_level", [1, 2, 3], ids=id_func, indirect=True
)
@pytest.mark.parametrize("test_type", ["T1", "T2", "T3"], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_missing_per_segment(
    test_df,
    select_segmenting_level,
    df_type,
    segment_property,
    test_type,
    shuffle_seg,
    have_non_empty_row,
):
    test_df_ = test_df.draw(
        df_with_select_col_and_segments(
            have_segments=len(select_segmenting_level),
            non_seg_cols=df_type,
            seg_has_na=segment_property,
            shuffle_seg=shuffle_seg,
            non_empty_row=have_non_empty_row,
        )
    )
    seg_eda_obj = SegmentedEDAReport(test_df_, segment_by=select_segmenting_level)
    if test_type == "T1":
        # Testing if the function doesn't raise any exception
        seg_eda_obj.missing_per_segment()
    elif test_type == "T2":
        # Testing if the output df has all the relevant segments
        calculated_df = seg_eda_obj.missing_per_segment()
        expected_segments = []
        calculated_segments = []
        if calculated_df.shape[0] > 0:
            expected_segments = (
                calculated_df[select_segmenting_level]
                .replace(NA_VALUES, np.nan)
                .dropna()
                .drop_duplicates()
                .values.tolist()
            )
            calculated_segments = calculated_df[select_segmenting_level].values.tolist()
        assert expected_segments == calculated_segments
    else:
        # Testing if the output df has no NaN values
        calculated_df = seg_eda_obj.missing_per_segment()
        cols = [
            col for col in calculated_df.columns if col not in select_segmenting_level
        ]
        # if calculated_df.shape[0] > 0:
        non_seg_values = calculated_df[cols].copy()
        non_seg_values = non_seg_values.replace(NA_VALUES, np.nan)
        assert non_seg_values.isnull().sum().sum() == 0


# --------------------------------------------------------------------------------------
# ---------------------- rows_per_segment testing ---------------------------------
@pytest.mark.parametrize("shuffle_seg", ["no", "yes"], ids=id_func)
@pytest.mark.parametrize("have_non_empty_row", ["have", "not_have"], ids=id_func)
@pytest.mark.parametrize("segment_property", [False, True], ids=id_func)
@pytest.mark.parametrize("df_type", ["Mixed_df"], ids=id_func, indirect=True)
@pytest.mark.parametrize(
    "select_segmenting_level", [1, 2, 3], ids=id_func, indirect=True
)
@pytest.mark.parametrize("test_type", ["T1", "T2", "T3"], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_rows_per_segment(
    test_df,
    select_segmenting_level,
    df_type,
    segment_property,
    test_type,
    shuffle_seg,
    have_non_empty_row,
):
    test_df_ = test_df.draw(
        df_with_select_col_and_segments(
            have_segments=len(select_segmenting_level),
            non_seg_cols=df_type,
            seg_has_na=segment_property,
            shuffle_seg=shuffle_seg,
            non_empty_row=have_non_empty_row,
        )
    )
    seg_eda_obj = SegmentedEDAReport(test_df_, segment_by=select_segmenting_level)
    if test_type == "T1":
        # Testing if the function doesn't raise any exception
        seg_eda_obj.rows_per_segment()
    elif test_type == "T2":
        # Testing if the output df has all the relevant segments
        calculated_df = seg_eda_obj.rows_per_segment()
        expected_segments = []
        calculated_segments = []
        if calculated_df.shape[0] > 0:
            expected_segments = (
                calculated_df[select_segmenting_level]
                .replace(NA_VALUES, np.nan)
                .dropna()
                .drop_duplicates()
                .values.tolist()
            )
            calculated_segments = calculated_df[select_segmenting_level].values.tolist()
        assert expected_segments == calculated_segments
    else:
        # Testing if the output df has no NaN values
        calculated_df = seg_eda_obj.rows_per_segment()
        cols = [
            col for col in calculated_df.columns if col not in select_segmenting_level
        ]
        non_seg_values = calculated_df[cols].copy()
        non_seg_values = non_seg_values.replace(NA_VALUES, np.nan)
        assert non_seg_values.isnull().sum().sum() == 0


# --------------------------------------------------------------------------------------
# ---------------------- outliers_per_segment testing ---------------------------------
@pytest.mark.parametrize("shuffle_seg", ["no", "yes"], ids=id_func)
@pytest.mark.parametrize("have_non_empty_row", ["have", "not_have"], ids=id_func)
@pytest.mark.parametrize("segment_property", [False, True], ids=id_func)
@pytest.mark.parametrize("df_type", ["Mixed_df"], ids=id_func, indirect=True)
@pytest.mark.parametrize(
    "select_segmenting_level", [1, 2, 3], ids=id_func, indirect=True
)
@pytest.mark.parametrize("test_type", ["T1", "T2", "T3"], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_outliers_per_segment(
    test_df,
    select_segmenting_level,
    df_type,
    segment_property,
    test_type,
    shuffle_seg,
    have_non_empty_row,
):
    test_df_ = test_df.draw(
        df_with_select_col_and_segments(
            have_segments=len(select_segmenting_level),
            non_seg_cols=df_type,
            seg_has_na=segment_property,
            shuffle_seg=shuffle_seg,
            non_empty_row=have_non_empty_row,
        )
    )
    seg_eda_obj = SegmentedEDAReport(test_df_, segment_by=select_segmenting_level)

    if test_type == "T1":
        # Testing if the function doesn't raise any exception
        seg_eda_obj.outliers_per_segment()
    elif test_type == "T2":
        # Testing if the output df has all the relevant segments
        calculated_df = seg_eda_obj.outliers_per_segment()
        expected_segments = []
        calculated_segments = []
        if calculated_df.shape[0] > 0:
            expected_segments = (
                calculated_df[select_segmenting_level]
                .replace(NA_VALUES, np.nan)
                .dropna()
                .drop_duplicates()
                .values.tolist()
            )
            calculated_segments = calculated_df[select_segmenting_level].values.tolist()
        assert expected_segments == calculated_segments
    else:
        # Testing if the output df has no NaN values
        calculated_df = seg_eda_obj.outliers_per_segment()
        cols = [
            col for col in calculated_df.columns if col not in select_segmenting_level
        ]
        non_seg_values = calculated_df[cols].copy()
        non_seg_values = non_seg_values.replace(NA_VALUES, "")
        assert non_seg_values.isnull().sum().sum() == 0


# --------------------------------------------------------------------------------------
# ---------------------- segments_summary testing ---------------------------------
# Already tested in test_DP_base.py while testing _calculate_all_segments()


# --------------------------------------------------------------------------------------
# ---------------------- numeric_summary testing ---------------------------------
@pytest.mark.parametrize("quick_state", ["Q_F", "Q_T"], ids=id_func, indirect=True)
@pytest.mark.parametrize("shuffle_seg", ["no", "yes"], ids=id_func)
@pytest.mark.parametrize("have_non_empty_row", ["have", "not_have"], ids=id_func)
@pytest.mark.parametrize("segment_property", [False, True], ids=id_func)
@pytest.mark.parametrize("df_type", ["Only_nums_df"], ids=id_func, indirect=True)
@pytest.mark.parametrize(
    "select_segmenting_level", [1, 2, 3], ids=id_func, indirect=True
)
@pytest.mark.parametrize("test_type", ["T1", "T2"], ids=id_func)
@settings(max_examples=1, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=data()
)  # For testing in CI max_examples is set to 1, Can be increased upto 200 for developers
def test_numeric_summary(
    test_df,
    select_segmenting_level,
    df_type,
    segment_property,
    test_type,
    shuffle_seg,
    have_non_empty_row,
    quick_state,
    request,
):
    test_df_ = td.DataFrame(
        test_df.draw(
            df_with_select_col_and_segments(
                have_segments=len(select_segmenting_level),
                non_seg_cols=df_type,
                seg_has_na=segment_property,
                shuffle_seg=shuffle_seg,
                non_empty_row=have_non_empty_row,
            )
        )
    )
    seg_eda_obj = SegmentedEDAReport(test_df_, segment_by=select_segmenting_level)
    if test_type == "T1":
        # Testing if the function doesn't raise any exception
        seg_eda_obj.numeric_summary(quick=quick_state)
    elif test_type == "T2":
        # Testing if the output dict has all the relevant segments
        calculated_dict = seg_eda_obj.numeric_summary(quick=quick_state)
        calculated_dict_keys = [list(calculated_dict.keys())]
        if type(calculated_dict["Complete Data"]) != str and not quick_state:
            calculated_dict_keys += [
                list(calculated_dict[i + "_wise"].keys())
                for i in select_segmenting_level
            ]
        calculated_dict_keys_flat = [
            item for sublist in calculated_dict_keys for item in sublist
        ]
        calculated_dict_keys_flat.sort()
        if quick_state:
            expected_dict_keys = ["Complete Data"]
        else:
            expected_dict_keys = ["Complete Data"]
            if type(calculated_dict["Complete Data"]) != str:
                expected_dict_keys = expected_dict_keys + [
                    i + "_wise" for i in select_segmenting_level
                ]
                if (
                    request.node.name.split("-")[2] == "Mixed_df"
                    or request.node.name.split("-")[2] == "Only_nums_df"
                ):
                    expected_dict_keys = expected_dict_keys + (
                        ["Num_col"] * len(select_segmenting_level)
                    )
        expected_dict_keys.sort()
        assert expected_dict_keys == calculated_dict_keys_flat


# --------------------------------------------------------------------------------------
# ---------------------- non_numeric_summary testing ---------------------------------
@pytest.mark.parametrize("quick_state", ["Q_F", "Q_T"], ids=id_func, indirect=True)
@pytest.mark.parametrize("shuffle_seg", ["no", "yes"], ids=id_func)
@pytest.mark.parametrize("have_non_empty_row", ["have"], ids=id_func)
@pytest.mark.parametrize("segment_property", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "df_type",
    ["Mixed_df", "Only_nums_df", "Only_cat_df", "Only_dt_df", "Only_bool_df"],
    ids=id_func,
    indirect=True,
)
@pytest.mark.parametrize(
    "select_segmenting_level", [1, 2, 3], ids=id_func, indirect=True
)
@pytest.mark.parametrize("test_type", ["T1", "T2"], ids=id_func)
@settings(max_examples=1, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=data()
)  # For testing in CI max_examples is set to 1, Can be increased upto 50 for developers
def test_non_numeric_summary(
    test_df,
    select_segmenting_level,
    df_type,
    segment_property,
    test_type,
    shuffle_seg,
    have_non_empty_row,
    quick_state,
    request,
):
    test_df_ = td.DataFrame(
        test_df.draw(
            df_with_select_col_and_segments(
                have_segments=len(select_segmenting_level),
                non_seg_cols=df_type,
                seg_has_na=segment_property,
                shuffle_seg=shuffle_seg,
                non_empty_row=have_non_empty_row,
            )
        )
    )
    seg_eda_obj = SegmentedEDAReport(test_df_, segment_by=select_segmenting_level)
    if test_type == "T1":
        # Testing if the function doesn't raise any exception
        seg_eda_obj.non_numeric_summary(quick=quick_state)
    elif test_type == "T2":
        # Testing if the output dict has all the relevant segments
        calculated_dict = seg_eda_obj.non_numeric_summary(quick=quick_state)
        calculated_dict_keys = [list(calculated_dict.keys())]
        if request.node.name.split("-")[2] != "Only_nums_df":
            if request.node.name.split("-")[2] != "Only_bool_df" and not quick_state:
                if type(calculated_dict["Complete Data"]) != str:
                    calculated_dict_keys += [
                        list(calculated_dict[i + "_wise"].keys())
                        for i in select_segmenting_level
                    ]
        calculated_dict_keys_flat = [
            item for sublist in calculated_dict_keys for item in sublist
        ]
        calculated_dict_keys_flat.sort()
        if quick_state:
            expected_dict_keys = ["Complete Data"]
        else:
            expected_dict_keys = ["Complete Data"]
            if type(calculated_dict["Complete Data"]) != str:
                level_wise_list = [i + "_wise" for i in select_segmenting_level]
                if request.node.name.split("-")[2] == "Mixed_df":
                    expected_dict_keys = (
                        expected_dict_keys
                        + level_wise_list
                        + (
                            ["Cat_col", "Dt_col", "Bool_col"]
                            * len(select_segmenting_level)
                        )
                    )
                elif request.node.name.split("-")[2] == "Only_cat_df":
                    expected_dict_keys = (
                        expected_dict_keys
                        + level_wise_list
                        + (["Cat_col"] * len(select_segmenting_level))
                    )
                elif request.node.name.split("-")[2] == "Only_dt_df":
                    expected_dict_keys = (
                        expected_dict_keys
                        + level_wise_list
                        + (["Dt_col"] * len(select_segmenting_level))
                    )
        expected_dict_keys.sort()
        assert expected_dict_keys == calculated_dict_keys_flat


# --------------------------------------------------------------------------------------
# ---------------------- correlation_table testing ---------------------------------
@pytest.mark.parametrize("shuffle_seg", ["no", "yes"], ids=id_func)
@pytest.mark.parametrize("have_non_empty_row", ["have"], ids=id_func)
@pytest.mark.parametrize("segment_property", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "df_type",
    [
        ["Num_col"],
        ["Num_col", "Num_col1"],
        ["Num_col", "Num_col2", "Num_col3"],
        ["Num_col", "Cat_col", "Dt_col", "Bool_col"],
    ],
    ids=id_func,
)
@pytest.mark.parametrize(
    "select_segmenting_level", [1, 2, 3], ids=id_func, indirect=True
)
@pytest.mark.parametrize("test_type", ["T1"], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_correlation_table(
    test_df,
    select_segmenting_level,
    df_type,
    segment_property,
    test_type,
    shuffle_seg,
    have_non_empty_row,
):
    test_df_ = td.DataFrame(
        test_df.draw(
            df_with_select_col_and_segments(
                have_segments=len(select_segmenting_level),
                non_seg_cols=df_type,
                seg_has_na=segment_property,
                shuffle_seg=shuffle_seg,
                non_empty_row=have_non_empty_row,
            )
        )
    )
    seg_eda_obj = SegmentedEDAReport(test_df_, segment_by=select_segmenting_level)
    if test_type == "T1":
        # Testing if the function doesn't raise any exception
        seg_eda_obj.correlation_table()


# --------------------------------------------------------------------------------------
# ---------------------- correlation_heatmap testing ---------------------------------
@pytest.mark.parametrize("shuffle_seg", ["no", "yes"], ids=id_func)
@pytest.mark.parametrize("have_non_empty_row", ["have"], ids=id_func)
@pytest.mark.parametrize("segment_property", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "df_type",
    [
        ["Num_col"],
        ["Num_col", "Num_col1"],
        ["Num_col", "Num_col2", "Num_col3"],
        ["Num_col", "Cat_col", "Dt_col", "Bool_col"],
    ],
    ids=id_func,
)
@pytest.mark.parametrize(
    "select_segmenting_level", [1, 2, 3], ids=id_func, indirect=True
)
@pytest.mark.parametrize("test_type", ["T1"], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_correlation_heatmap(
    test_df,
    select_segmenting_level,
    df_type,
    segment_property,
    test_type,
    shuffle_seg,
    have_non_empty_row,
):
    test_df_ = td.DataFrame(
        test_df.draw(
            df_with_select_col_and_segments(
                have_segments=len(select_segmenting_level),
                non_seg_cols=df_type,
                seg_has_na=segment_property,
                shuffle_seg=shuffle_seg,
                non_empty_row=have_non_empty_row,
            )
        )
    )
    seg_eda_obj = SegmentedEDAReport(test_df_, segment_by=select_segmenting_level)
    if test_type == "T1":
        # Testing if the function doesn't raise any exception
        seg_eda_obj.correlation_heatmap()


# --------------------------------------------------------------------------------------
# ---------------------- correlation_with_y testing ---------------------------------
@pytest.mark.parametrize("shuffle_seg", ["no", "yes"], ids=id_func)
@pytest.mark.parametrize("have_non_empty_row", ["have"], ids=id_func)
@pytest.mark.parametrize("segment_property", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "df_type",
    [
        ["Num_col"],
        ["Num_col", "Num_col1"],
        ["Num_col", "Num_col2", "Num_col3"],
        ["Num_col", "Cat_col", "Dt_col", "Bool_col"],
    ],
    ids=id_func,
)
@pytest.mark.parametrize(
    "select_segmenting_level", [1, 2, 3], ids=id_func, indirect=True
)
@pytest.mark.parametrize("test_type", ["T1"], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_correlation_with_y(
    test_df,
    select_segmenting_level,
    df_type,
    segment_property,
    test_type,
    shuffle_seg,
    have_non_empty_row,
):
    test_df_ = td.DataFrame(
        test_df.draw(
            df_with_select_col_and_segments(
                have_segments=len(select_segmenting_level),
                non_seg_cols=df_type,
                seg_has_na=segment_property,
                shuffle_seg=shuffle_seg,
                non_empty_row=have_non_empty_row,
            )
        )
    )
    test_df_ = test_df_.replace([np.inf, -np.inf], np.nan).dropna()

    seg_eda_obj = SegmentedEDAReport(
        test_df_, segment_by=select_segmenting_level, y="Num_col"
    )
    if test_type == "T1":
        # Testing if the function doesn't raise any exception
        seg_eda_obj.correlation_with_y()


# --------------------------------------------------------------------------------------
# ---------------------- _get_segment_contribution testing ---------------------------------
@pytest.mark.parametrize("shuffle_seg", ["no", "yes"], ids=id_func)
@pytest.mark.parametrize("have_non_empty_row", ["have", "not_have"], ids=id_func)
@pytest.mark.parametrize("segment_property", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "df_type", ["Mixed_df", "Only_nums_df"], ids=id_func, indirect=True
)
@pytest.mark.parametrize(
    "select_segmenting_level", [1, 2, 3], ids=id_func, indirect=True
)
@pytest.mark.parametrize("test_type", ["T1", "T4"], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 100 for developers
def test_get_segment_contribution(
    test_df,
    select_segmenting_level,
    df_type,
    segment_property,
    test_type,
    shuffle_seg,
    have_non_empty_row,
):
    test_df_ = td.DataFrame(
        test_df.draw(
            df_with_select_col_and_segments(
                have_segments=len(select_segmenting_level),
                non_seg_cols=df_type,
                seg_has_na=segment_property,
                shuffle_seg=shuffle_seg,
                non_empty_row=have_non_empty_row,
            )
        )
    )
    seg_eda_obj = SegmentedEDAReport(test_df_, segment_by=select_segmenting_level)
    if test_type == "T1":
        # Testing if the function doesn't raise any exception
        seg_eda_obj._get_segment_contribution(
            metric_column="Num_col",
            metric_aggregation=sum,
            group_segments_by=select_segmenting_level,
            bin_limits=None,
        )
    elif test_type == "T4":
        # Testing if the plot type is correct
        computed_plot = seg_eda_obj._get_segment_contribution(
            metric_column="Num_col",
            metric_aggregation=sum,
            group_segments_by=select_segmenting_level,
            bin_limits=None,
        )
        assert type(computed_plot) == holoviews.core.overlay.Overlay
