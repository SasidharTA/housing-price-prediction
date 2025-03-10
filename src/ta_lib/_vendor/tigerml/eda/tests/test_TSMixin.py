import os
from datetime import datetime, timedelta

import holoviews
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.pandas import columns, data_frames, range_indexes
from hypothesis.strategies import (
    composite,
    data,
    floats,
    integers,
    lists,
    sampled_from,
    text,
    tuples,
)

from tigerml.eda.Analyser import TSAnalyser
from tigerml.eda.time_series import SegmentedTSReport, TSReport

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------
# ----------------- creating composite strategies for testing -------------------------
# a composite hypotesis.strategy that generates a tuple with float, string and mixed values
@composite
def mixed_tuple(draw):
    fl = draw(floats(allow_nan=True, allow_infinity=True,))
    tx = draw(sampled_from(["Red", "Blue", "Green", None]))
    s_f = draw(sampled_from(["text", 0.527, 50000, np.nan, None]))
    return tuple([fl, tx, s_f])


# a composite hypotesis.strategy that generates a single column time_series dateframe
@composite
def make_ts(draw, start_date=None, have_miss_dups=False, shuffle_ts="sorted_ts"):
    del_row1 = 1
    period = draw(
        integers(min_value=7, max_value=10)
    )  # Earlier min_value=3 (i.e. without have_miss_dups)
    if not start_date:
        start_date = pd.Timestamp("10/1/1995")  # pd.Timestamp.now()  # datetime.now()
        period = draw(
            integers(min_value=6, max_value=13)
        )  # Earlier min_value=2 (i.e. without have_miss_dups)
        # if min_value=5 then compute_periodicity(),
        # no_of_periods() and missing_periods() fails
        del_row1 = 0
    freq_level = draw(
        sampled_from(["weeks", "days", "hours", "minutes", "seconds", "microseconds"])
    )
    freq_value = draw(
        floats(min_value=1, max_value=10)
    )  # draw(sampled_from([2, 2.5])) --> to avoid hypothesis crash
    exp = (
        "pd.date_range(start='{}', periods={}, freq=pd.DateOffset({}={}))"
        ".to_frame(name='ts_col').reset_index(drop=True)".format(
            start_date.strftime("%m/%d/%Y, %H:%M:%S.%f"), period, freq_level, freq_value
        )
    )
    ts_df = eval(exp)
    ts_df = ts_df[del_row1:]
    ts_df = ts_df.reset_index(drop=True)
    act_ts_df = ts_df.copy()
    act_ts_df.columns = ["old_ts"]
    no_of_periods = len(ts_df)
    act_first_period = ts_df["ts_col"][0]
    act_last_period = ts_df["ts_col"][len(ts_df) - 1]

    if have_miss_dups:
        drop_index = draw(integers(min_value=1, max_value=len(ts_df) - 2))
        ts_df = ts_df.drop(ts_df.index[[drop_index]]).reset_index(drop=True)
        nat_index = draw(integers(min_value=1, max_value=len(ts_df) - 2))
        ts_df["ts_col"][nat_index] = pd.NaT
        valid_index = list(range(len(ts_df)))
        rep_index = draw(
            sampled_from([index for index in valid_index if index not in [nat_index]])
        )
        ts_df = (
            ts_df[: rep_index + 1]
            .append(
                pd.DataFrame({"ts_col": [ts_df["ts_col"][rep_index]]}),
                ignore_index=True,
            )
            .append(ts_df[rep_index + 1 :], ignore_index=True)
        )

    if shuffle_ts == "shuffled_ts":
        ts_df = ts_df.sample(frac=1).reset_index(drop=True)

    ts_df = ts_df.join(act_ts_df)
    ts_df["act_periodicity"] = eval("timedelta({}={})".format(freq_level, freq_value))
    ts_df["act_no_of_periods"] = no_of_periods
    ts_df["act_first_period"] = act_first_period
    ts_df["act_last_period"] = act_last_period
    return ts_df


# a composite hypotesis.strategy that generates a dateframe with a time_series, float, string and mixed column
@composite
def df_with_ts_col(
    draw,
    multi_period=False,
    return_cols=None,
    have_miss_dups=False,
    have_segments=0,
    shuffle_ts="sorted_ts",
):
    if not multi_period:
        ts_df = draw(make_ts(have_miss_dups=have_miss_dups, shuffle_ts=shuffle_ts))
        data_df = draw(
            data_frames(
                index=range_indexes(min_size=len(ts_df), max_size=len(ts_df)),
                columns=columns(["float_col", "string_col", "mixed_col"], dtype=float),
                rows=mixed_tuple(),
            )
        )
        data_df_all = ts_df.join(data_df)
    else:
        nos = 2  # draw(integers(min_value=3, max_value=10))
        data_df_all = pd.DataFrame()
        for i in range(nos):
            if i == 0:
                ts_df = draw(make_ts(have_miss_dups=have_miss_dups))
            else:
                ts_df = draw(
                    make_ts(
                        start_date=data_df["act_last_period"].max(),
                        have_miss_dups=have_miss_dups,
                        shuffle_ts=shuffle_ts,
                    )
                )
            data_df = draw(
                data_frames(
                    index=range_indexes(min_size=len(ts_df), max_size=len(ts_df)),
                    columns=columns(
                        ["float_col", "string_col", "mixed_col"], dtype=float
                    ),
                    rows=mixed_tuple(),
                )
            )
            data_df = ts_df.join(data_df)
            data_df_all = data_df_all.append(data_df, ignore_index=True)

    if return_cols:
        if return_cols == "no":
            cols = ["ts_col", "float_col", "string_col", "mixed_col"]
        else:
            cols = ["ts_col"] + return_cols + ["float_col", "string_col", "mixed_col"]
        data_df_all = data_df_all[cols]

    if have_segments == 1:
        segments = [["L1_S1"], ["L1_S2"]]
        frames = []
        for segment in segments:
            frames += [
                data_df_all.join(
                    pd.DataFrame([segment] * len(data_df_all), columns=["Lev_1"])
                )
            ]
        data_df_all = pd.concat(frames, ignore_index=True)

    elif have_segments == 2:
        segments = [
            ["L1_S1", "L2_S1"],
            ["L1_S1", "L2_S2"],
            ["L1_S2", "L2_S1"],
            ["L1_S3", "L2_S1"],
            ["L1_S3", "L2_S2"],
            ["L1_S3", "L2_S3"],
        ]
        frames = []
        for segment in segments:
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
            frames += [
                data_df_all.join(
                    pd.DataFrame(
                        [segment] * len(data_df_all),
                        columns=["Lev_1", "Lev_2", "Lev_3"],
                    )
                )
            ]
        data_df_all = pd.concat(frames, ignore_index=True)

    return data_df_all


@composite
def df_with_ts_col_variable_segments(
    draw,
    multi_period=False,
    return_cols=None,
    have_miss_dups=False,
    have_segments=0,
    shuffle_ts="sorted_ts",
):

    if have_segments == 0:
        data_df_all = draw(
            df_with_ts_col(
                multi_period=multi_period,
                return_cols=return_cols,
                have_miss_dups=have_miss_dups,
                have_segments=0,
                shuffle_ts=shuffle_ts,
            )
        )

    elif have_segments == 1:
        segments = [["L1_S1"], ["L1_S2"]]
        frames = []
        for segment in segments:
            data_df_all = draw(
                df_with_ts_col(
                    multi_period=multi_period,
                    return_cols=return_cols,
                    have_miss_dups=have_miss_dups,
                    have_segments=0,
                    shuffle_ts=shuffle_ts,
                )
            )
            frames += [
                data_df_all.join(
                    pd.DataFrame([segment] * len(data_df_all), columns=["Lev_1"])
                )
            ]
        data_df_all = pd.concat(frames, ignore_index=True)

    elif have_segments == 2:
        segments = [
            ["L1_S1", "L2_S1"],
            ["L1_S1", "L2_S2"],
            ["L1_S2", "L2_S1"],
            ["L1_S3", "L2_S1"],
            ["L1_S3", "L2_S2"],
            ["L1_S3", "L2_S3"],
        ]
        frames = []
        for segment in segments:
            data_df_all = draw(
                df_with_ts_col(
                    multi_period=multi_period,
                    return_cols=return_cols,
                    have_miss_dups=have_miss_dups,
                    have_segments=0,
                    shuffle_ts=shuffle_ts,
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
                df_with_ts_col(
                    multi_period=multi_period,
                    return_cols=return_cols,
                    have_miss_dups=have_miss_dups,
                    have_segments=0,
                    shuffle_ts=shuffle_ts,
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

    return data_df_all


# Function to give meaningful test_ids for all the test combinations
def id_func(param):
    id_dict = {
        "0": "TSAnalyser",
        "1": "SegmentedTSAnalyser(1 level)",
        "2": "SegmentedTSAnalyser(2 levels)",
        "3": "SegmentedTSAnalyser(3 levels)",
        "True": "miss_dup_ts",
        "False": "clean_ts",
    }
    if type(param) == list:
        if len(param) > 1:
            return "[" + ",".join(param) + "]"
        else:
            return param[0]
    return id_dict[str(param)]


# --------------------------------------------------------------------------------------
# ---------------------- _compute_periodicity testing ---------------------------------
@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_compute_periodicity(test_df_, have_segments, have_miss_dups, shuffle_ts):
    test_df = test_df_.draw(
        df_with_ts_col(
            multi_period=True,
            have_miss_dups=have_miss_dups,
            return_cols=["old_ts", "act_periodicity"],
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    expected_periodicity = test_df["act_periodicity"].min()
    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        computed_periodicity = tsa_obj._compute_periodicity()
        assert computed_periodicity == expected_periodicity
    elif have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        computed_periodicity = tsa_obj._compute_periodicity()
        assert (computed_periodicity == expected_periodicity).sum() == 6
    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        computed_periodicity = tsa_obj._compute_periodicity()
        assert (computed_periodicity == expected_periodicity).sum() == 3


# --------------------------------------------------------------------------------------
# ---------------------- first_period_in_series testing ---------------------------------
@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_first_period_in_series(test_df_, have_segments, have_miss_dups, shuffle_ts):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols=["act_first_period"],
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    expected_first_period = test_df["act_first_period"].min()
    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        computed_first_period = tsa_obj.first_period_in_series()
        assert computed_first_period == expected_first_period
    elif have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        computed_first_period = tsa_obj.first_period_in_series()
        assert (computed_first_period == expected_first_period).sum() == 6
    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        computed_first_period = tsa_obj.first_period_in_series()
        assert (computed_first_period == expected_first_period).sum() == 3


# --------------------------------------------------------------------------------------
# ---------------------- last_period_in_series testing ---------------------------------
@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_last_period_in_series(test_df_, have_segments, have_miss_dups, shuffle_ts):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols=["act_last_period"],
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    expected_last_period = test_df["act_last_period"].max()
    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        computed_last_period = tsa_obj.last_period_in_series()
        assert computed_last_period == expected_last_period
    elif have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        computed_last_period = tsa_obj.last_period_in_series()
        assert (computed_last_period == expected_last_period).sum() == 6
    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        computed_last_period = tsa_obj.last_period_in_series()
        assert (computed_last_period == expected_last_period).sum() == 3


# --------------------------------------------------------------------------------------
# ---------------------------- no_of_periods testing ---------------------------------
@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_no_of_periods(test_df_, have_segments, have_miss_dups, shuffle_ts):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols=["old_ts", "act_no_of_periods"],
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    expected_no_of_periods = test_df["act_no_of_periods"][0]
    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        computed_no_of_periods = tsa_obj.no_of_periods()
        assert computed_no_of_periods == expected_no_of_periods
    elif have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        computed_no_of_periods = tsa_obj.no_of_periods()
        assert (computed_no_of_periods == expected_no_of_periods).sum() == 6
    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        computed_no_of_periods = tsa_obj.no_of_periods()
        assert (computed_no_of_periods == expected_no_of_periods).sum() == 3


# --------------------------------------------------------------------------------------
# ---------------------------- missing_periods testing ---------------------------------
@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_missing_periods(test_df_, have_segments, have_miss_dups, shuffle_ts):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols=["old_ts"],
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    if have_miss_dups:
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            computed_missing_periods = tsa_obj.missing_periods(return_values=False)[
                "no_of_missing_periods"
            ]
            assert computed_missing_periods == 2
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            computed_missing_periods = tsa_obj.missing_periods(return_values=False)[
                "no_of_missing_periods"
            ]
            assert (computed_missing_periods == 2).sum() == 6
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            computed_missing_periods = tsa_obj.missing_periods(return_values=False)[
                "no_of_missing_periods"
            ]
            assert (computed_missing_periods == 2).sum() == 3
    else:
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            computed_missing_periods = tsa_obj.missing_periods(return_values=False)
            assert computed_missing_periods == "No missing periods in series"
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            computed_missing_periods = tsa_obj.missing_periods(return_values=False)
            assert computed_missing_periods == "No missing periods in series"
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            computed_missing_periods = tsa_obj.missing_periods(return_values=False)
            assert computed_missing_periods == "No missing periods in series"


# --------------------------------------------------------------------------------------
# ---------------------------- get_time_repetitions testing ---------------------------------
@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_get_time_repetitions(test_df_, have_segments, have_miss_dups, shuffle_ts):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols=["old_ts"],
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    if have_miss_dups:
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            computed_time_repetitions = tsa_obj.get_time_repetitions(
                return_values=False
            )
            if isinstance(computed_time_repetitions, dict):
                computed_time_repetitions = computed_time_repetitions["no_of_repetitions"]
            assert computed_time_repetitions == 1
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            computed_time_repetitions = tsa_obj.get_time_repetitions(
                return_values=False
            )
            if isinstance(computed_time_repetitions, dict):
                computed_time_repetitions = computed_time_repetitions["no_of_repetitions"]            
            assert (computed_time_repetitions == 1).sum() == 6
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            computed_time_repetitions = tsa_obj.get_time_repetitions(
                return_values=False
            )
            if isinstance(computed_time_repetitions, dict):
                computed_time_repetitions = computed_time_repetitions["no_of_repetitions"]
            assert (computed_time_repetitions == 1).sum() == 3
    else:
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            computed_time_repetitions = tsa_obj.get_time_repetitions(
                return_values=False
            )
            if isinstance(computed_time_repetitions, dict):
                computed_time_repetitions = computed_time_repetitions["no_of_repetitions"]
            assert computed_time_repetitions == "No repetitions in series"
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            computed_time_repetitions = tsa_obj.get_time_repetitions(
                return_values=False
            )
            if isinstance(computed_time_repetitions, dict):
                computed_time_repetitions = computed_time_repetitions["no_of_repetitions"]
            assert computed_time_repetitions == "No repetitions in series"
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            computed_time_repetitions = tsa_obj.get_time_repetitions(
                return_values=False
            )
            if isinstance(computed_time_repetitions, dict):
                computed_time_repetitions = computed_time_repetitions["no_of_repetitions"]
            assert computed_time_repetitions == "No repetitions in series"


# --------------------------------------------------------------------------------------
# ---------------------------- get_outliers testing ---------------------------------
@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "cols",
    [
        ["float_col"],
        ["string_col"],
        ["mixed_col"],
        ["float_col", "string_col", "mixed_col"],
    ],
    ids=id_func,
)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_get_outliers(test_df_, have_segments, have_miss_dups, shuffle_ts, cols):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols="no",
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        tsa_obj.get_outliers(cols=cols)
    elif have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        tsa_obj.get_outliers(cols=cols)
    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        tsa_obj.get_outliers(cols=cols)


# --------------------------------------------------------------------------------------
# ---------------------------- get_change_points testing ---------------------------------
@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "cols",
    [
        ["float_col"],
        ["string_col"],
        ["mixed_col"],
        ["float_col", "string_col", "mixed_col"],
    ],
    ids=id_func,
)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_get_change_points(test_df_, have_segments, have_miss_dups, shuffle_ts, cols):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols="no",
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        tsa_obj.get_change_points(cols=cols)
    elif have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        tsa_obj.get_change_points(cols=cols)
    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        tsa_obj.get_change_points(cols=cols)


# --------------------------------------------------------------------------------------
# ---------------------Test plot_last_period_occurrence --------------------------------


@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_plot_last_period_occurence(
    test_df_, have_segments, have_miss_dups, shuffle_ts
):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols=["act_last_period"],
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )

    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        computed_last_period = tsa_obj.last_period_in_series()
        plot = tsa_obj.plot_last_period_occurrence()
        assert isinstance(computed_last_period, datetime) and plot is None

    if have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        plot = tsa_obj.plot_last_period_occurrence()
        assert type(plot) == holoviews.element.chart.Bars

    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        plot = tsa_obj.plot_last_period_occurrence()
        assert type(plot) == holoviews.element.chart.Bars


# ---------------------------------------------------------------------------------------------------------
# ---------------------------Plot occurence of first period -----------------------------------------------


@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 50 for developers
def test_plot_first_period_occurence(
    test_df_, have_segments, have_miss_dups, shuffle_ts
):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols=["act_first_period"],
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )

    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        computed_first_period = tsa_obj.first_period_in_series()
        plot = tsa_obj.plot_first_period_occurrence()
        assert isinstance(computed_first_period, datetime) and plot is None
    if have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        plot = tsa_obj.plot_first_period_occurrence()
        assert type(plot) == holoviews.element.chart.Bars
    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        plot = tsa_obj.plot_first_period_occurrence()
        assert type(plot) == holoviews.element.chart.Bars


# -----------------------------------------------------------------------------------------------------
# ---------------------- Test Outliers Plot -----------------------------------------------------------


@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "cols",
    [
        ["float_col"],
        ["string_col"],
        ["mixed_col"],
        ["float_col", "string_col", "mixed_col"],
    ],
    ids=id_func,
)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_outliers_plot(test_df_, have_segments, have_miss_dups, shuffle_ts, cols):
    test_df = test_df_.draw(
        df_with_ts_col_variable_segments(
            have_miss_dups=have_miss_dups,
            return_cols="no",
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )

    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        if isinstance(tsa_obj.get_outliers(), (str, dict)):
            assert tsa_obj.outliers_plot() is None
    elif have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        if isinstance(tsa_obj.get_outliers(), (str)):
            assert type(tsa_obj.outliers_plot()) is str
        elif isinstance(tsa_obj.get_outliers(), (dict)):
            assert tsa_obj.outliers_plot() is None
        else:
            plot = tsa_obj.outliers_plot()
            assert type(plot) == holoviews.element.chart.Bars
    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        if isinstance(tsa_obj.get_outliers(), (str)):
            assert type(tsa_obj.outliers_plot()) is str
        elif isinstance(tsa_obj.get_outliers(), (dict)):
            assert tsa_obj.outliers_plot() is None
        else:
            plot = tsa_obj.outliers_plot()
            assert type(plot) == holoviews.element.chart.Bars


# ---------------------------- show_change_points testing ---------------------------------


@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "cols",
    [
        ["float_col"],
        ["string_col"],
        ["mixed_col"],
        ["float_col", "string_col", "mixed_col"],
    ],
    ids=id_func,
)
@pytest.mark.parametrize("have_segments", [0, 1, 2, 3], ids=id_func)
@pytest.mark.parametrize("test_type", ["output_generation_test", "output_type_test"])
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_show_change_points(
    test_df_, have_segments, have_miss_dups, shuffle_ts, cols, test_type
):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols="no",
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    if test_type == "output_generation_test":
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            tsa_obj.show_change_points(cols=cols)
        elif have_segments == 1:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1"]
            )
            tsa_obj.show_change_points(cols=cols)
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            tsa_obj.show_change_points(cols=cols)
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            tsa_obj.show_change_points(cols=cols)
    else:
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            computed_output = tsa_obj.show_change_points(cols=cols)
            if isinstance(computed_output, dict):
                type_list = [
                    type(computed_output[i]) == holoviews.core.overlay.Overlay
                    for i in computed_output.keys()
                ]
                assert len(computed_output) == np.array(type_list).sum()
            else:
                str_list = [
                    "Change point detection not",
                    "No change points in the data",
                    "No numeric column",
                ]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
        elif have_segments == 1:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1"]
            )
            filtered_data = tsa_obj.data[(tsa_obj.data["Lev_1"] == "L1_S1")]
            computed_output = tsa_obj.show_change_points(data=filtered_data, cols=cols)
            if isinstance(computed_output, dict):
                type_list = [
                    type(computed_output[i]) == holoviews.core.overlay.Overlay
                    for i in computed_output.keys()
                ]
                assert len(computed_output) == np.array(type_list).sum()
            else:
                str_list = [
                    "Change point detection not",
                    "No change points in the data",
                    "No numeric column",
                    "Segment data",
                ]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            filtered_data = tsa_obj.data[
                (tsa_obj.data["Lev_1"] == "L1_S1") & (tsa_obj.data["Lev_2"] == "L2_S1")
            ]
            computed_output = tsa_obj.show_change_points(data=filtered_data, cols=cols)
            if isinstance(computed_output, dict):
                type_list = [
                    type(computed_output[i]) == holoviews.core.overlay.Overlay
                    for i in computed_output.keys()
                ]
                assert len(computed_output) == np.array(type_list).sum()
            else:
                str_list = [
                    "Change point detection not",
                    "No change points in the data",
                    "No numeric column",
                    "Segment data",
                ]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            filtered_data = tsa_obj.data[
                (tsa_obj.data["Lev_1"] == "L1_S1")
                & (tsa_obj.data["Lev_2"] == "L2_S1")
                & (tsa_obj.data["Lev_3"] == "L3_S1")
            ]
            computed_output = tsa_obj.show_change_points(data=filtered_data, cols=cols)
            if isinstance(computed_output, dict):
                type_list = [
                    type(computed_output[i]) == holoviews.core.overlay.Overlay
                    for i in computed_output.keys()
                ]
                assert len(computed_output) == np.array(type_list).sum()
            else:
                str_list = [
                    "Change point detection not",
                    "No change points in the data",
                    "No numeric column",
                    "Segment data",
                ]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"


# ----------------------------------------------------------------------------------------------------
# -----------------------------test plot acf ---------------------------------------------------------

# @pytest.mark.parametrize("shuffle_ts",
#                          ['sorted_ts', 'shuffled_ts'])
# @pytest.mark.parametrize("have_miss_dups",
#                          [False, True],
#                          ids=id_func)
# @pytest.mark.parametrize("cols",
#                          [["float_col"], ["string_col"], ["mixed_col"], ["float_col", "string_col", "mixed_col"]],
#                          ids=id_func)
# @pytest.mark.parametrize("have_segments",
#                          [0, 2, 3],
#                          ids=id_func)
# @settings(max_examples=2, deadline=None,suppress_health_check=HealthCheck.all())
# @given(test_df_=data()) # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
# def test_acf_plot(test_df_, have_segments, have_miss_dups, shuffle_ts, cols):
#     test_df = test_df_.draw(df_with_ts_col_variable_segments(have_miss_dups=have_miss_dups, return_cols='no',
#                                                              shuffle_ts=shuffle_ts, have_segments=have_segments))
#     test_df=test_df.replace((np.inf,-np.inf),np.nan).dropna(subset=['float_col'])
#     if test_df.size == 0:
#         pass
#     else :
#         if have_segments == 0:
#             tsa_obj = TSReport(test_df, 'ts_col',y='float_col')
#             plot = tsa_obj.get_acf_plot()
#             assert (type(plot) == holoviews.core.overlay.Overlay)
#         elif have_segments == 2:
#             tsa_obj = SegmentedTSReport(test_df, 'ts_col', ts_identifiers=["Lev_1", "Lev_2"],y='float_col')
#             plot = tsa_obj.get_acf_plot()
#             assert (type(plot) == holoviews.core.overlay.Overlay)
#         elif have_segments == 3:
#             tsa_obj = SegmentedTSReport(test_df, 'ts_col', ts_identifiers=["Lev_1", "Lev_2", "Lev_3"],y='float_col')
#             plot = tsa_obj.get_acf_plot()
#             assert (type(plot) == holoviews.core.overlay.Overlay)


# --------------------------------------------------------------------------------------
# ---------------------------- show_outliers testing ---------------------------------


@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "cols",
    [
        ["float_col"],
        ["string_col"],
        ["mixed_col"],
        ["float_col", "string_col", "mixed_col"],
    ],
    ids=id_func,
)
@pytest.mark.parametrize("have_segments", [0, 1, 2, 3], ids=id_func)
@pytest.mark.parametrize("test_type", ["output_generation_test", "output_type_test"])
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_show_outliers(
    test_df_, have_segments, have_miss_dups, shuffle_ts, cols, test_type
):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols="no",
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    if test_type == "output_generation_test":
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            tsa_obj.show_outliers(cols=cols)
        elif have_segments == 1:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1"]
            )
            tsa_obj.show_outliers(cols=cols)
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            tsa_obj.show_outliers(cols=cols)
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            tsa_obj.show_outliers(cols=cols)
    else:
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            computed_output = tsa_obj.show_outliers(cols=cols)
            if isinstance(computed_output, dict):
                type_list = [
                    type(computed_output[i]) == holoviews.core.overlay.Overlay
                    for i in computed_output.keys()
                ]
                assert len(computed_output) == np.array(type_list).sum()
            else:
                str_list = [
                    "Outlier detection not",
                    "No outliers in the data",
                    "No numeric column",
                ]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
        elif have_segments == 1:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1"]
            )
            filtered_data = tsa_obj.data[(tsa_obj.data["Lev_1"] == "L1_S1")]
            computed_output = tsa_obj.show_outliers(data=filtered_data, cols=cols)
            if isinstance(computed_output, dict):
                type_list = [
                    type(computed_output[i]) == holoviews.core.overlay.Overlay
                    for i in computed_output.keys()
                ]
                assert len(computed_output) == np.array(type_list).sum()
            else:
                str_list = [
                    "Outlier detection not",
                    "No outliers in the data",
                    "No numeric column",
                    "Segment data",
                ]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            filtered_data = tsa_obj.data[
                (tsa_obj.data["Lev_1"] == "L1_S1") & (tsa_obj.data["Lev_2"] == "L2_S1")
            ]
            computed_output = tsa_obj.show_outliers(data=filtered_data, cols=cols)
            if isinstance(computed_output, dict):
                type_list = [
                    type(computed_output[i]) == holoviews.core.overlay.Overlay
                    for i in computed_output.keys()
                ]
                assert len(computed_output) == np.array(type_list).sum()
            else:
                str_list = [
                    "Outlier detection not",
                    "No outliers in the data",
                    "No numeric column",
                    "Segment data",
                ]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            filtered_data = tsa_obj.data[
                (tsa_obj.data["Lev_1"] == "L1_S1")
                & (tsa_obj.data["Lev_2"] == "L2_S1")
                & (tsa_obj.data["Lev_3"] == "L3_S1")
            ]
            computed_output = tsa_obj.show_outliers(data=filtered_data, cols=cols)
            if isinstance(computed_output, dict):
                type_list = [
                    type(computed_output[i]) == holoviews.core.overlay.Overlay
                    for i in computed_output.keys()
                ]
                assert len(computed_output) == np.array(type_list).sum()
            else:
                str_list = [
                    "Outlier detection not",
                    "No outliers in the data",
                    "No numeric column",
                    "Segment data",
                ]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"


# ----------------------------------------------------------------------------------------------------
# -----------------------------test plot pacf ---------------------------------------------------------

# @pytest.mark.parametrize("shuffle_ts",
#                          ['sorted_ts', 'shuffled_ts'])
# @pytest.mark.parametrize("have_miss_dups",
#                          [False, True],
#                          ids=id_func)
# @pytest.mark.parametrize("cols",
#                          [["float_col"], ["string_col"], ["mixed_col"], ["float_col", "string_col", "mixed_col"]],
#                          ids=id_func)
# @pytest.mark.parametrize("have_segments",
#                          [0, 2, 3],
#                          ids=id_func)
# @settings(max_examples=2, deadline=None,suppress_health_check=HealthCheck.all())
# @given(test_df_=data()) # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
# def test_pacf_plot(test_df_, have_segments, have_miss_dups, shuffle_ts, cols):
#     test_df = test_df_.draw(df_with_ts_col_variable_segments(have_miss_dups=have_miss_dups, return_cols='no',
#                                                              shuffle_ts=shuffle_ts, have_segments=have_segments))
#     test_df=test_df.replace((np.inf,-np.inf),np.nan).dropna()
#     if test_df.size == 0:
#         pass
#     else :
#         if have_segments == 0:
#             tsa_obj = TSReport(test_df,'ts_col',y='float_col')
#             plot = tsa_obj.get_pacf_plot()
#             assert (type(plot) == holoviews.core.overlay.Overlay)
#         elif have_segments == 2:
#             tsa_obj = SegmentedTSReport(test_df, 'ts_col', ts_identifiers=["Lev_1", "Lev_2"],y='float_col')
#             plot = tsa_obj.get_pacf_plot()
#             assert (type(plot) == holoviews.core.overlay.Overlay)
#         elif have_segments == 3:
#             tsa_obj = SegmentedTSReport(test_df, 'ts_col', ts_identifiers=["Lev_1", "Lev_2", "Lev_3"],y='float_col')
#             plot = tsa_obj.get_pacf_plot()
#             assert (type(plot) == holoviews.core.overlay.Overlay)


# --------------------------------------------------------------------------------------
# ---------------------------- show_missing_periods testing ---------------------------------


@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 1, 2, 3], ids=id_func)
@pytest.mark.parametrize("test_type", ["output_generation_test", "output_type_test"])
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_show_missing_periods(
    test_df_, have_segments, have_miss_dups, shuffle_ts, test_type
):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols="no",
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    if test_type == "output_generation_test":
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            tsa_obj.show_missing_periods()
        elif have_segments == 1:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1"]
            )
            tsa_obj.show_missing_periods()
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            tsa_obj.show_missing_periods()
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            tsa_obj.show_missing_periods()
    else:
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            computed_output = tsa_obj.show_missing_periods()
            if isinstance(computed_output, str):
                str_list = ["No missing periods"]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
            else:
                assert type(computed_output) == holoviews.element.chart.Curve

        elif have_segments == 1:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1"]
            )
            filtered_data = tsa_obj.data[(tsa_obj.data["Lev_1"] == "L1_S1")]
            computed_output = tsa_obj.show_missing_periods(data=filtered_data)
            if isinstance(computed_output, str):
                str_list = ["No missing periods", "Segment data"]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
            else:
                assert type(computed_output) == holoviews.element.chart.Curve

        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            filtered_data = tsa_obj.data[
                (tsa_obj.data["Lev_1"] == "L1_S1") & (tsa_obj.data["Lev_2"] == "L2_S1")
            ]
            computed_output = tsa_obj.show_missing_periods(data=filtered_data)
            if isinstance(computed_output, str):
                str_list = ["No missing periods", "Segment data"]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
            else:
                assert type(computed_output) == holoviews.element.chart.Curve

        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            filtered_data = tsa_obj.data[
                (tsa_obj.data["Lev_1"] == "L1_S1")
                & (tsa_obj.data["Lev_2"] == "L2_S1")
                & (tsa_obj.data["Lev_3"] == "L3_S1")
            ]
            computed_output = tsa_obj.show_missing_periods(data=filtered_data)
            if isinstance(computed_output, str):
                str_list = ["No missing periods", "Segment data"]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
            else:
                assert type(computed_output) == holoviews.element.chart.Curve


# ----------------------------------------------------------------------------------------------------
# -----------------------------test plot get_conf_lines ---------------------------------------------------------


@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize(
    "cols",
    [
        ["float_col"],
        ["string_col"],
        ["mixed_col"],
        ["float_col", "string_col", "mixed_col"],
    ],
    ids=id_func,
)
@pytest.mark.parametrize("have_segments", [0, 2, 3], ids=id_func)
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_get_conf_lines_plot(test_df_, have_segments, have_miss_dups, shuffle_ts, cols):
    test_df = test_df_.draw(
        df_with_ts_col_variable_segments(
            have_miss_dups=have_miss_dups,
            return_cols="no",
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )

    if have_segments == 0:
        tsa_obj = TSReport(test_df, "ts_col")
        plot = tsa_obj.get_conf_lines()
        assert type(plot) == holoviews.core.overlay.Overlay
    elif have_segments == 2:
        tsa_obj = SegmentedTSReport(
            test_df, "ts_col", ts_identifiers=["Lev_1", "Lev_2"]
        )
        plot = tsa_obj.get_conf_lines()
        assert type(plot) == holoviews.core.overlay.Overlay
    elif have_segments == 3:
        tsa_obj = SegmentedTSReport(
            test_df, "ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
        )
        plot = tsa_obj.get_conf_lines()
        assert type(plot) == holoviews.core.overlay.Overlay


# --------------------------------------------------------------------------------------
# ---------------------------- show_time_repetitions testing ---------------------------------


@pytest.mark.parametrize("shuffle_ts", ["sorted_ts", "shuffled_ts"])
@pytest.mark.parametrize("have_miss_dups", [False, True], ids=id_func)
@pytest.mark.parametrize("have_segments", [0, 1, 2, 3], ids=id_func)
@pytest.mark.parametrize("test_type", ["output_generation_test", "output_type_test"])
@settings(max_examples=2, deadline=None, suppress_health_check=HealthCheck.all())
@given(
    test_df_=data()
)  # For testing in CI max_examples is set to 2, Can be increased upto 200 for developers
def test_show_time_repetitions(
    test_df_, have_segments, have_miss_dups, shuffle_ts, test_type
):
    test_df = test_df_.draw(
        df_with_ts_col(
            have_miss_dups=have_miss_dups,
            return_cols="no",
            shuffle_ts=shuffle_ts,
            have_segments=have_segments,
        )
    )
    if test_type == "output_generation_test":
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            tsa_obj.show_time_repetitions()
        elif have_segments == 1:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1"]
            )
            tsa_obj.show_time_repetitions()
        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            tsa_obj.show_time_repetitions()
        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            tsa_obj.show_time_repetitions()
    else:
        if have_segments == 0:
            tsa_obj = TSReport(test_df, "ts_col")
            computed_output = tsa_obj.show_time_repetitions()
            if isinstance(computed_output, str):
                str_list = ["No repetitions"]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
            else:
                assert type(computed_output) == holoviews.element.chart.Curve

        elif have_segments == 1:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1"]
            )
            filtered_data = tsa_obj.data[(tsa_obj.data["Lev_1"] == "L1_S1")]
            computed_output = tsa_obj.show_time_repetitions(data=filtered_data)
            if isinstance(computed_output, str):
                str_list = ["No repetitions", "Segment data"]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
            else:
                assert type(computed_output) == holoviews.element.chart.Curve

        elif have_segments == 2:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2"]
            )
            filtered_data = tsa_obj.data[
                (tsa_obj.data["Lev_1"] == "L1_S1") & (tsa_obj.data["Lev_2"] == "L2_S1")
            ]
            computed_output = tsa_obj.show_time_repetitions(data=filtered_data)
            if isinstance(computed_output, str):
                str_list = ["No repetitions", "Segment data"]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
            else:
                assert type(computed_output) == holoviews.element.chart.Curve

        elif have_segments == 3:
            tsa_obj = SegmentedTSReport(
                test_df, ts_column="ts_col", ts_identifiers=["Lev_1", "Lev_2", "Lev_3"]
            )
            filtered_data = tsa_obj.data[
                (tsa_obj.data["Lev_1"] == "L1_S1")
                & (tsa_obj.data["Lev_2"] == "L2_S1")
                & (tsa_obj.data["Lev_3"] == "L3_S1")
            ]
            computed_output = tsa_obj.show_time_repetitions(data=filtered_data)
            if isinstance(computed_output, str):
                str_list = ["No repetitions", "Segment data"]
                assert any(
                    subtext in computed_output for subtext in str_list
                ), f"'{computed_output}' not in {str_list}"
            else:
                assert type(computed_output) == holoviews.element.chart.Curve
