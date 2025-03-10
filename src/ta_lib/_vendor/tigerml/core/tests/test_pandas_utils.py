from hypothesis import given, settings
from hypothesis.extra.pandas import columns, data_frames
from hypothesis.strategies import (
    booleans,
    characters,
    dates,
    floats,
    integers,
    text,
    tuples,
)

from tigerml.core.utils.pandas import *


class TestPandas:
    """Test pandas class ."""

    def fixed_given(self):
        """Tests pandas class ."""
        return given(
            test_df=data_frames(
                columns=columns(
                    [
                        "float_col1",
                        "string_col1",
                        "date_col1",
                        "bool_col1",
                        "int_col1",
                        "cat_col1",
                        "bool_col2",
                    ],
                    dtype=float,
                ),
                rows=tuples(
                    floats(allow_nan=True, allow_infinity=True),
                    text(),
                    dates(),
                    booleans(),
                    integers(),
                    characters(),
                    booleans(),
                ),
            )
        )(self)

    # -----------------------------------------------------------------------------
    # ---------------------- test_get_bool_cols testing ---------------------------
    @settings(max_examples=1, deadline=None)
    @fixed_given
    def test_get_bool_cols(self, test_df):
        """Test get_bool_cols."""
        bool_cols = get_bool_cols(test_df)
        if not test_df.empty:
            assert len(bool_cols) == 2
        else:
            assert len(bool_cols) == 0

    # ---------------------------------------------------------------------------
    # ---------------------- test_get_num_cols testing --------------------------
    @settings(max_examples=1, deadline=None)
    @fixed_given
    def test_get_num_cols(self, test_df):
        """Test get_num_cols."""
        num_cols = get_num_cols(test_df)
        if not test_df.empty:
            assert len(num_cols) == 2
            assert num_cols == ["float_col1", "int_col1"]

        else:
            assert len(num_cols) == 0

    # -------------------------------------------------------------------------------
    # ---------------------- test_get_non_num_cols testing --------------------------
    @settings(max_examples=1, deadline=None)
    @fixed_given
    def test_get_non_num_cols(self, test_df):
        """Test get_non_num_cols."""
        non_num_cols = get_non_num_cols(test_df)
        if not test_df.empty:
            assert len(non_num_cols) == 5
            assert non_num_cols == [
                "string_col1",
                "date_col1",
                "bool_col1",
                "cat_col1",
                "bool_col2",
            ]
        else:
            assert len(non_num_cols) == 0

    # --------------------------------------------------------------------------
    # ------------------- test_get_dt_cols testing -----------------------------
    @settings(max_examples=1, deadline=None)
    @fixed_given
    def test_get_dt_cols(self, test_df):
        """Test get_dt_cols."""
        dt_cols = get_dt_cols(test_df)
        if not test_df.empty:
            assert len(dt_cols) == 1
            assert dt_cols == ["date_col1"]
        else:
            assert len(dt_cols) == 0

    # ---------------------------------------------------------------------------
    # --------------------- test_get_cat_cols testing ---------------------------
    @settings(max_examples=1, deadline=None)
    @fixed_given
    def test_get_cat_cols(self, test_df):
        """Test get_cat_cols."""
        cat_cols = get_cat_cols(test_df)
        if not test_df.empty:
            assert len(cat_cols) == 1
            assert cat_cols == ["cat_col1"]
        else:
            assert len(cat_cols) == 0
