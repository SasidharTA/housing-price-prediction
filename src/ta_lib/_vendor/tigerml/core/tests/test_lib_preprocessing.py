from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from tigerml.core.preprocessing import *

# FIXME: use hypothesis library to generate test data.
TEST_DATA = pd.DataFrame(
    {
        "integral": range(10),
        "text": ["\u018e", ""] + ["abc"] * 8,
        "timestamp1": (
            pd.date_range("2016-01-01", periods=10, freq="1D").strftime("%y-%m-%d")
        ),
        "timestamp2": pd.date_range("2016-01-01", periods=10, freq="1D"),
        "object": [object()] + [object()] * 9,
        "numeric": np.arange(10).astype(float),
        # 'list': [[1,2]] * 10,
    }
)


class TestLib(TestCase):
    """Test lib class."""

    def test_handle_outliers(self):
        """Test handle_outliers."""
        return 0

    def test_string_cleaning(self):
        """Test string_cleaning."""
        return 0

    def test_string_diff(self):
        """Test string_diff."""
        return 0

    def test_binning(self):
        """Test binning."""
        return 0

    def test_read_files_in_dir(self):
        """Test read_files_in_dir."""
        return 0
