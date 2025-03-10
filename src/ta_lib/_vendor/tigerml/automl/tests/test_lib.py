import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from unittest import TestCase

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
    """Testlib class."""

    def test_handle_outliers(self):
        """Testlib class."""
        return 0
