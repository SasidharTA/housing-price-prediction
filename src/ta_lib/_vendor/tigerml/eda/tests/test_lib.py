from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from tigerml.core.utils.constants import SUMMARY_KEY_MAP

from ..Analyser import Analyser

# FIXME: use hypothesis library to generate test data.
TEST_DATA = pd.DataFrame(
    {
        "integral": range(10),
        "text": ["\u018e", ""] + ["abc"] * 8,
        "timestamp1": (
            pd.date_range("2016-01-01", periods=10, freq="1D").strftime("%y-%m-%d")
        ),
        "timestamp2": pd.date_range("2016-01-01", periods=10, freq="1D"),
        # "object": [object()] + [object()] * 9,
        "numeric": np.arange(10).astype(int),
        # 'list': [[1,2]] * 10,
    }
)


class TestAnalyser(TestCase):
    """Test analyser class."""

    analyser = Analyser(TEST_DATA.copy(deep=True))

    def test_duplicate_columns(self):
        """Tests duplicate_columns function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.duplicate_columns()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        # verify duplicate columns
        self.assertIsNotNone(out)

        # check for the duplicate column value
        self.assertListEqual(out[SUMMARY_KEY_MAP.duplicate_col].tolist(), ["numeric"])

    def test_missing_value_summary(self):
        """Tests missing_value_summary function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.missing_value_summary()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        # check missing value result
        self.assertEqual(out, "No Missing Values")

    def test_correlation_table(self):
        """Tests correlation_table function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.correlation_table()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        # check for Correlation Coefficient value
        self.assertListEqual(out[SUMMARY_KEY_MAP.corr_coef].tolist(), [1.0])

    def test_non_numeric_summary(self):
        """Tests non numerical summary function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        self.analyser.non_numeric_summary()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

    def test_numeric_summary(self):
        """Tests numerical summary function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.numeric_summary()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        # check no. of unique values
        assert_allclose(out[SUMMARY_KEY_MAP.num_unique].tolist(), [10, 10])

        # check 75th percentile
        assert_allclose(out[SUMMARY_KEY_MAP.percentile_75].tolist(), [6.75, 6.75])

        # check max value
        assert_allclose(out[SUMMARY_KEY_MAP.max_value].tolist(), [9, 9])

    def test_variable_summary(self):
        """Tests variable_summary function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.variable_summary()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        # check all variable have a summary
        self.assertListEqual(
            out[SUMMARY_KEY_MAP.variable_names].tolist(), TEST_DATA.columns.tolist(),
        )

        # check unique counts
        self.assertListEqual(
            out[SUMMARY_KEY_MAP.num_unique].tolist(),
            [10, 3, 10, 10, 10]
            # [10, 3, 10, 10, 2, 10]
        )

        # check variable names
        self.assertListEqual(
            out[SUMMARY_KEY_MAP.variable_names].tolist(),
            ["integral", "text", "timestamp1", "timestamp2", "numeric"],
            # ["integral", "text", "timestamp1", "timestamp2", "object", "numeric"],
        )

        # check col dtypes
        exp_dtypes = TEST_DATA.dtypes
        self.assertListEqual(out[SUMMARY_KEY_MAP.dtype].tolist(), exp_dtypes.tolist())

    def test_data_health(self):
        """Tests data_health function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.data_health()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        self.assertIsNotNone(out)

    def test_missing_plot(self):
        """Tests missing_plot function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.missing_plot()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        self.assertIsNotNone(out)

    def test_non_numeric_frequency_plot(self):
        """Tests non_numeric_frequency_plot function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.non_numeric_frequency_plot(TEST_DATA)

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        self.assertIsNotNone(out)

    def test_percentile_plots(self):
        """Tests percentile_plots function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.percentile_plots()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        self.assertIsNotNone(out)

    def test_density_plots(self):
        """Tests density_plots function."""
        # copy_test_data = TEST_DATA.copy(deep=True)
        out = self.analyser.density_plots()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))

        self.assertIsNotNone(out)

    def test_bivariate_plots(self):
        """Tests bivariate_plots function."""
        # local_test = TEST_DATA.drop(["object"], axis=1)
        # copy_test_data = local_test.copy(deep=True)
        out = self.analyser.bivariate_plots()

        # verify object is not modified
        self.assertTrue(TEST_DATA.equals(self.analyser.data._data))
        # self.assertTrue(
        #     local_test.equals(self.analyser.data._data.drop(["object"], axis=1))
        # )

        self.assertIsNotNone(out)
