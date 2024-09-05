import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from asyncops.async_op import AsyncTask
from asyncops.async_pandas import dataframe_async_apply, groupby_async_apply, hash_pandas, series_async_apply


class TestAsyncPandas(unittest.TestCase):

    def setUp(self):
        self.series = pd.Series([1, 2, 3, 4, 5])
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        self.grouped = self.df.groupby('A')

    def test_series_async_apply(self):
        result = series_async_apply(self.series, lambda x: x * 2)
        self.assertIsInstance(result, AsyncTask)
        output = result._get_result()
        self.assertIsInstance(output, pd.Series)
        np.testing.assert_array_equal(output.values, [2, 4, 6, 8, 10])

    def test_dataframe_async_apply(self):
        result = dataframe_async_apply(self.df, lambda row: row['A'] + row['B'], axis=1)
        self.assertIsInstance(result, AsyncTask)
        output = result._get_result()
        self.assertIsInstance(output, pd.Series)
        np.testing.assert_array_equal(output.values, [11, 22, 33, 44, 55])

    def test_groupby_async_apply(self):
        result = groupby_async_apply(self.grouped, lambda group: group['B'].sum())
        self.assertIsInstance(result, AsyncTask)
        output = result._get_result()
        self.assertIsInstance(output, pd.Series)
        self.assertEqual(output[1], 10)
        self.assertEqual(output[5], 50)

    @patch('asyncops.async_pandas.async_execute')
    def test_series_async_apply_with_caching(self, mock_async_execute):
        mock_async_execute.return_value = lambda func, **kwargs: AsyncTask(future=None,
                                                                           default_value=pd.Series([2, 4, 6, 8, 10]))

        result = series_async_apply(self.series, lambda x: x * 2, n_days_to_cache=1)
        output = result._get_result()

        mock_async_execute.assert_called_once()
        self.assertIsInstance(output, pd.Series)
        np.testing.assert_array_equal(output.values, [2, 4, 6, 8, 10])

    def test_series_async_apply_with_progress_bar(self):
        result = series_async_apply(self.series, lambda x: x * 2, show_progress_bars=True)
        output = result._get_result()
        self.assertIsInstance(output, pd.Series)
        np.testing.assert_array_equal(output.values, [2, 4, 6, 8, 10])

    def test_dataframe_async_apply_with_custom_workers(self):
        result = dataframe_async_apply(self.df, lambda row: row['A'] * row['B'], axis=1, workers=2)
        output = result._get_result()
        self.assertIsInstance(output, pd.Series)
        np.testing.assert_array_equal(output.values, [10, 40, 90, 160, 250])

    def test_series_async_apply_with_exception(self):
        def error_func(x):
            raise ValueError("Test error")

        result = series_async_apply(self.series, error_func)
        with self.assertRaises(RuntimeError):
            result._get_result()

    def test_hash_pandas(self):
        hash1 = hash_pandas(self.df)
        hash2 = hash_pandas(self.df.copy())
        self.assertEqual(hash1, hash2)

        modified_df = self.df.copy()
        modified_df.iloc[0, 0] = 100
        hash3 = hash_pandas(modified_df)
        self.assertNotEqual(hash1, hash3)


if __name__ == '__main__':
    unittest.main()
