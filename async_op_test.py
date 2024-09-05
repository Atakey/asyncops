import time
import unittest
from unittest.mock import patch

from asyncops import async_execute, set_max_workers
from asyncops.async_op import _ExecutorManager, AsyncTask, clean_up


class TestAsyncOp(unittest.TestCase):

    def test_async_execute_basic(self):
        @async_execute()
        def example_function(x):
            return x * 2

        result = example_function(5)
        self.assertIsInstance(result, AsyncTask)
        self.assertEqual(result._get_result(), 10)

    def test_async_execute_with_timeout(self):
        @async_execute(timeout=1, default_value="Timeout")
        def slow_function():
            time.sleep(2)
            return "Done"

        result = slow_function()
        self.assertEqual(result._get_result(), "Timeout")

    def test_async_execute_with_exception(self):
        @async_execute()
        def error_function():
            raise ValueError("Test error")

        result = error_function()
        with self.assertRaises(RuntimeError):
            result._get_result()

    def test_set_max_workers(self):
        original_workers = _ExecutorManager.get_executor()._max_workers
        set_max_workers(4)
        self.assertEqual(_ExecutorManager.get_executor()._max_workers, 4)
        set_max_workers(original_workers)  # Reset to original value

    def test_async_task_then(self):
        @async_execute()
        def first_task():
            return 5

        result = first_task().then(lambda x: x * 2)
        self.assertEqual(result._get_result(), 10)

    def test_async_task_magic_methods(self):
        @async_execute()
        def number_task():
            return 5

        result = number_task()
        self.assertEqual(result + 3, 8)
        self.assertEqual(3 + result, 8)
        self.assertEqual(result * 2, 10)
        self.assertEqual(abs(-result), 5)

    @patch('asyncops.async_op.cache_exists')
    @patch('asyncops.async_op.cache_function_value')
    def test_async_execute_with_caching(self, mock_cache_function_value, mock_cache_exists):
        mock_cache_exists.return_value = (False, None)

        @async_execute(n_days_to_cache=1)
        def cacheable_function(x):
            return x * 2

        result = cacheable_function(5)
        self.assertEqual(result._get_result(), 10)
        mock_cache_function_value.assert_called_once()

    def test_async_task_context_manager(self):
        @async_execute()
        def example_function():
            return "test"

        with example_function() as result:
            self.assertEqual(result, "test")

    def test_clean_up(self):
        with patch.object(_ExecutorManager, 'shutdown_executor') as mock_shutdown:
            clean_up()
            mock_shutdown.assert_called_once_with(wait=True)


if __name__ == '__main__':
    unittest.main()
