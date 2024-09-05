#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
@Author: huanghailong
@contact: huanghailong@xiaoyouzi.com
@time: 2024/9/2 11:14
@desc: 
"""
import hashlib
from typing import Callable

import pandas as pd
import psutil
from pandarallel.core import parallelize_with_pipe
from pandarallel.progress_bars import ProgressBarsType

from .async_op import async_execute, AsyncTask

NB_PHYSICAL_CORES = psutil.cpu_count(logical=False)


def hash_pandas(data) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()


def series_async_apply(
        series: pd.Series,
        func: Callable,
        show_progress_bars: bool = False,
        workers: int = NB_PHYSICAL_CORES,
        n_days_to_cache: int = 0,
        **kwargs
) -> AsyncTask:
    """
    Asynchronously applies a function to a pandas Series in parallel with optional caching.

    Args:
        series (pd.Series): The pandas Series on which the function will be applied.
        func (Callable): The function to apply to each element of the Series.
        show_progress_bars (bool): If True, shows a progress bar during the parallel execution. Defaults to False.
        workers (int): The number of parallel workers to use for applying the function.
                       Defaults to the number of physical CPU cores (NB_PHYSICAL_CORES).
        n_days_to_cache (int): The number of days to cache the result of the applied function.
                               - If greater than 0, the result is cached for `n_days_to_cache` days.
                               - If 0, no caching is performed.
                               - If less than 0, forces a cache reset, recomputing the result.
                               Defaults to no cache.
        **kwargs: Additional keyword arguments to pass to the function `func`.

    Returns:
        AsyncTask: An `AsyncTask` object that wraps the result of the asynchronous operation.
                   The result of the applied function can be retrieved using the `AsyncTask`.

    Notes:
        - The function is applied in parallel using the `pandarallel` library, which optimizes the task distribution
          across multiple CPU cores.
        - Caching is handled asynchronously. If the result is already cached and valid for the specified cache duration,
          the cached result will be returned without recomputing the function.
        - If a cache miss occurs or `n_days_to_cache` is set to a value less than or equal to zero, the function is
          recomputed, and the result is cached for future calls.
        - The task is executed using a global thread pool managed by `_ExecutorManager`.
        - Progress bars can be enabled for better visibility during execution, especially with large datasets.

    Example usage:

        # Define a function to apply to each element of the Series
        def my_func(x):
            return x ** 2

        # Apply the function asynchronously in parallel
        result_task = series_async_apply(my_series, my_func, show_progress_bars=True, workers=4)

        # Retrieve the result once the task is complete
        result = result_task.get_result()
    """
    if workers <= 1:
        future = async_execute(series.apply, n_days_to_cache=n_days_to_cache)(func, **kwargs)
        return future
    from pandarallel.data_types import Series
    parallelize = parallelize_with_pipe
    progress_bars = (
        ProgressBarsType.InUserDefinedFunction
        if show_progress_bars
        else ProgressBarsType.No
    )
    pd.Series.parallel_apply = parallelize(workers, Series.Apply, progress_bars)

    future = async_execute(series.parallel_apply, n_days_to_cache=n_days_to_cache)(func, **kwargs)
    return future


def dataframe_async_apply(
        df: pd.DataFrame,
        func: Callable,
        show_progress_bars: bool = False,
        workers: int = NB_PHYSICAL_CORES,
        n_days_to_cache: int = 0,
        **kwargs
) -> AsyncTask:
    """
    Asynchronously applies a function to a pandas DataFrame in parallel with optional caching.

    Args:
        df (pd.DataFrame): The pandas DataFrame on which the function will be applied.
        func (Callable): The function to apply to each element or row of the DataFrame.
                         It should be compatible with pandas' `apply` function.
        show_progress_bars (bool): If True, shows a progress bar during the parallel execution. Defaults to False.
        workers (int): The number of parallel workers to use for applying the function.
                       Defaults to the number of physical CPU cores (NB_PHYSICAL_CORES).
        n_days_to_cache (int): The number of days to cache the result of the applied function.
                               - If greater than 0, the result is cached for `n_days_to_cache` days.
                               - If 0, no caching is performed.
                               - If less than 0, forces a cache reset, recomputing the result.
                               Defaults to 0 (no caching).
        **kwargs: Additional keyword arguments to pass to the `func`.

    Returns:
        AsyncTask: An `AsyncTask` object that wraps the result of the asynchronous operation.
                   The result of the applied function can be retrieved using the `AsyncTask`.

    Notes:
        - The function is applied in parallel using the `pandarallel` library, which distributes the task across
          multiple CPU cores for efficient processing.
        - If `n_days_to_cache > 0`, the result is cached for the specified number of days.
        - If the cache is valid, the cached result is returned immediately without re-executing the function.
        - If a cache miss occurs, the function is re-executed, and the result is cached for future use.
        - Task execution is handled using a global thread pool managed by `_ExecutorManager`.
        - Progress bars can be enabled for monitoring long-running operations, particularly useful with large datasets.

    Example usage:

        # Define a function to apply to each row of the DataFrame
        def my_func(row):
            return row.sum()

        # Apply the function asynchronously in parallel
        result_task = dataframe_async_apply(my_dataframe, my_func, show_progress_bars=True, workers=4)

        # Retrieve the result once the task is complete
        result = result_task.get_result()
    """
    if workers <= 1:
        future = async_execute(df.apply, n_days_to_cache=n_days_to_cache)(func, **kwargs)
        return future
    from pandarallel.data_types import DataFrame
    parallelize = parallelize_with_pipe
    progress_bars = (
        ProgressBarsType.InUserDefinedFunction
        if show_progress_bars
        else ProgressBarsType.No
    )
    pd.DataFrame.parallel_apply = parallelize(workers, DataFrame.Apply, progress_bars)

    future = async_execute(df.parallel_apply, n_days_to_cache=n_days_to_cache)(func, **kwargs)
    return future


def groupby_async_apply(
        df: pd.core.groupby.DataFrameGroupBy,
        func: Callable,
        show_progress_bars: bool = False,
        workers: int = NB_PHYSICAL_CORES,
        n_days_to_cache: int = 0,
        **kwargs
) -> AsyncTask:
    """
    Asynchronously applies a function to a pandas DataFrameGroupBy object in parallel with optional caching.

    Args:
        df (pd.core.groupby.DataFrameGroupBy): The pandas DataFrameGroupBy object to apply the function on.
        func (Callable): The function to apply to each group within the DataFrameGroupBy object.
                         It should be compatible with pandas' `apply` method.
        show_progress_bars (bool): If True, shows a progress bar during the parallel execution. Defaults to False.
        workers (int): The number of parallel workers to use for applying the function.
                       Defaults to the number of physical CPU cores (NB_PHYSICAL_CORES).
        n_days_to_cache (int): The number of days to cache the result of the applied function.
                               - If greater than 0, the result is cached for `n_days_to_cache` days.
                               - If 0, no caching is performed.
                               - If less than 0, forces a cache reset, recomputing the result.
                               Defaults to 0 (no caching).
        **kwargs: Additional keyword arguments to pass to the `func`.

    Returns:
        AsyncTask: An `AsyncTask` object that wraps the result of the asynchronous operation.
                   The result of the applied function can be retrieved using the `AsyncTask`.

    Notes:
        - The function is applied in parallel using the `pandarallel` library, which distributes the task across
          multiple CPU cores for efficient processing.
        - If `n_days_to_cache > 0`, the result is cached for the specified number of days.
        - If a valid cache exists, the cached result is returned immediately without re-executing the function.
        - If a cache miss occurs, the function is re-executed, and the result is cached for future use.
        - Task execution is handled using a global thread pool managed by `_ExecutorManager`.
        - Progress bars can be enabled for monitoring long-running operations, particularly useful with large datasets.

    Example usage:

        # Define a function to apply to each group in the grouped DataFrame
        def my_group_func(group):
            return group.sum()

        # Group the DataFrame by a column and apply the function asynchronously
        grouped_df = my_dataframe.groupby("column_name")
        result_task = groupby_async_apply(grouped_df, my_group_func, show_progress_bars=True, workers=4)

        # Retrieve the result once the task is complete
        result = result_task.get_result()
    """
    if workers <= 1:
        future = async_execute(df.apply, n_days_to_cache=n_days_to_cache)(func, **kwargs)
        return future
    from pandarallel.data_types import DataFrameGroupBy
    parallelize = parallelize_with_pipe
    progress_bars = (
        ProgressBarsType.InUserDefinedFunction
        if show_progress_bars
        else ProgressBarsType.No
    )
    pd.core.groupby.DataFrameGroupBy.parallel_apply = parallelize(workers, DataFrameGroupBy.Apply, progress_bars)

    future = async_execute(df.parallel_apply, n_days_to_cache=n_days_to_cache)(func, **kwargs)
    return future
