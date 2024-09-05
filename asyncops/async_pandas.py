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


if __name__ == '__main__':
    import time

    dataf = pd.DataFrame({
        'text': ['This is a sentence.', 'Another sentence.', 'Yet another one.']
    })

    task1 = series_async_apply(dataf['text'], len)
    dataf['text_len'] = task1._get_result()

    dataf = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': [5, 6, 7, 8]
    })

    # Apply function to each row
    task2 = dataframe_async_apply(dataf, lambda row: row['col1'] + row['col2'], axis=1)
    dataf['sum'] = task2._get_result()


    def text_clean(text: str):
        """
        对文本进行清洗
        :param text: 输入的文本
        :return: 清洗后的文本
        """
        import re
        from jionlp import clean_text
        text = clean_text(text, remove_parentheses=False)
        text = re.sub('<(img|image)[^>]*(alt|desc|data-name|title)="([^"]+)"[^>]*?/>',
                      lambda x: '[' + x.group(3).strip() + ']', text)
        text = re.sub('<em\s[^>]*(class|style|channel_name|alt)="([^"]*)"[^>]*?>', "[图片]", text)
        text = re.sub(
            '(\s)(\s+)',
            lambda t: '\n' if re.search('\n', t.group(2)) else t.group(1),
            text
        )
        return text.strip()


    dataf = pd.read_csv(
        r"../../search-content-query-intent-classifier-service\tests\20240830_pool_topic_corpus.csv.gz",
        usecols=['content'], nrows=20_0000).fillna("")
    dataf1 = dataf[:10_0000]
    dataf2 = dataf[10_0000:]

    dataf1['t7'] = series_async_apply(dataf1['content'], text_clean, n_days_to_cache=1)._get_result()
    dataf2['t7'] = series_async_apply(dataf2['content'], text_clean, n_days_to_cache=1)._get_result()
    print(sum(x1 == x2 for x1, x2 in zip(dataf1['t7'], dataf2['t7'])), len(dataf1))

    begin = time.time()
    task7 = series_async_apply(dataf['content'], text_clean, n_days_to_cache=1)
    dataf['t7'] = task7._get_result()
    print(time.time() - begin)

    begin = time.time()
    dataf['t8'] = dataf['content'].apply(text_clean)
    print(time.time() - begin)
