#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
@Author: huanghailong
@contact: huanghailong@xiaoyouzi.com
@time: 2024/8/31 19:39
@desc:
"""
import atexit
import hashlib
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from functools import partial, wraps
from types import FunctionType
from typing import Any, Callable, Generic, Optional, TypeVar

from cache_to_disk import cache_exists, cache_function_value, load_cache_metadata_json

T = TypeVar('T')
__all__ = ['async_execute', 'set_max_workers']


class _ExecutorManager:
    """
    Global manager for handling the thread pool executor.
    This class ensures that only one thread pool executor instance is used throughout the application.
    """
    _executor = None
    _lock = threading.Lock()

    @classmethod
    def get_executor(cls, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """
        Returns the thread pool executor, creating it if it doesn't already exist.

        Args:
            max_workers (Optional[int]): The maximum number of worker threads. If not provided,
                                         it defaults to the value of the environment variable "ASYNC_OP_MAX_WORKERS"
                                         or 2 if the variable is not set.

        Returns:
            ThreadPoolExecutor: The executor instance.
        """
        with cls._lock:
            if cls._executor is None:
                max_workers = max_workers or int(os.environ.get("ASYNC_OP_MAX_WORKERS", 2))
                cls._executor = ThreadPoolExecutor(max_workers=max_workers)
            return cls._executor

    @classmethod
    def set_max_workers(cls, max_workers: int):
        """
        Sets the maximum number of worker threads for the executor.

        Args:
            max_workers (int): The new maximum number of worker threads.
        """
        with cls._lock:
            if cls._executor is not None:
                cls._executor.shutdown(wait=True)
            cls._executor = ThreadPoolExecutor(max_workers=max_workers)

    @classmethod
    def shutdown_executor(cls, wait: bool = True):
        """
        Shuts down the thread pool executor.

        Args:
            wait (bool): If True, wait for all currently executing tasks to finish before shutting down.
                         If False, the executor will shut down immediately. Defaults to True.
        """
        with cls._lock:
            if cls._executor is not None:
                cls._executor.shutdown(wait=wait)
                cls._executor = None


def set_max_workers(max_workers: int):
    """
    Set the maximum number of worker threads for the global thread pool executor.

    Args:
        max_workers (int): The new maximum number of worker threads.
    """
    if max_workers <= 0:
        raise ValueError('max_workers must be greater than 0.')
    _ExecutorManager.set_max_workers(max_workers)


def _generate_cache_key(func: Callable, *args, **kwargs) -> str:
    """
    Generate a cache key based on the function name, module, and its arguments.

    Args:
        func (Callable): The function to generate the key for.
        args (tuple): Positional arguments for the function.
        kwargs (dict): Keyword arguments for the function.

    Returns:
        str: A unique cache key based on the function and its arguments.
    """

    def serialize(obj):
        """
        Serialize function types, partial objects, and other objects into a string representation.
        """
        # For function types, return module and qualified name
        if isinstance(obj, FunctionType):
            return f"{obj.__module__}.{obj.__qualname__}"

        # For partial objects, serialize both the function and its arguments
        if isinstance(obj, partial):
            func_str = serialize(obj.func)  # Recursively serialize the function
            args_str = tuple(serialize(arg) for arg in obj.args)  # Serialize positional arguments
            kwargs_str = {key: serialize(value) for key, value in obj.keywords.items()}  # Serialize keyword arguments
            return f"partial({func_str}, args={args_str}, kwargs={kwargs_str})"

        # For all other objects, use str as fallback
        return str(obj)

    # Serialize the function name, args, and kwargs
    func_name = serialize(func)
    serialized_args = tuple(serialize(arg) for arg in args)
    serialized_kwargs = {key: serialize(value) for key, value in kwargs.items()}

    key = f"{func_name}:{serialized_args}:{serialized_kwargs}"

    return hashlib.md5(key.encode()).hexdigest()


def async_execute(
        func: Optional[Callable] = None,
        timeout: Optional[float] = None,
        default_value: Optional[T] = None,
        n_days_to_cache: int = 0,
        prefix_key: str = ""
) -> Callable:
    """
    Decorator to execute a function asynchronously with optional caching and timeout.

    Args:
        func (Optional[Callable]): The function to be executed asynchronously.
        timeout (Optional[float]): The maximum amount of time (in seconds) to wait for the result of the async task.
                                   If the task exceeds this time, the `default_value` is returned. Defaults to None.
        default_value (Optional[T]): The value to return if the task times out or the function result is unavailable.
                                     Defaults to None.
        n_days_to_cache (int): The number of days to cache the function result.
                               - If greater than 0, the function result is cached for that many days.
                               - If equal to 0, no caching is performed.
                               - If less than 0, the existing cache is forcibly reset, and the result is recomputed.
                               Defaults to 0 (no caching).
        prefix_key (str): An optional prefix to be added to the cache key. This can be useful for namespacing caches
                          in systems where multiple functions may share similar keys. Defaults to an empty string.

    Returns:
        Callable: A wrapped version of the function that executes asynchronously, handles caching, and respects the
                  specified timeout.

    Notes:
        - If `n_days_to_cache > 0`, the result of the function is cached for the specified number of days.
        - If a valid cache exists and `n_days_to_cache > 0`, the cached value is returned without executing the function.
        - If `n_days_to_cache < 0`, the cache is reset (i.e., the function is forced to recompute even if a cache exists).
        - The function is executed in a thread pool using the global executor managed by `_ExecutorManager`.
        - The cache key is generated based on the function's name, arguments, and the `prefix_key`.
        - Upon task completion, the result can be cached (depending on `n_days_to_cache`), and the function result is returned asynchronously as an `AsyncTask`.

    Example usage:

        @async_execute(timeout=10, n_days_to_cache=7)
        def my_function(x):
            return x ** 2

        # Execute the function asynchronously and cache the result for 7 days.
        task = my_function(4)

        # Get the result (blocking until the task completes or the timeout is reached).
        result = task.get_result()

    """
    prefix_key = prefix_key + "_" if prefix_key else ""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> "AsyncTask":

            if callable(func):
                original_func = func
            else:
                original_func = partial(func.__func__, func.__self__)

            if n_days_to_cache > 0:
                cache_metadata = load_cache_metadata_json()
                computed_cache_key = prefix_key + _generate_cache_key(func, *args, **kwargs)
                already_cached, function_value = cache_exists(cache_metadata, computed_cache_key)
                if already_cached:
                    return AsyncTask(default_value=function_value)

            future = _ExecutorManager.get_executor().submit(original_func, *args, **kwargs)

            if n_days_to_cache != 0:
                def cache_result(fut):
                    if fut.done() and not fut.cancelled():
                        function_value = fut.result()
                        # TODO: Add lock to ensure safe cache metadata write
                        cache_function_value(
                            function_value, abs(n_days_to_cache), cache_metadata, computed_cache_key
                        )

                future.add_done_callback(cache_result)

            return AsyncTask(future, timeout=timeout, default_value=default_value)

        return wrapper

    return decorator(func) if func else decorator


class AsyncTask(Generic[T]):
    """
    A wrapper for the Future object returned by the thread pool executor, providing additional functionality.
    """

    def __init__(
            self,
            future: Optional[Future] = None,
            timeout: Optional[float] = None,
            default_value: Optional[T] = None
    ):
        """
        Initialize the AsyncTask.

        Args:
            future (Future): The future object representing the result of an asynchronous task.
            timeout (Optional[float]): The maximum time (in seconds) to wait for the task to complete. Defaults to None.
            default_value (Optional[T]): The value to return if the task times out. Defaults to None.
        """

        self._future = future
        self._timeout = timeout
        if future is None:
            self._result = default_value
        else:
            self._result = None
        self._default_value = default_value

    def _get_result(self) -> T:
        """
        Retrieve the result of the asynchronous task.

        Returns:
            T: The result of the task.

        Raises:
            TimeoutError: If the task exceeds the specified timeout and no default_value is provided.
            KeyboardInterrupt: If the execution is interrupted by the user.
            RuntimeError: If an error occurs during task execution.
        """
        if self._result is None:
            try:
                self._result = self._future.result(timeout=self._timeout)
            except TimeoutError as e:
                if self._default_value is not None:
                    self._result = self._default_value
                else:
                    raise TimeoutError(f"Task timed out after {self._timeout} seconds.") from e
            except KeyboardInterrupt as e:
                clean_up(wait=False)
                raise KeyboardInterrupt("Execution was interrupted by the user.") from e
            except Exception as e:
                raise RuntimeError("An error occurred while executing the task.") from e
        return self._result

    def then(
            self,
            func: Callable[[T], Any],
            timeout: Optional[float] = None,
            default_value: Optional[Any] = None
    ) -> 'AsyncTask':
        """
        Chains the execution of another function, using the result of the current task as input.

        Args:
            func (Callable[[T], Any]): The function to execute next, using the result of the current task.
            timeout (Optional[float]): The maximum time (in seconds) to wait for the task to complete. Defaults to None.
            default_value (Optional[Any]): The value to return if the task times out. Defaults to None.

        Returns:
            AsyncTask: The new AsyncResult wrapping the result of the next task.
        """

        def wrapped_func(result: T):
            return func(result)

        next_async_execute = async_execute(timeout=timeout, default_value=default_value)
        next_future = _ExecutorManager.get_executor().submit(next_async_execute(wrapped_func), self._get_result())
        return AsyncTask(next_future, timeout=timeout, default_value=default_value)

    def __getattr__(self, name: str) -> Any:
        # Proxy methods to interact with the result as if interacting directly with the result itself
        return getattr(self._get_result(), name)

    def __str__(self) -> str:
        return str(self._get_result())

    def __repr__(self) -> str:
        return repr(self._get_result())

    def __bool__(self) -> bool:
        return bool(self._get_result())

    def __eq__(self, other):
        if isinstance(other, AsyncTask):
            return self._get_result() == other._get_result()
        return self._get_result() == other

    def __format__(self, format_spec):
        return format(self._get_result(), format_spec)

    def __enter__(self) -> T:
        # Explicitly wait for the result
        return self._get_result()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        # Attempt to cancel the async task when the object is destroyed
        if self._result is None and not self._future.done():
            self._future.cancel()

    # Magic methods for arithmetic operations

    def __add__(self, other):
        return self._get_result() + other

    def __radd__(self, other):
        return other + self._get_result()

    def __iadd__(self, other):
        self._result = self._get_result() + other
        return self

    def __sub__(self, other):
        return self._get_result() - other

    def __rsub__(self, other):
        return other - self._get_result()

    def __isub__(self, other):
        self._result = self._get_result() - other
        return self

    def __mul__(self, other):
        return self._get_result() * other

    def __rmul__(self, other):
        return other * self._get_result()

    def __imul__(self, other):
        self._result = self._get_result() * other
        return self

    def __truediv__(self, other):
        return self._get_result() / other

    def __rtruediv__(self, other):
        return other / self._get_result()

    def __itruediv__(self, other):
        self._result = self._get_result() / other
        return self

    def __floordiv__(self, other):
        return self._get_result() // other

    def __rfloordiv__(self, other):
        return other // self._get_result()

    def __ifloordiv__(self, other):
        self._result = self._get_result() // other
        return self

    def __mod__(self, other):
        return self._get_result() % other

    def __rmod__(self, other):
        return other % self._get_result()

    def __imod__(self, other):
        self._result = self._get_result() % other
        return self

    def __pow__(self, other):
        return self._get_result() ** other

    def __rpow__(self, other):
        return other ** self._get_result()

    def __ipow__(self, other):
        self._result = self._get_result() ** other
        return self

    # Bitwise operations
    def __and__(self, other):
        return self._get_result() & other

    def __rand__(self, other):
        return other & self._get_result()

    def __iand__(self, other):
        self._result = self._get_result() & other
        return self

    def __or__(self, other):
        return self._get_result() | other

    def __ror__(self, other):
        return other | self._get_result()

    def __ior__(self, other):
        self._result = self._get_result() | other
        return self

    def __xor__(self, other):
        return self._get_result() ^ other

    def __rxor__(self, other):
        return other ^ self._get_result()

    def __ixor__(self, other):
        self._result = self._get_result() ^ other
        return self

    def __lshift__(self, other):
        return self._get_result() << other

    def __rlshift__(self, other):
        return other << self._get_result()

    def __ilshift__(self, other):
        self._result = self._get_result() << other
        return self

    def __rshift__(self, other):
        return self._get_result() >> other

    def __rrshift__(self, other):
        return other >> self._get_result()

    def __irshift__(self, other):
        self._result = self._get_result() >> other
        return self

    # Unary operations
    def __neg__(self):
        return -self._get_result()

    def __pos__(self):
        return +self._get_result()

    def __abs__(self):
        return abs(self._get_result())

    def __invert__(self):
        return ~self._get_result()

    # Container methods
    def __len__(self) -> int:
        return len(self._get_result())

    def __getitem__(self, key):
        return self._get_result()[key]

    def __setitem__(self, key, value):
        self._get_result()[key] = value

    def __delitem__(self, key):
        del self._get_result()[key]

    def __iter__(self):
        return iter(self._get_result())

    def __contains__(self, item):
        return item in self._get_result()

    # Callable support
    def __call__(self, *args, **kwargs):
        return self._get_result()(*args, **kwargs)


def clean_up(wait: bool = True):
    """
    Cleanup function to shut down the executor, typically called at program exit.

    Args:
        wait (bool): If True, wait for all currently executing tasks to finish before shutting down.
                     If False, the executor will shut down immediately. Defaults to True.
    """
    _ExecutorManager.shutdown_executor(wait=wait)


# Register the cleanup function to ensure it runs when the program exits
atexit.register(clean_up)
