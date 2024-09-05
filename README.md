# AsyncOps

AsyncOps is a Python library that provides asynchronous operation capabilities for both general Python functions and Pandas operations. It aims to improve performance and efficiency in data processing tasks by leveraging asynchronous execution and parallel processing.

## Features

### async_op.py

- `async_execute`: A decorator for executing functions asynchronously.
- `AsyncTask`: A wrapper class for asynchronous tasks, providing additional functionality and ease of use.
- Timeout handling for asynchronous operations.
- Optional caching of function results.
- Ability to chain multiple asynchronous operations.

### async_pandas.py

- `series_async_apply`: Asynchronously apply a function to a Pandas Series.
- `dataframe_async_apply`: Asynchronously apply a function to a Pandas DataFrame.
- `groupby_async_apply`: Asynchronously apply a function to a Pandas GroupBy object.
- Parallel processing capabilities for Pandas operations.
- Progress bar support for long-running operations.
- Optional caching of results for repeated operations.

## Installation

You can install AsyncOps using pip:

```
pip install asyncops
```

## Usage

### Basic Usage of async_op

```python
from asyncops.async_op import async_execute

@async_execute(timeout=5, default_value="Timeout occurred")
def long_running_task():
    # Some time-consuming operation
    return "Task completed"

result = long_running_task()
print(result._get_result())  # Will print "Task completed" or "Timeout occurred"
```

### Using async_pandas

```python
import pandas as pd
from asyncops.async_pandas import series_async_apply, dataframe_async_apply

# Series example
series = pd.Series([1, 2, 3, 4, 5])
result = series_async_apply(series, lambda x: x ** 2, show_progress_bars=True)
print(result._get_result())

# DataFrame example
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = dataframe_async_apply(df, lambda row: row['A'] * row['B'], axis=1)
print(result._get_result())
```

## Contributing

Contributions to AsyncOps are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.