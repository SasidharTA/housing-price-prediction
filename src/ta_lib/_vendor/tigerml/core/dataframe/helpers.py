import numpy as np
import pandas as pd
from tigerml.core.utils import DictObject

BACKENDS = DictObject({"pandas": "pandas", "dask": "dask", "vaex": "vaex"})


def convert_to_tiger_assets(value):
    from tigerml.core.dataframe import BACKENDS, DataFrame, Scalar, Series

    if (
        hasattr(value, "__module__")
        and value.__module__.split(".")[0] in BACKENDS.keys()
    ):
        if value.__class__.__name__ == "Series":
            # import pdb
            # pdb.set_trace()
            value = Series(value)
        elif value.__class__.__name__ == "DataFrame":
            value = DataFrame(value)
        elif value.__class__.__name__ == "Scalar":
            value = Scalar(value)
        elif "groupby" in value.__class__.__name__.lower():
            value = TigerWrapper(value)
    return value


class TigerWrapper:
    """Tiger wrapper class."""

    def __init__(self, class_obj):
        self._data = class_obj

    def __getattr__(self, item):
        attr = getattr(self._data, item)
        if (
            callable(attr)
            and getattr(self._data.__class__, item).__class__.__name__ != "property"
        ):
            # result = attr(*args, **kwargs)
            attr = tigerify(attr)
        else:
            attr = convert_to_tiger_assets(attr)
        return attr

    def __getitem__(self, item):
        return convert_to_tiger_assets(self._data.__getitem__(item))


def detigerify_inputs(args=tuple(), kwargs={}):
    new_args = []
    new_kwargs = {}
    for arg in args:
        arg = detigerify(arg)
        new_args.append(arg)
    new_args = tuple(new_args)
    for key in kwargs:
        new_kwargs[key] = detigerify(kwargs[key])
    return new_args, new_kwargs


def tigerify_function(func_obj):
    def inner(*args, **kwargs):
        args, kwargs = detigerify_inputs(args, kwargs)
        result = func_obj(*args, **kwargs)
        result = convert_to_tiger_assets(result)
        return result

    return inner


def tigerify(callable_obj):
    if not callable(callable_obj):
        return convert_to_tiger_assets(callable_obj)
    elif callable_obj.__class__.__name__ not in ["method", "function"]:
        return TigerWrapper(callable_obj)
    else:
        return tigerify_function(callable_obj)


def create_func(func, df, args, kwargs):
    # import pdb
    # pdb.set_trace()
    return func(df, *args, **kwargs)


def daskify_pandas(dask_df, func):
    def inner(*args, **kwargs):
        import dask

        result = dask_df.map_partitions(lambda df: create_func(func, df, args, kwargs))
        result = convert_to_tiger_assets(dask.compute(result)[0])
        return result

    return inner


def get_module(data):
    if not hasattr(data, "__module__"):
        return BACKENDS.pandas
    if data.__module__.startswith("tigerml.core.dataframe"):
        return "tigerml"
    if data.__module__.startswith("pandas"):
        return BACKENDS.pandas
    if data.__module__.startswith("dask"):
        return BACKENDS.dask
    if data.__module__.startswith("vaex"):
        return BACKENDS.vaex


def is_series(data):
    module = get_module(data)
    if module == BACKENDS.pandas:
        return (
            isinstance(data, pd.Series)
            or isinstance(data, list)
            or (isinstance(data, np.ndarray) and len(data.shape) == 1)
        )
    if module == BACKENDS.dask:
        import dask.dataframe as dd

        return isinstance(data, dd.Series)
    if module == "tigerml":
        from .dataframe import Series

        return isinstance(data, Series)
    else:
        return False


def is_dask(value):
    return get_module(value) == "dask" or (
        get_module(value) == "tigerml" and value.backend == "dask"
    )


def convert_series_to_df(data, **kwargs):
    module = get_module(data)
    assert is_series(data), "Passed data is not a series"
    from .dataframe import BACKENDS

    if module == BACKENDS.dask:
        data = data.to_frame()
    elif module == BACKENDS.pandas:
        data = pd.DataFrame(data, **kwargs)
    elif module == "tigerml":
        from .dataframe import DataFrame

        data = convert_series_to_df(data._data)
        data = DataFrame(data)
    else:
        raise Exception("Given data module not supported")
    return data


def detigerify(data):
    if (
        hasattr(data, "__module__")
        and data.__module__.startswith("tigerml.core.dataframe")
        and not callable(data)
    ):
        return data._data
    return data


def is_date(series_data):
    if "datetime" in str(series_data.dtype) and all(
        series_data.dt.time.astype(str) == "00:00:00"
    ):
        return True
    return False


def get_formatted_values(series):
    if is_date(series):
        return series.dt.date.astype(str).values.tolist()
    else:
        return series.values.tolist()
