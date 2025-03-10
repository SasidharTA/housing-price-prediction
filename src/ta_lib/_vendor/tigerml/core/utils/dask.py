import numpy as np
import pandas as pd


def is_dask_scalar(value):
    if (
        hasattr(value, "__module__")
        and value.__module__.startswith("dask")
        and value.__class__.__name__ == "Scalar"
    ):
        return True
    return False


def compute_if_dask(value):
    from tigerml.core.dataframe import DataFrame, Scalar, Series

    if isinstance(value, list) or isinstance(value, np.ndarray):
        value = [compute_if_dask(val) for val in value]
    elif isinstance(value, dict):
        # import pdb
        # pdb.set_trace()
        keys = value.keys()
        values = [x for x in value.values()]
        computed_values = compute_if_dask(values)
        value = dict(zip(keys, computed_values))
    elif isinstance(value, tuple):
        value = tuple(compute_if_dask(list(value)))
    elif isinstance(value, pd.Series):
        if str(value.dtype) == "category":
            if [x for x in value.dtype.categories if is_dask_scalar(x)]:
                value = value.apply(lambda s: compute_if_dask(s))
        elif "datetime" in str(value.dtype):
            value = value
        try:
            sum_value = value.sum()
            if is_dask_scalar(sum_value):
                value = value.apply(lambda s: compute_if_dask(s))
        except Exception:
            value = value
    elif isinstance(value, pd.DataFrame):
        # import pdb
        # pdb.set_trace()
        for col in value.columns:
            value[col] = compute_if_dask(value[col])
    elif hasattr(value, "__module__") and value.__module__.startswith("dask"):
        value = value.compute()
    elif isinstance(value, DataFrame) or isinstance(value, Series):
        value = value.compute()
    elif isinstance(value, Scalar) or is_dask_scalar(value):
        value = value.compute()
    return value


def persist_if_dask(value):
    from tigerml.core.dataframe import DataFrame, Series

    if hasattr(value, "__module__") and value.__module__.startswith("dask"):
        return value.persist()
    if isinstance(value, DataFrame) or isinstance(value, Series):
        return value.persist()
    return value
