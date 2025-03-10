import numpy as np
import pandas as pd


def sort(list):
    strings = [x for x in list if isinstance(x, str)]
    numbers = [x for x in list if not isinstance(x, str)]
    return sorted(numbers) + sorted(strings)


def is_numeric(dtype):
    return issubclass(dtype.type, np.number)


def get_num_cols(df):
    if df.empty:
        return list()
    normal_num_cols = list(df.select_dtypes(include=np.number).columns)
    obj_type_num_cols = []
    for col in df.columns:
        try:
            df[col] + 1  # This operation fails for non-numeric columns
            obj_type_num_cols += [col]
        except TypeError:
            pass
    obj_type_num_cols = [col for col in obj_type_num_cols if col not in normal_num_cols]
    return sort(normal_num_cols + obj_type_num_cols)


def get_non_num_cols(df):
    if df.empty:
        return list()
    num_cols = get_num_cols(df)
    return sort(list(set(df.columns) - set(num_cols)))


def get_cat_cols(df):
    if df.empty:
        return list()
    num_cols = get_num_cols(df) + get_bool_cols(df)
    dt_cols = get_dt_cols(df)
    return sort(list(set(df.columns) - set(num_cols) - set(dt_cols)))


def get_dt_cols(df):
    if df.empty:
        return list()
    return sort(list(df.select_dtypes(include=[np.datetime64]).columns))


def get_bool_cols(df):
    if df.empty:
        return list()
    dtypes = df.dtypes
    bool_type_1 = dtypes[dtypes == "bool"].index.tolist()
    bool_type_2 = [
        col
        for col in df.columns
        if set(df[col].unique().tolist()) <= {0, 1} and col not in bool_type_1
    ]
    return sort(bool_type_1 + bool_type_2)


def convert_to_dt(df):
    mask = df.astype(str).apply(
        lambda x: x.str.match(r"(\d{2,4}-\d{2}-\d{2,4})+").all()
    )
    if mask.sum() > 0:
        print(
            "Detected {} columns as datetime format. Converting to datetime".format(
                list(mask[mask is True].index)
            )
        )
        df.loc[:, mask] = df.loc[:, mask].apply(pd.to_datetime)
    return df


def is_discrete(series_data):
    is_discrete = str(
        series_data.dtype
    ) == "object" or series_data.nunique() < 0.05 * len(series_data)
    return is_discrete


def reduce_mem_usage(df, ctg_thresh=0.25, verbose=True):
    """Reduce memory usage.

    Numerical Type
    - Based on max range of column limiting datatype to np.int8 or np.int16 or
      np.int32 or np.float16 or np.float32
    Object type
    - https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#categorical-memory
    - The memory usage of a Categorical is proportional to the number of
      categories plus the length of the data.
    - In contrast, an object dtype is a constant times the length of the data.
    - If the number of categories approaches the length of the data,
      the Categorical will use nearly the same or more memory than an equivalent
      object dtype representation.
    - Hence by default converting object to category only if unique values are
      25% of total value
    - Example
    -- df = reduce_mem_usage(df_train, 0.4, True)
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / (1024**2)
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        if col_type == "object":
            if df[col].nunique() / df.shape[0] <= ctg_thresh:
                df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / (1024**2)
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
