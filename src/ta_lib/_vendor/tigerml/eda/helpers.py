import numpy as np
from tigerml.core.utils import compute_if_dask


def is_missing(iterable, na_values):
    if "Series" in str(type(iterable)) and "datetime" in str(
        iterable.dtype
    ):  # If datatime col, string values like 'NA' etc will raise format error.
        na_values = [val for val in na_values if not isinstance(val, str)]
    return iterable.isin(na_values) | (
        iterable.isna() if np.nan in na_values else False
    )


def split_sets(df, cols, by):
    if isinstance(by, int):
        row_dict = df.iloc[by][cols].to_dict()
        for key in row_dict.keys():
            row_dict[key] = str(row_dict[key])
    elif by == "mean":
        # means = []
        row_dict = {}
        num_cols = [col for col in cols if col in df.numeric_columns]
        # df[cols] = df[cols].replace([np.inf, -np.inf], np.NAN)
        # df = df.dropna(subset=cols, how='all')
        # for col in num_cols:
        #     second_max = list(df[col].nlargest(2))[1]
        #     df[col] = df[col].apply(lambda x: second_max if x == np.inf else x)
        #     second_min = list(df[col].nsmallest(2))[1]
        #     df[col] = df.apply(lambda x: x[col].replace(-np.inf, second_min), axis=1)
        #     df[col] = df[col].apply(lambda x: second_min if x == -np.inf else x)
        for col in num_cols:
            row_dict[col] = compute_if_dask(df[col].mean())
            # means.append(compute_if_dask(df[col].mean()))
        # row_dict = dict(zip(cols, means))
        cat_cols = [col for col in cols if col not in num_cols]
        row_dict.update(dict(zip(cat_cols, ["cat_cols"] * len(cat_cols))))
    unique_values = set(list(row_dict.values()))
    sets = [
        [key for key in row_dict.keys() if row_dict[key] == val]
        for val in unique_values
    ]
    return sets
