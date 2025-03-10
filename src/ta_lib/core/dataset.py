"""Utility functions for working with datasets."""

import posixpath as pp

import os
import pandas as pd
import re
import warnings
from openpyxl import load_workbook

import ta_lib.core.io as io
import ta_lib.core.utils as utils


def _key_to_tuple(key):
    key = pp.normpath(key)
    if key.startswith("/"):
        key = key[1:]
    if key.endswith("/"):
        key = key[:-1]
    return key.split("/")


def _get_val(dct, key):
    key_tple = _key_to_tuple(key)
    val = dct
    for key in key_tple:
        val = val[key]
    return val


def list_datasets(context, prefix="/"):
    """List datasets available in the data_catalog."""

    def _get_datasets(dct, prefix, _datasets):
        if "type" in dct:
            _datasets.append(prefix)
            return

        for k, v in dct.items():
            if isinstance(v, dict):
                _get_datasets(v, pp.join(prefix, k), _datasets)

    datasets = []
    _get_datasets(context.data_catalog["datasets"], "/", datasets)
    datasets = [item for item in datasets if prefix in item]
    return datasets


def _get_uri_from_template(path, args):
    """Parse the uri based on the args provided."""
    # ${core.data_base_path} are already been expanded in the path
    fields = re.findall(r"\{([^\}]*)\}", path)
    if fields:
        if "" in fields:
            raise (
                ValueError(
                    f"NUll String Exists between the paranthesis in the URI - {path}"
                )
            )
        _set_analyse_ = utils.setanalyse(fields, args.keys(), simplify=False)
        if _set_analyse_["A-B"]:
            raise (
                ValueError(
                    f"Following arguments are required but missing - {_set_analyse_['A-B']}"
                )
            )
        if _set_analyse_["B-A"]:
            warnings.warn(
                f"Invalid arguments provided - {_set_analyse_['B-A']} :: Ignoring them .."
            )
            args = {k: v for k, v in args.items() if k in fields}
        return path.format(**args)
    else:
        return path


def load_dataset(context, key, skip=False, **kwargs):
    """Return a dataset from the data_catalog."""
    try:
        ds = _get_val(context.data_catalog["datasets"], key)
        ds["uri"] = _get_uri_from_template(ds["uri"], kwargs)
    except KeyError:
        avlb_keys = list_datasets(context)
        raise ValueError(
            f"Invalid dataset key: {key}. \n\nAvailable datasets: {avlb_keys}"
        )

    load_params = ds.get("driver_params", {}).get("load", {})
    fs = io.fs(context, ds["uri"], ds.get("credential_id"))

    data_uri = fs.glob(ds["uri"])
    df = utils.load_data(data_uri[0], fs=fs, **load_params)
    cols = set(df.columns)
    for uri_ in data_uri[1:]:
        try:
            temp_df = utils.load_data(uri_, fs=fs, **load_params)
            if cols != set(temp_df.columns):
                raise ValueError(f"{uri_} columns don't match.")
            df = df.append(temp_df)
        except Exception as e:
            if skip:
                warnings.warn(f"Error: {e}")
                warnings.warn(f"skipping {uri_}")
                continue
            raise e
    return df


def save_dataset(context, df, key, **kwargs):
    """Return a dataset from the data_catalog."""
    try:
        ds = _get_val(context.data_catalog["datasets"], key)
        ds["uri"] = _get_uri_from_template(ds["uri"], kwargs)
    except KeyError:
        avlb_keys = list_datasets(context)
        raise ValueError(
            f"Invalid dataset key: {key}. \n\nAvailable datasets: {avlb_keys}"
        )
    fs = io.fs(context, ds["uri"], ds.get("credential_id"))
    save_params = ds.get("driver_params", {}).get("save", {})
    return utils.save_data(df, ds["uri"], fs=fs, **save_params)


def save_to_excel(df, path, sheet_name):
    """Save a pandas DataFrame to an Excel file at the specified path.

    It takes Dataframe as input and the path of the excel file where the excel should be generated.

    Parameters
    ----------
    Dataframe : pandas.DataFrame
                The pandas DataFrame to be saved.
    path : str
           The path of the Excel file where the DataFrame will be saved. Must include the file name.
    sheet_name : str
                 The name of the Excel sheet where the DataFrame will be saved.

    Examples
    --------
    >>> data = pd.DataFrame({'Label': [0, 0, 1, 1], 'x': [1, 2, 3, 4], 'y': [4, 5, 6, 7]})
    >>> save_to_excel(data, './example.xlsx', 'Sheet1')
    """
    writer = pd.ExcelWriter(path, engine="openpyxl")
    if os.path.isfile(path):
        writer.book = load_workbook(path)
    df.to_excel(writer, sheet_name=sheet_name)
    writer.save()
    writer.close()
