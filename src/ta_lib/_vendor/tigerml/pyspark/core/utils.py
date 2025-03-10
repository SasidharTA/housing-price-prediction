"""Collection of utility functions."""

import fsspec
import importlib
import janitor  # noqa
import logging
import numpy as np
import os
import os.path as op  # noqa
import pandas as pd
import panel as pn
import random
import tigerml
import yaml
from contextlib import contextmanager  # noqa

logger = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True


def time_now_readable():
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_extension_of_path(path):
    _, ext = os.path.splitext(os.path.basename(path))
    return ext


def append_file_to_path(path, file_name):
    if get_extension_of_path(path):
        return path
    else:
        return os.path.join(path, file_name)


def load_class(qual_cls_name):
    """Load the class/object from string specified.

    Parameters
    ----------
        qual_cls_name - str
            string representing the class/estimator

    Returns
    -------
        class/Object

    Example
    -------
        >>> load_class('pyspark.ml.regression.RandomForestRegressor')
        RandomForestRegressor
    """
    module_name, cls_name = qual_cls_name.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, cls_name)


def list_models_from_config(cfg):
    """List all combinations of models from the config."""
    models = []
    assert cfg["models"] is not None, "No models specified in config in config"  # noqa
    for algo, algo_spec in cfg["models"].items():
        _estimator = load_class(algo_spec["estimator"])
        if algo_spec["params"] is not None:
            grid_params = list(ParameterGrid(algo_spec["params"]))  # noqa
            for params in grid_params:
                models.append(_estimator(**params))
        else:
            models.append(_estimator())
    return models


def flatten_list(args):
    """Flatten the list of lists into a single list.

    Parameters
    ----------
    args: list of lists

    Returns
    -------
    new_list: flattened list
    """
    if not isinstance(args, list):
        args = [args]
    new_list = []
    for x in args:
        if isinstance(x, list):
            new_list += flatten_list(list(x))
        # elif isinstance(x, sp.Array):
        #   new_list += flatten_list(list(x))
        else:
            new_list.append(x)
    return new_list


def display_as_tabs(figs, width=300, height=300):
    """To display multiple dataset outputs as tabbed panes.

    Parameters
    ----------
    figs: list(tuples)
        List of ('tab_name',widget) to be displayed
    width: int, optional
        width of the output, default 300
    height: int, optional
        height of the output, default 300

    Returns
    -------
    pn.Tabs()
    """
    tabs = pn.Tabs()

    plts = []
    for name, wdgt in figs:
        if isinstance(wdgt, pd.DataFrame):
            wdgt.columns = map(str, wdgt.columns)
            cols = wdgt.select_dtypes("object").columns.tolist()
            wdgt = wdgt.transform_columns(cols, str)
            wdgt = pn.widgets.DataFrame(wdgt, name=name, width=width, height=height)
        plts.append((name, wdgt))

    tabs.extend(plts)
    return tabs


def get_fs_and_abs_path(path, storage_options=None):
    """Get the Filesystem and paths from a urlpath and options.

    Parameters
    ----------
    path : string or iterable
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``), or globstring pointing to data.
    storage_options : dict, optional
        Additional keywords to pass to the filesystem class.

    Returns
    -------
    fsspec.FileSystem
       Filesystem Object
    list(str)
        List of paths in the input path.
    """
    fs, _, paths = fsspec.core.get_fs_token_paths(path, storage_options=storage_options)
    if len(paths) == 1:
        return fs, paths[0]
    else:
        return fs, paths


def get_package_version():
    """Return the version of the package."""
    return tigerml.__version__


def initialize_random_seed(seed):
    """Initialise random seed using the input ``seed``.

    Parameters
    ----------
    seed : int

    Returns
    -------
    int
        seed integer
    """
    logger.info(f"Initialized Random Seed : {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    return seed


def load_yml(path, *, fs=None, **kwargs):
    """Load a yml file from the input `path`.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    dict
        dictionery of the loaded yml file
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, mode="r") as fp:
        return yaml.safe_load(fp, **kwargs)


def get_data_dir_path():
    """Fetch the data directory path."""
    return op.join(get_package_path(), "..", "data")


def get_package_path():
    """Get the path of the current installed tigerml package.

    Returns
    -------
    str
        path string in the current system where the tigerml package is loaded from
    """
    path = list(tigerml.__path__)
    return op.dirname(op.abspath(path[0]))


def import_python_file(py_file_path):
    mod_name, ext = op.splitext(op.basename(op.abspath(py_file_path)))
    if ext != ".py":
        raise ValueError("Invalid file extension : {ext}. Expected a py file")
    spec = importlib.util.spec_from_file_location(mod_name, py_file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_model(model, loc):
    """Save  in a location.

    Parameters
    ----------
    model : A trained pyspark.ml model
        Model predictions to be saved
    loc : str
        Path string of the location where the model has to be saved
    """
    model.write().overwrite().save(loc)
