import hashlib
import joblib
import logging
import os
import os.path as op
import tempfile
from datetime import datetime
from functools import wraps
from io import BytesIO


def dict_depth(input_dict):
    assert isinstance(input_dict, dict), "input_dict has to be dict"
    max_depth = max(
        [
            dict_depth(value) if isinstance(value, dict) else 0
            for value in input_dict.values()
        ]
    )
    return max_depth + 1


def flatten_list(args):
    if not isinstance(args, list):
        args = [args]
    new_list = []
    for x in args:
        if isinstance(x, list):
            new_list += flatten_list(list(x))
        else:
            new_list.append(x)
    return new_list


def cartesian_product(*args, unique=True):
    import itertools

    if unique:
        combinations = sorted(
            list(
                set(
                    [
                        tuple(set(x))
                        for x in itertools.product(*args)
                        if len(x) == len(args)
                    ]
                )
            )
        )
    else:
        combinations = itertools.product(*args)
    return combinations


def normalized(df):
    """Returns the dataframe with all variables normalized.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df : pd.DataFrame
        updated with all columns of the dataframe normalized.
    """
    return df.apply(lambda x: (x - x.mean()) / x.std())


def get_x_y_vars(data_columns, x_vars=None, y_vars=None):
    """Returns the x_vars, y_vars and required columns which are union of x_vars and y_vars.

    Parameters
    ----------
    data_columns : `list`
        columns of a dataframe which will help us get `x_vars` and `y_vars`
    x_vars : `list` or a `string`
        default : empty list, a list of variables for which we want
        to plot bivariate plots
    y_vars : list or string
        default : empty list, a list of variables for which we want
        to plot bivariate plots

    Returns
    -------
    x_vars, y_vars : after checking that it exists in the data_columns
    and set(x_vars) and set(y_vars) are subsets in data_columns
    req_cols : `list`
        union of set of x_vars and y_vars
    """
    if not x_vars:
        x_vars = data_columns
    elif isinstance(x_vars, str):
        x_vars = [x_vars]
    if not y_vars:
        y_vars = data_columns
    elif isinstance(y_vars, str):
        y_vars = [y_vars]
    if len(set(x_vars).intersection(set(data_columns))) == 0:
        raise ValueError(
            "Variables not found in dataset: ",
            set(x_vars) - set(data_columns),
        )
    elif len(set(y_vars).intersection(set(data_columns))) == 0:
        raise ValueError(
            "Variables not found in dataset: ",
            set(y_vars) - set(data_columns),
        )
    elif len(set(x_vars).union(y_vars) - set(data_columns)) > 0:
        print(
            "Following variables not found in dataset,",
            "bivariate_plots will be generated without them: ",
            set(x_vars).union(y_vars) - set(data_columns),
        )
        x_vars = list(set(x_vars).intersection(set(data_columns)))
        y_vars = list(set(y_vars).intersection(set(data_columns)))
    req_cols = list(set(x_vars).union(set(y_vars)))
    return x_vars, y_vars, req_cols


def import_from_module_path(name):
    m = __import__(name)
    for n in name.split(".")[1:]:
        m = getattr(m, n)
    return m


class Wrapper(object):
    """
    DataSet wrapper class.

    An object wrapper is initialized with the object it wraps then proxies
    any unhandled getattribute methods to it. If no attribute is found either
    on the wrapper or the wrapped object, an AttributeError is raised.

    .. seealso:: http://code.activestate.com/recipes/577555-object-wrapper-class/

    Parameters
    ----------
    obj : object
        The object to wrap with the wrapper class
    """

    def __init__(self, obj):
        self._wrapped = obj

    def __getattr__(self, attr):
        # proxy to the wrapped object
        return getattr(self._wrapped, attr)


class WrapperClone(object):
    """Wrapper clone class."""

    def __init__(self, obj):
        for att in [a for a in dir(obj) if a not in dir(self)]:
            try:
                setattr(self, att, getattr(obj, att))
            except Exception:
                pass


def time_now_readable():
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def prettify_slug(slug):
    if slug is not None:
        slug = slug.replace("_", " ")
        import string

        return string.capwords(slug)
    else:
        return ""


def slugify(text, separator="_"):
    if not text:
        return ""
    from slugify import slugify

    return slugify(text, separator=separator)


def fail_gracefully(logger=None):
    def fail_gracefully_with_log(func):
        # allow_exceptions = os.environ.get('TA_ALLOW_EXCEPTIONS', 'True')
        # if allow_exceptions.upper() == "TRUE":
        #     return func
        @wraps(func)
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    "Cannot compute {}. Error - {}".format(func.__name__, e)
                )
                return "Cannot compute {}. Error - {}".format(func.__name__, e)

        return inner

    return fail_gracefully_with_log


def measure_time(logger=None):
    def measure_time_with_log(func):
        @wraps(func)
        def inner(*args, **kwargs):
            logger.info("Started running {}".format(func.__name__))
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            logger.info("Finished running {}".format(func.__name__))
            logger.info(
                "Time taken to run {} is {} seconds".format(
                    func.__name__, (end_time - start_time).seconds
                )
            )
            return result

        return inner

    return measure_time_with_log


def create_safe_filename(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace("*", "_")
    )


class DictObject:
    """Dict object class."""

    def __init__(self, dict):
        self.dict = dict

    def __getattr__(self, item):
        if item in self.dict.keys():
            return self.dict[item]
        return eval(f"self.dict.{item}")

    def __getitem__(self, item):
        return self.dict[item]

    def __iter__(self):
        return iter(self.dict)

    def __len__(self):
        return len(self.dict)


def params_to_dict(**kwargs):
    return kwargs


def hash_object(obj, expensive=False, block_size=4096):
    """Return a content based hash for the input `obj`.

    The returned hash value can be used to verify equality of two objects.
    If the hash values of two objects are equal, then they are identical and
    one can be replaced with another.

    Parameters
    ----------
    obj: Object

    Returns
    -------
    string
    """
    hasher = hashlib.sha256()

    if expensive:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fname = op.join(tmp_dir, "tmp.joblib")
            joblib.dump(obj, tmp_fname)
            with open(tmp_fname, "rb") as fp:
                while True:
                    data = fp.read(block_size)
                    if len(data) <= 0:
                        break
                    hasher.update(data)
    else:
        fp = BytesIO()
        joblib.dump(obj, fp)
        data = fp.getvalue()
        hasher.update(data)

    return hasher.hexdigest()


def set_logger_config(verbose=0, level_name=True, log_time=False):
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    # To avoid printing even tracebacks when verbose=0, use the following
    # levels = [logging.CRITICAL, logging.ERROR,
    #           logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, verbose)]
    logging_formats = {
        True: {
            True: "\n%(asctime)s - %(levelname)s: \n%(message)s",
            False: "\n%(levelname)s - \n%(message)s",
        },
        False: {True: "\n%(asctime)s: \n%(message)s", False: "\n%(message)s"},
    }

    # dynamically resetting the logging level for root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(logging.StreamHandler())
    root_logger.setLevel(level=level)
    logging.captureWarnings(True)

    # setting the logging level and format for default handler in root logger
    root_logger.handlers[0].setLevel(level=level)
    root_logger.handlers[0].setFormatter(
        logging.Formatter(
            fmt=logging_formats[level_name][log_time], datefmt="%d-%m-%Y %H:%M:%S"
        )
    )

    # setting the logging level for bokeh logger
    from bokeh.util.logconfig import bokeh_logger

    bokeh_logger.setLevel(level=level)

    # setting the format for default handler in bokeh logger
    root_logger.handlers[1].handlers[0].setFormatter(
        logging.Formatter(
            fmt=logging_formats[level_name][log_time], datefmt="%d-%m-%Y %H:%M:%S"
        )
    )

    bokeh_logger.handlers[0].setFormatter(
        logging.Formatter(
            fmt=logging_formats[level_name][log_time], datefmt="%d-%m-%Y %H:%M:%S"
        )
    )

    # setting the logging level for param logger
    from param import parameterized

    param_logger = parameterized.get_logger()
    param_logger.setLevel(level=level)

    # setting the format for default handler in param logger
    param_logger.handlers[0].setFormatter(
        logging.Formatter(
            fmt=logging_formats[level_name][log_time], datefmt="%d-%m-%Y %H:%M:%S"
        )
    )


def set_logger(logger_name, verbose=2, log_dir=None):
    # Todo: enhancement for logging control in terms
    #  of decision on showing what/when to console,
    #  by default log level for StreamHandler
    #  will be used as 'WARNING' (verbose=2).
    levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ]
    verbose = 0 if verbose < 0 else verbose
    log_level = levels[min(len(levels) - 1, verbose)]
    log_format = {
        "default": "%(asctime)s - %(levelname)s "
        "- %(name)s::%(funcName)s::"
        "%(lineno)d - %(message)s",
        "simple": "%(levelname)s - %(message)s",
    }

    # define root logger
    root_logger = logging.getLogger(logger_name)
    root_logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter(log_format["default"], datefmt="%Y-%m-%d %H:%M:%S")
    )
    root_logger.addHandler(console_handler)
    if log_dir is not None:
        # Create file handlers
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        # info logs file handler
        info_file_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir + "/info.log", when="midnight"
        )
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(
            logging.Formatter(log_format["default"], datefmt="%d-%m-%Y %H:%M:%S")
        )
        root_logger.addHandler(info_file_handler)
        # error logs file handler
        error_file_handler = logging.handlers.TimedRotatingFileHandler(
            log_dir + "/error.log", when="midnight"
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(
            logging.Formatter(log_format["default"], datefmt="%d-%m-%Y %H:%M:%S")
        )
        root_logger.addHandler(error_file_handler)
    root_logger.propagate = False
    return root_logger


def set_logger_path(path):
    """
    Set the path for creating info and error log files.

    This function allows you to specify the directory where both info and error log files will be created.

    Parameters
    ----------
    path : str
        The absolute or relative directory path where log files will be generated.

    Returns
    -------
    None

    Examples
    --------
    >>> from tigerml.core.utils import set_logger_path
    >>> set_logger_path("/path/to/log_directory/")
    """
    os.environ["logger_path"] = path
