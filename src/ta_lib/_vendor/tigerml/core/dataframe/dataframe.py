import logging
from pandas.api.types import CategoricalDtype
from tigerml.core.utils import compute_if_dask, measure_time

from .base import *

_LOGGER = logging.getLogger(__name__)


def read_csv(path, backend="pandas", **kwargs):
    if backend == "pandas":
        try:
            df = pd.read_csv(path, **kwargs)
        except MemoryError as e:
            raise Exception(
                "Dataset too big for memory. "
                'Try using backend="dask" or "vaex". {}'.format(e)
            )
    elif backend == "dask":
        import dask.dataframe as dd

        df = dd.read_csv(path, **kwargs)
    elif backend == "vaex":
        import vaex

        df = vaex.open(path, **kwargs)
    else:
        raise Exception("backend must be one of pandas, dask or vaex.")
    return DataFrame(df)


def read_parquet(path, backend="pandas", **kwargs):
    if backend == "pandas":
        try:
            df = pd.read_parquet(path, **kwargs)
        except MemoryError as e:
            raise Exception(
                "Dataset too big for memory. "
                'Try using backend="dask" or "vaex". {}'.format(e)
            )
    elif backend == "dask":
        import dask.dataframe as dd

        df = dd.read_parquet(path, **kwargs)
    elif backend == "vaex":
        import vaex

        df = vaex.open(path, **kwargs)
    else:
        raise Exception("backend must be one of pandas, dask or vaex.")
    return DataFrame(df)


def read_excel(path, backend="pandas", **kwargs):
    if backend == "pandas":
        try:
            df = pd.read_excel(path, **kwargs)
        except MemoryError as e:
            raise Exception(
                "Dataset too big for memory. "
                'Try using backend="dask" or "vaex". {}'.format(e)
            )
    elif backend == "dask":
        import dask.dataframe as dd

        df = dd.read_csv(path, **kwargs)
    elif backend == "vaex":
        import vaex

        df = vaex.open(path, **kwargs)
    else:
        raise Exception("backend must be one of pandas, dask or vaex.")
    return DataFrame(df)


def concat(objs, **kwargs):
    if is_dask(objs[0]):
        import dask.dataframe as dd

        result = dd.concat([detigerify(obj) for obj in objs], **kwargs)
    else:
        result = pd.concat([detigerify(obj) for obj in objs], copy=False, **kwargs)
    return convert_to_tiger_assets(result)


def get_dummies(data, **kwargs):
    data = tigerify(data)
    if data.backend == BACKENDS.dask:
        if "dummy_na" in kwargs and data.isna().sum().compute() == 0:
            kwargs.pop("dummy_na")
        import dask.dataframe as dd

        result = dd.get_dummies(data._data, **kwargs)
    else:
        result = pd.get_dummies(data._data, **kwargs)
    return convert_to_tiger_assets(result)


def merge(left, right, *args, **kwargs):
    if left.backend == BACKENDS.dask:
        import dask.dataframe as dd

        result = dd.merge(left._data, right._data, *args, **kwargs)
    else:
        right = compute_if_dask(right)
        result = pd.merge(left._data, right._data, *args, **kwargs)
    return convert_to_tiger_assets(result)


def to_datetime(arg, *args, **kwargs):
    module = get_module(arg)
    if module != "tigerml":
        arg = tigerify(arg)
    if arg.backend == BACKENDS.dask:
        import dask.dataframe as dd

        result = dd.to_datetime(arg._data, *args, **kwargs)
    else:
        result = pd.to_datetime(arg._data, *args, **kwargs)
    return convert_to_tiger_assets(result)


class DataFrame(BackendMixin):
    """Dataframe class."""

    def __init__(self, data=None, backend=BACKENDS.pandas, **kwargs):
        if data is not None:
            if get_module(data) == "tigerml":
                data = data._data
            elif get_module(data) == "pandas":
                data = pd.DataFrame(data, **kwargs)
            if is_series(data):
                data = convert_series_to_df(data, **kwargs)
            module = get_module(data)
            if module not in BACKENDS.keys():
                raise Exception("tigerml currently supports only pandas, dask and vaex")
        else:
            assert backend in BACKENDS
            module = backend
            if module == BACKENDS.pandas:
                data = pd.DataFrame(**kwargs)
            elif module == BACKENDS.dask:
                data = pd.DataFrame(**kwargs)
                import dask

                dask.config.set(scheduler="processes")
                import dask.dataframe as dd

                data = dd.from_pandas(data, chunksize=10000)
            elif module == BACKENDS.vaex:
                import vaex

                data = vaex.DataFrame(**kwargs)
        self._data = data
        # if module == BACKENDS.dask:
        #     # from dask.distributed import Client
        #     # self.client = Client()
        #     self._data = dask.persist(self._data)[0]

    @property
    def numeric_columns(self):
        """Returns numeric columns."""
        from tigerml.core.utils import get_num_cols

        return get_num_cols(self._data)

    @property
    def cat_columns(self):
        """Returns categorical columns."""
        from tigerml.core.utils import get_cat_cols

        return get_cat_cols(self._data)

    def apply(self, func, axis=0, **kwargs):
        """Applies provided function on data."""
        if self.backend == BACKENDS.dask:
            if axis == 0:
                series = pd.Series(
                    [func(self._data[col]) for col in self._data.columns],
                    index=list(self._data.columns),
                )
                return Series(series)
        return tigerify(self._data.apply(func, axis=axis, **kwargs))

    def describe(self, *args, **kwargs):
        """Returns data description."""
        if self.backend == BACKENDS.dask:
            return self._data.describe(
                datetime_is_numeric=True, *args, **kwargs
            ).compute()
        return self._data.describe(datetime_is_numeric=True, *args, **kwargs)

    def corr(self):
        """Returns correlation matrix."""
        if self.backend == BACKENDS.dask:
            import dask

            return dask.compute(self._data.corr())[0]
        return self._data.corr()

    def categorize(self, set_order=False):
        """Sets the categorical columns."""
        if self.backend == BACKENDS.dask:
            self._data = self._data.categorize()
        elif self.backend == BACKENDS.pandas:
            from tigerml.core.utils import get_cat_cols

            cat_cols = get_cat_cols(self)
            for col in cat_cols:
                if set_order:
                    cat_type = CategoricalDtype(
                        categories=self._data[col].dropna().unique().tolist(),
                        ordered=True,
                    )
                    self._data[col] = self._data[col].astype(cat_type, copy=False)
                else:
                    self._data[col] = self._data[col].astype("category", copy=False)
        return self

    def order_categories(self):
        """Orders CategoricalDtype columns."""
        if self.backend == BACKENDS.dask:
            cat_cols = self._data.dtypes[
                self._data.dtypes.astype(str) == "category"
            ].index.values
            if len(cat_cols) > 0:
                for col in cat_cols:
                    self._data[col] = self._data[col].cat.as_ordered()
        return self

    @measure_time(_LOGGER)
    def convert_datetimes(self, format_dict=None, **pandas_kwargs):
        """Converts datetime columns."""
        if not format_dict:
            _LOGGER.info(
                "TigerML Information: It is recommended to pass a dict, "
                "with column names as key and their format in the .csv file"
                "\n\t\t(Note: The format displayed in .xlsx file"
                " may not be same as that present in .csv) as values,"
                "\n\t\tto format_dict argument to perform correct conversion. "
                "\n\t\tEg: df = df.convert_datetimes(format_dict="
                "{'Col_1':'%d-%m-%Y', Col_2':'%I:%M:%S %p'})"
            )
            selected_cols = []
            converted_cols = []
            from dateutil.parser import parse
            from pandas.core.tools.datetimes import (
                _guess_datetime_format_for_array,
            )

            def possible_datetime_value(timestr):
                if type(timestr) == str:
                    try:
                        float(timestr)
                        return 0, None
                    except ValueError as float_conversion_error:
                        if "could not convert" in str(float_conversion_error):
                            try:
                                parse(timestr)
                                return 1, _guess_datetime_format_for_array([timestr])
                            # except (TypeError, ValueError) as parsing_error:
                            except ValueError as parsing_error:
                                """
                                Handling only the known ValueError
                                parse(3) -> TypeError: Parser must be
                                a string or character stream, not int
                                parse(3.0) -> TypeError: Parser must be
                                a string or character stream, not float
                                parse('text') -> ValueError:
                                ('Unknown string format:', 'text')
                                parse('0') -> ValueError:
                                day is out of range for month
                                parse('32:0') -> ValueError: hour must be in 0..23
                                """
                                know_exceptions = [
                                    "Unknown string format",
                                    "out of range",
                                    "does not contain a date",
                                    "must be in",
                                ]
                                if any(
                                    subtext in str(parsing_error)
                                    for subtext in know_exceptions
                                ):
                                    return 0, None
                                else:
                                    _LOGGER.debug(
                                        f"printing timestr before "
                                        f"raising parsing_error... {timestr}"
                                    )
                                    _LOGGER.warning(f"{parsing_error}")
                                    # giving a warning message instead of raising an exception
                                    return 0, None
                        else:
                            _LOGGER.debug(
                                f"printing timestr before raising"
                                f" float_conversion_error... {timestr}"
                            )
                            raise float_conversion_error
                else:
                    return 0, None

            for col in self.dtypes[self.dtypes.astype(str) == "object"].index:
                col_data = self._data[col]
                col_data_df = pd.DataFrame(self._data[col].unique(), columns=[col])
                convert_val, format_val = np.vectorize(possible_datetime_value)(
                    col_data_df[col]
                )
                format_df = pd.DataFrame(
                    np.array([convert_val, format_val]).T,
                    columns=["convert_col", "format_col"],
                )
                col_data_df = col_data_df.join(format_df)
                col_data_df["convert_col"] = col_data_df["convert_col"].astype(float)
                if col_data_df["convert_col"].sum() > 0.3 * len(col_data_df):
                    selected_cols += [col]
                    col_data_df.dropna(inplace=True)
                    if not col_data_df.empty:
                        inferred_format_list = col_data_df["format_col"].unique()
                        for format_ in inferred_format_list:
                            try:
                                to_datetime(
                                    col_data_df[col], format=format_, errors="raise"
                                )
                            except ValueError as e:
                                if "(match)" in str(e):
                                    inferred_format_list = np.delete(
                                        inferred_format_list,
                                        np.where(inferred_format_list == format_),
                                    )
                                else:
                                    raise e
                        if len(inferred_format_list) > 1:
                            _LOGGER.info(
                                f"Multiple datetime formats {inferred_format_list} "
                                f"were detected for column {col}."
                            )
                            _LOGGER.info(f"\nTo convert {col} to datetime format run..")
                            _LOGGER.info(
                                "'df = df.convert_datetimes(format_dict"
                                "={'" + col + "' : <correct_format>})"
                            )
                        elif len(inferred_format_list) == 1:
                            _LOGGER.info(
                                f"Column '{col}' is formatted with "
                                f"format '{inferred_format_list[0]}'."
                            )
                            self._data[col] = to_datetime(
                                col_data,
                                errors="coerce",
                                format=inferred_format_list[0],
                            )._data
                            converted_cols += [col]
            if len(selected_cols) == 0:
                _LOGGER.info(
                    "None of the columns was converted to have "
                    "datetime format by TigerML"
                )
                _LOGGER.info(
                    "To convert any specific column(s) to datetime format run "
                    "'df = df.convert_datetimes(format_dict={<col_1>: "
                    "<correct_format>, <col_2>: <correct_format>})'"
                )
            elif len(selected_cols) > 0 and len(converted_cols) == 0:
                _LOGGER.info(
                    f"Column(s) {selected_cols} was found to have some "
                    f"datetime values, but not converted to datetime format "
                    f"by TigerML due to some inconsistencies in format."
                )
                _LOGGER.info(
                    "To convert any specific column(s) to datetime format run "
                    "'df = df.convert_datetimes(format_dict={<col_1>: "
                    "<correct_format>, <col_2>: <correct_format>})'"
                )
            elif 0 < len(selected_cols) == len(converted_cols):
                _LOGGER.info(
                    f"Column(s) {selected_cols} was found to have some datetime "
                    f"values, and was converted to datetime format by TigerML"
                )
            else:
                non_converted_cols = [
                    item for item in selected_cols if item not in converted_cols
                ]
                _LOGGER.info(
                    f"Column(s) {selected_cols} were found to have some "
                    f"datetime values, but only column(s) {converted_cols} "
                    f"was converted to datetime format by TigerML due to some "
                    f"format inconsistencies in column(s) {non_converted_cols}."
                )
                _LOGGER.info(
                    "To convert any specific column(s) to datetime format run "
                    "'df = df.convert_datetimes(format_dict="
                    "{<col_1>: <correct_format>, <col_2>: <correct_format>})'"
                )
        else:
            for col in format_dict.keys():
                col_data = self._data[col]
                self._data[col] = to_datetime(
                    col_data, errors="coerce", format=format_dict[col], **pandas_kwargs
                )._data
        return self

    def merge(self, right, **kwargs):
        """Merges dataframes."""
        if self.backend == BACKENDS.pandas:
            right = compute_if_dask(right)
        if get_module(right) == "tigerml":
            right = right._data
        return tigerify(self._data.merge(right, **kwargs))

    def groupby(self, *args, **kwargs):
        """Performs groupby operation."""
        # if self.backend == BACKENDS.dask:
        return TigerWrapper(self._data.groupby(*args, **kwargs))


class Series(BackendMixin):
    """Series class."""

    def __init__(self, data=None, backend=BACKENDS.pandas, **kwargs):
        if data is not None:
            if isinstance(data, list):
                data = pd.Series(data, **kwargs)
            assert is_series(data), "passed data is not series"
            module = get_module(data)
            if module not in BACKENDS.keys():
                raise Exception("tigerml currently supports only pandas, dask and vaex")
        else:
            assert backend in BACKENDS
            module = backend
            if module == BACKENDS.pandas:
                data = pd.Series()
            elif module == BACKENDS.dask:
                data = pd.Series()
                import dask.dataframe as dd

                data = dd.from_pandas(data)
            elif module == BACKENDS.vaex:
                import vaex

                data = vaex.Series()
        self._data = data
        if module == BACKENDS.dask:
            # from dask.distributed import Client
            # self.client = Client()
            import dask

            self._data = dask.persist(self._data)[0]

    def mode(self):
        """Returns mode."""
        if self.backend == BACKENDS.dask:
            return pd.Series(self._data.value_counts().compute().index[0])
        return self._data.mode()

    def median(self):
        """Returns median."""
        if self.backend == BACKENDS.dask:
            return self._data.describe().loc["50%"]
        return self._data.median()

    def corr(self, other, **kwargs):
        """Returns correlation."""
        if other.__module__.startswith("tigerml"):
            other = other._data
        if self.backend == BACKENDS.dask:
            import dask

            return dask.compute(self._data.corr(other, **kwargs))[0]
        return self._data.corr(other, **kwargs)

    def tolist(self):
        """Converts to list."""
        if self.backend == BACKENDS.dask:
            return self._data.compute().tolist()
        return self._data.tolist()


class Scalar:
    """Scalar class."""

    def __init__(self, value):
        import dask.dataframe as dd

        if isinstance(value, dd.core.Scalar):
            self._data = value
        else:
            raise Exception("Wrong input for dask Scalar interface.")

    def __getattr__(self, item):
        return self._data.__getattribute__(item)

    def __eq__(self, other):
        other = compute_if_dask(other)
        return self.compute().__eq__(other)

    def __ne__(self, other):
        other = compute_if_dask(other)
        return self.compute().__ne__(other)

    def __ge__(self, other):
        other = compute_if_dask(other)
        return self.compute().__ge__(other)

    def __gt__(self, other):
        other = compute_if_dask(other)
        return self.compute().__gt__(other)

    def __le__(self, other):
        other = compute_if_dask(other)
        return self.compute().__le__(other)

    def __lt__(self, other):
        other = compute_if_dask(other)
        return self.compute().__lt__(other)

    def __and__(self, other):
        other = compute_if_dask(other)
        return self.compute().__and__(other)

    def __or__(self, other):
        other = compute_if_dask(other)
        return self.compute().__or__(other)

    def __add__(self, other):
        other = compute_if_dask(other)
        return self.compute().__add__(other)

    def __sub__(self, other):
        other = compute_if_dask(other)
        return self.compute().__sub__(other)

    def __mul__(self, other):
        other = compute_if_dask(other)
        return self.compute().__mul__(other)

    def __floordiv__(self, other):
        other = compute_if_dask(other)
        return self.compute().__floordiv__(other)

    def __div__(self, other):
        other = compute_if_dask(other)
        return self.compute().__div__(other)

    def __truediv__(self, other):
        other = compute_if_dask(other)
        return self.compute().__truediv__(other)

    def __mod__(self, other):
        other = compute_if_dask(other)
        return self.compute().__mod__(other)

    def __divmod__(self, other):
        other = compute_if_dask(other)
        return self.compute().__divmod__(other)

    def __pow__(self, other):
        other = compute_if_dask(other)
        return self.compute().__pow__(other)

    def __bool__(self):
        return self.compute().__bool__()
