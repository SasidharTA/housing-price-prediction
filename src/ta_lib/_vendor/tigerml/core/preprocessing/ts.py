import pandas as pd
from tigerml.core.utils import DictObject


def convert_to_date(s: pd.Series(), date_format="%Y-%m-%d"):
    """Converting to date.

    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    unique_dates = s.unique()
    dates = {date: pd.to_datetime(date, format=date_format) for date in unique_dates}
    return s.map(dates)


def get_day_of_month(s: pd.Series()):
    pass


def get_month(s: pd.Series()):
    pass


def get_year(s: pd.Series()):
    pass


def get_weekday(s: pd.Series()):
    pass


def get_hour(s: pd.Series()):
    pass


def get_minute(s: pd.Series()):
    pass


# class TSQuery(Query):
#
#     GROUPBY = DictObject({
#         'DATE': convert_to_date,
#         'DAY': get_day_of_month,
#         'MONTH': get_month,
#         'YEAR': get_year,
#         'WEEKDAY': get_weekday,
#         'HOUR': get_hour,
#         'MINUTE': get_minute
#     })
#
#     def __init__(self, data, ts_column, fetch=None, groupby=None, filter=None):
#         super(TSQuery, self).__init__(data, fetch, groupby, filter)
#         self.ts_column = ts_column
#
#     def set_time_window(self, start=None, end=None):
#         self.filter((self.data[self.ts_column] > start & self.data[self.ts_column] < end))
