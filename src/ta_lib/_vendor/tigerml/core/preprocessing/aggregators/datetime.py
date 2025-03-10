import inspect
import math
import pandas as pd
from datetime import date, datetime, time, timedelta, timezone

from .external import datetime_aggregators as aggregators

for agg in aggregators:
    if inspect.isclass(agg):
        exec(f"from {agg.__module__} import {agg.__name__}")
    else:
        exec(f"{agg.__class__.name} = agg")

# for agg_class in agg_classes:
# 	exec('from tsfresh_primitives import ' + agg_class.__name__)


class Lag:
    """Lag class."""

    pass


class DateToHoliday:
    """Date to holiday class."""

    pass


class DistanceFromHoliday:
    """Distance from holiday class."""

    pass


def TO_STRING(value):
    if type(value).__name__ == "Series":
        return value.astype(str)
    return str(value)


def ADDYEARS(datetime, no_of_years):
    if type(datetime).__name__ == "Series":
        if type(no_of_years).__name__ == "Series":
            return pd.concat([datetime, no_of_years], axis=1).apply(
                lambda s: ADDYEARS(s[0], s[1]), axis=1
            )
        else:
            return datetime.apply(lambda s: ADDYEARS(s, no_of_years))
    elif type(no_of_years).__name__ == "Series":
        return datetime.apply(lambda s: ADDYEARS(datetime, s))
    else:
        return datetime.replace(year=datetime.year + int(no_of_years))


def ADDMONTHS(datetime, no_of_months):
    if type(datetime).__name__ == "Series":
        if type(no_of_months).__name__ == "Series":
            return pd.concat([datetime, no_of_months], axis=1).apply(
                lambda s: ADDMONTHS(s[0], s[1]), axis=1
            )
        else:
            return datetime.apply(lambda s: ADDMONTHS(s, no_of_months))
    elif type(no_of_months).__name__ == "Series":
        return datetime.apply(lambda s: ADDMONTHS(datetime, s))
    else:
        no_of_months = int(no_of_months)
        new_month = datetime.month + no_of_months
        new_year = datetime.year
        if new_month > 12 or new_month <= 0:
            new_year = new_year + math.floor(new_month / 12)
            if new_month % 12 == 0:
                new_year -= 1
                new_month = 12
            else:
                new_month = new_month % 12
        try:
            return datetime.replace(year=new_year, month=new_month)
        except ValueError:
            return None


def ADDDAYS(datetime, no_of_days):
    if type(datetime).__name__ == "Series":
        if type(no_of_days).__name__ == "Series":
            return pd.concat([datetime, no_of_days], axis=1).apply(
                lambda s: ADDDAYS(s[0], s[1]), axis=1
            )
        else:
            return datetime.apply(lambda s: ADDDAYS(s, no_of_days))
    elif type(no_of_days).__name__ == "Series":
        return datetime.apply(lambda s: ADDDAYS(datetime, s))
    else:
        return datetime + timedelta(days=int(no_of_days))


def ADDHOURS(datetime, no_of_hours):
    if type(datetime).__name__ == "Series":
        if type(no_of_hours).__name__ == "Series":
            return pd.concat([datetime, no_of_hours], axis=1).apply(
                lambda s: ADDHOURS(s[0], s[1]), axis=1
            )
        else:
            return datetime.apply(lambda s: ADDHOURS(s, no_of_hours))
    elif type(no_of_hours).__name__ == "Series":
        return datetime.apply(lambda s: ADDHOURS(datetime, s))
    else:
        return datetime + timedelta(seconds=int(no_of_hours) * 3600)


def ADDMINUTES(datetime, no_of_minutes):
    if type(datetime).__name__ == "Series":
        if type(no_of_minutes).__name__ == "Series":
            return pd.concat([datetime, no_of_minutes], axis=1).apply(
                lambda s: ADDMINUTES(s[0], s[1]), axis=1
            )
        else:
            return datetime.apply(lambda s: ADDMINUTES(s, no_of_minutes))
    elif type(no_of_minutes).__name__ == "Series":
        return datetime.apply(lambda s: ADDMINUTES(datetime, s))
    else:
        return datetime + timedelta(seconds=int(no_of_minutes) * 60)


def ADDSECONDS(datetime, no_of_seconds):
    if type(datetime).__name__ == "Series":
        if type(no_of_seconds).__name__ == "Series":
            return pd.concat([datetime, no_of_seconds], axis=1).apply(
                lambda s: ADDSECONDS(s[0], s[1]), axis=1
            )
        else:
            return datetime.apply(lambda s: ADDSECONDS(s, no_of_seconds))
    elif type(no_of_seconds).__name__ == "Series":
        return datetime.apply(lambda s: ADDSECONDS(datetime, s))
    else:
        return datetime + timedelta(seconds=int(no_of_seconds))


def DATE(year, month=1, day=1):
    if type(year).__name__ == "Series":
        if len(year) == 0:
            return None
        if type(year.iloc[0]).__name__ in ["Timestamp", "datetime"]:
            return year.dt.date
    if "Series" in [type(year).__name__, type(month).__name__, type(day).__name__]:
        length = len([x for x in [year, month, day] if type(x).__name__ == "Series"][0])
        year = (
            year.astype(int)
            if type(year).__name__ == "Series"
            else pd.Series([int(year)] * length)
        )
        month = (
            month.astype(int)
            if type(month).__name__ == "Series"
            else pd.Series([int(month)] * length)
        )
        day = (
            day.astype(int)
            if type(day).__name__ == "Series"
            else pd.Series([int(day)] * length)
        )
        return pd.to_datetime(
            pd.concat(
                [year.rename("year"), month.rename("month"), day.rename("day")], axis=1
            )
        ).dt.date
    if type(year).__name__ in ["Timestamp", "datetime"]:
        return year.date()
    else:
        if not month or not day:
            raise ValueError("Expected 3 arguments")
    try:
        return date(year=int(year), month=int(month), day=int(day))
    except ValueError:
        return None


def TIME(hours, minutes=0, seconds=0):
    if type(hours).__name__ == "Series":
        if len(hours) == 0:
            return None
        if type(hours.iloc[0]).__name__ in ["Timestamp", "datetime"]:
            return hours.dt.time
    if "Series" in [
        type(hours).__name__,
        type(minutes).__name__,
        type(seconds).__name__,
    ]:
        length = len(
            [x for x in [hours, minutes, seconds] if type(x).__name__ == "Series"][0]
        )
        year = pd.Series([int(2000)] * length)
        month = pd.Series([int(1)] * length)
        day = pd.Series([int(1)] * length)
        hours = (
            hours.astype(int)
            if type(hours).__name__ == "Series"
            else pd.Series([int(hours)] * length)
        )
        minutes = (
            minutes.astype(int)
            if type(minutes).__name__ == "Series"
            else pd.Series([int(minutes)] * length)
        )
        seconds = (
            seconds.astype(int)
            if type(seconds).__name__ == "Series"
            else pd.Series([int(seconds)] * length)
        )
        return pd.to_datetime(
            pd.concat(
                [
                    year.rename("year"),
                    month.rename("month"),
                    day.rename("day"),
                    hours.rename("hours"),
                    minutes.rename("minutes"),
                    seconds.rename("seconds"),
                ],
                axis=1,
            )
        ).dt.time
    if type(hours).__name__ in ["Timestamp", "datetime"]:
        return hours.time()
    try:
        return time(hour=int(hours), minute=int(minutes), second=int(seconds))
    except ValueError:
        return None


def explicit_checker(f):
    varnames = f.__code__.co_varnames

    def wrapper(*a, **kw):
        kw["explicit_params"] = set(list(varnames[: len(a)]) + list(kw.keys()))
        return f(*a, **kw)

    return wrapper


@explicit_checker
def DATETIME(year, month=1, day=1, hours=0, minutes=0, seconds=0, explicit_params=None):
    if type(year).__name__ == "Series":
        if len(year) == 0:
            return None
        if type(year.iloc[0]).__name__ == "date":
            if type(month).__name__ == "Series":
                if len(month) == 0:
                    return None
                if type(month.iloc[0]).__name__ == "time":
                    return pd.to_datetime(year.astype(str) + " " + month.astype(str))
                else:
                    hours = month if month in explicit_params else 0
                    minutes = day if day in explicit_params else 0
                    seconds = hours
                    return pd.to_datetime(
                        year.astype(str)
                        + " "
                        + TO_STRING(TIME(hours, minutes, seconds))
                    )
    if "Series" in [
        type(year).__name__,
        type(month).__name__,
        type(day).__name__,
        type(hours).__name__,
        type(minutes).__name__,
        type(seconds).__name__,
    ]:
        return pd.to_datetime(
            TO_STRING(DATE(year, month, day))
            + " "
            + TO_STRING(TIME(hours, minutes, seconds))
        )
    if type(year).__name__ == "date":
        if type(month).__name__ == "time":
            return datetime.combine(year, month)
        else:
            hours = month if month in explicit_params else 0
            minutes = day if day in explicit_params else 0
            seconds = hours
            return datetime.combine(date, TIME(hours, minutes, seconds))
    try:
        return datetime(
            year=int(year),
            month=int(month),
            day=int(day),
            hour=int(hours),
            minute=int(minutes),
            second=int(seconds),
        )
    except ValueError:
        return None


def NOW():
    return timezone.now()
