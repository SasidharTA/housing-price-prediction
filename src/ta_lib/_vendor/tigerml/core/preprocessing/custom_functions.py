import math
import numpy as np
import pandas as pd
from datetime import date, datetime, time, timedelta

# Django imports
from django.utils import timezone

# Comparison Functions
FUNCTIONS_DICT = {
    "EQUAL": {
        "no_of_params": 2,
        "input_type": [
            "IntergerField",
            "DecimalField",
            "TextField",
            "DateField",
            "DateTimeField",
            "TimeField",
        ],
        "return_type": "BooleanField",
    },
    "NOTEQUAL": {
        "no_of_params": 2,
        "input_type": ["IntergerField", "DecimalField", "TextField"],
        "return_type": "BooleanField",
    },
    "LESSTHAN": {
        "no_of_params": 2,
        "input_type": ["IntergerField", "DecimalField", "TextField"],
        "return_type": "BooleanField",
    },
    "LESSTHANEQUAL": {
        "no_of_params": 2,
        "input_type": ["IntergerField", "DecimalField", "TextField"],
        "return_type": "BooleanField",
    },
    "GREATERTHAN": {
        "no_of_params": 2,
        "input_type": ["IntergerField", "DecimalField", "TextField"],
        "return_type": "BooleanField",
    },
    "GREATERTHANEQUAL": {
        "no_of_params": 2,
        "input_type": ["IntergerField", "DecimalField", "TextField"],
        "return_type": "BooleanField",
    },
    "CONTAINS": {
        "no_of_params": 2,
        "input_type": ["IntergerField", "DecimalField", "TextField"],
        "return_type": "BooleanField",
    },
    "DOESNOTCONTAIN": {
        "no_of_params": 2,
        "input_type": ["IntergerField", "DecimalField", "TextField"],
        "return_type": "BooleanField",
    },
    "HASVALUE": {
        "no_of_params": 2,
        "input_type": ["IntergerField", "DecimalField", "TextField"],
        "return_type": "BooleanField",
    },
    "ISBLANK": {
        "no_of_params": 2,
        "input_type": ["IntergerField", "DecimalField", "TextField"],
        "return_type": "BooleanField",
    },
}


def flatten_list(*args):
    arg_list = []
    for arg in args:
        if type(arg).__name__ in ["list", "tuple"]:
            arg_list += flatten_list(*arg)
        else:
            arg_list.append(arg)
    return arg_list


def OR(*args):
    if type(args[0]).__name__ == "Series":
        return pd.concat(args, axis=1).any(axis=1)
    for arg in args:
        if bool(arg):
            return True
    return False


def AND(*args):
    if type(args[0]).__name__ == "Series":
        return pd.concat(args, axis=1).all(axis=1)
    for arg in args:
        if not bool(arg):
            return False
    return True


def EQUALTO(val1, val2):
    return val1 == val2


def NOTEQUAL(val1, val2):
    return val1 != val2


def LESSTHAN(val1, val2):
    return val1 < val2


def LESSTHANEQUAL(val1, val2):
    return val1 <= val2


def GREATERTHAN(val1, val2):
    return val1 > val2


def GREATERTHANEQUAL(val1, val2):
    return val1 >= val2


def CONTAINS(find_in, find_this):
    if type(find_in).__name__ == "Series":
        if type(find_this).__name__ != "Series":
            find_this = [find_this] * len(find_in)
    else:
        if type(find_this).__name__ == "Series":
            find_in = [find_in] * len(find_this)
        else:
            return (
                str(find_this) if type(find_in).__name__ != "list" else find_this
            ) in (str(find_in) if type(find_in).__name__ != "list" else find_in)
    values = [
        (str(x[0]) if type(x[1]).__name__ != "list" else x[0])
        in (str(x[1]) if type(x[1]).__name__ != "list" else x[1])
        for x in zip(list(find_this), list(find_in))
    ]
    return pd.Series(values)


def DOESNOTCONTAIN(string, substring):
    values = ~CONTAINS(string, substring)
    if values == -1:
        return True
    elif values == -2:
        return False
    else:
        return values


def HASVALUE(value):
    if type(value).__name__ == "Series":
        return value.astype(bool)
    return True if value else False


def ISBLANK(value):
    values = ~HASVALUE(value)
    if values == -1:
        return True
    elif values == -2:
        return False
    else:
        return values


# Arithmetic Functions


def SUM(*args):
    args = flatten_list(*args)
    if len(args) == 1 and type(args[0]).__name__ == "Series":
        if len(args[0]) == 0:
            return None
        if type(args[0].iloc[0]).__name__ != "list":
            return args[0].sum()
    new_args = []
    for arg in args:
        if type(arg).__name__ == "Series" and type(arg.iloc[0]).__name__ == "list":
            arg = arg.apply(sum)
        new_args.append(arg)
    return sum(new_args)


def COUNT(*args):
    args = flatten_list(*args)
    if len(args) == 1 and type(args[0]).__name__ == "Series":
        if len(args[0]) == 0:
            return None
        if type(args[0].iloc[0]).__name__ != "list":
            return len(args[0])
    new_args = []
    for arg in args:
        if type(arg).__name__ == "Series":
            if type(arg.iloc[0]).__name__ == "list":
                arg = arg.str.len()
            else:
                arg = arg.apply(lambda s: 1)
        else:
            arg = len(flatten_list(arg))
        new_args.append(arg)
    return sum(new_args)


def UNIQUECOUNT(*args):
    args = flatten_list(*args)
    if len(args) == 1 and type(args[0]).__name__ == "Series":
        if len(args[0]) == 0:
            return None
        if type(args[0].iloc[0]).__name__ != "list":
            return args[0].nunique()
    merged_args = []
    for arg in [x for x in args if type(x).__name__ == "Series"]:
        if type(arg.iloc[0]).__name__ == "list":
            merged_args = merged_args + arg
        else:
            arg = arg.apply(lambda s: flatten_list(s))
            merged_args = merged_args + arg
    merged_args = merged_args + list(
        set(flatten_list(*[x for x in args if type(x).__name__ != "Series"]))
    )
    if type(merged_args).__name__ == "Series":
        return merged_args.apply(lambda s: len(list(set(s))))
    else:
        return len(list(set(merged_args)))


def PRODUCT(*args):
    # import pdb
    # pdb.set_trace()
    from functools import reduce
    from operator import mul

    args = flatten_list(*args)
    if len(args) == 1 and type(args[0]).__name__ == "Series":
        if len(args[0]) == 0:
            return None
        if type(args[0].iloc[0]).__name__ != "list":
            return pd.DataFrame(args[0]).prod()
    new_args = []
    for arg in args:
        if type(arg).__name__ == "Series" and type(arg.iloc[0]).__name__ == "list":
            arg = arg.apply(lambda s: reduce(mul, s, 1))
        new_args.append(arg)
    return reduce(mul, new_args, 1)


def AVERAGE(*args):
    count = COUNT(*args)
    if type(count).__name__ == "Series":
        count[count == 0] = 1
    else:
        if count == 0:
            count = 1
    return SUM(*args) / count


def MERGE(*args):
    args = flatten_list(*args)
    if len(args) == 1 and type(args[0]).__name__ == "Series":
        if len(args[0]) == 0:
            return None
        if type(args[0].iloc[0]).__name__ != "list":
            return args[0].values.tolist()
    if "Series" in [type(arg).__name__ for arg in args]:
        df = pd.DataFrame()
        for index, arg in enumerate(args):
            df[index] = arg
        df["combined_list"] = df.values.tolist()
        return df["combined_list"].apply(flatten_list)
    else:
        combined_list = []
        for arg in args:
            if type(arg).__name__ == "list":
                combined_list += arg
            else:
                combined_list += [arg]
        return combined_list


def PERCENTILE(*args):
    if len(args) < 2:
        raise Exception("Please provide percentile value")
    args = flatten_list(*args)
    percentile = args[-1]
    if type(percentile).__name__ not in ["int", "float"]:
        raise Exception("Please provide right percentile value (int or float)")
    args = args[:-1]
    if len(args) == 1 and type(args[0]).__name__ == "Series":
        if len(args[0]) == 0:
            return None
        if type(args[0].iloc[0]).__name__ != "list":
            return args[0].quantile(percentile / 100.0)
    merged = MERGE(*args)
    if type(merged).__name__ == "Series":
        return merged.apply(lambda s: np.percentile(s, percentile))
    else:
        return np.percentile(merged, percentile)


def SHIFT(shift, values, sort=None):
    if sort:
        values = values.sort_values(ascending=(sort.lower() in ["ascending", "asc"]))
    return values.shift(periods=int(shift))


def MEDIAN(*args):
    return PERCENTILE(*args, 50)


def MAX(*args):
    return PERCENTILE(*args, 100)


def MIN(*args):
    return PERCENTILE(*args, 0)


def GROUP(df, string_dict, group_by, agg_expression):
    agg_expression = agg_expression[1:-1]
    from forms.field_conditions import evaluate_expression_old

    return df.groupby(by=group_by.name).apply(
        lambda s: evaluate_expression_old(agg_expression, s, string_dict)
    )


def COMPUTEIF(df, string_dict, check_series, function):
    function = function[1:-1]
    from forms.field_conditions import evaluate_expression_old

    new_df = pd.concat([df, check_series.rename("check")], axis=1)
    return evaluate_expression_old(function, new_df[new_df["check"]], string_dict)


def IF(data, string_dict, check, func, else_func):
    from forms.field_conditions import evaluate_expression_old

    if type(check).__name__ == "Series":
        new_df = pd.concat([data, check.rename("check")], axis=1)
        new_df["values"] = None
        new_df.loc[new_df["check"], "values"] = new_df[new_df["check"]].apply(
            lambda x: evaluate_expression_old(func, x, string_dict), axis=1
        )
        new_df.loc[~new_df["check"], "values"] = new_df[~new_df["check"]].apply(
            lambda x: evaluate_expression_old(else_func, x, string_dict), axis=1
        )
        return new_df["values"]
    else:
        if check:
            return evaluate_expression_old(func, data, string_dict)
        else:
            return evaluate_expression_old(else_func, data, string_dict)


def CONCATENATE(*args):
    args = flatten_list(*args)
    if len(args) == 1 and type(args[0]).__name__ == "Series":
        if len(args[0]) == 0:
            return None
        if type(args[0].iloc[0]).__name__ != "list":
            return args[0].str.cat()
    value = ""
    for arg in args:
        if type(arg).__name__ == "Series":
            if type(arg.iloc[0]).__name__ == "list":
                arg = arg.apply(lambda s: "".join([str(x) for x in s]))
            elif type(arg.iloc[0]).__name__ == "str":
                arg = arg.astype(str)
        elif type(arg).__name__ != "str":
            arg = str(arg)
        value = value + arg
    return value


def CEIL(value):
    if type(value).__name__ == "Series":
        return np.ceil(value)
    return math.ceil(float(value))


def FLOOR(value):
    if type(value).__name__ == "Series":
        return np.floor(value)
    return math.floor(float(value))


def LENGTH(string):
    if type(string).__name__ == "Series":
        return string.str.len()
    return len(string)


def ROUND(value):
    if type(value).__name__ == "Series":
        return np.round(value)
    return round(float(value))


def UPPERCASE(string):
    if type(string).__name__ == "Series":
        return string.str.upper()
    return string.upper()


def LOWERCASE(string):
    if type(string).__name__ == "Series":
        return string.str.lower()
    return string.lower()


def POWER(x, y):
    if type(x).__name__ == "Series" or type(y).__name__ == "Series":
        return np.power(x, y)
    return math.pow(float(x), int(y))


# Date and Time functions


def YEAR(datetime):
    if type(datetime).__name__ == "Series":
        return datetime.dt.year
    return datetime.year


def MONTH(datetime):
    if type(datetime).__name__ == "Series":
        return datetime.dt.month
    return datetime.month


def DAY(datetime):
    if type(datetime).__name__ == "Series":
        return datetime.dt.day
    return datetime.day


def WEEKDAY(datetime):
    if type(datetime).__name__ == "Series":
        return datetime.dt.strftime("%A")
    return datetime.strftime("%A")


def HOUR(datetime):
    if type(datetime).__name__ == "Series":
        return datetime.dt.hour
    return datetime.hour


def MINUTE(datetime):
    if type(datetime).__name__ == "Series":
        return datetime.dt.minute
    return datetime.minute


def SECOND(datetime):
    if type(datetime).__name__ == "Series":
        return datetime.dt.second
    return datetime.second


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


def ABSOLUTE(value):
    try:
        return abs(value)
    except TypeError:
        if type(value).__name__ == "list":
            return [abs(val) for val in value]
        else:
            return None


def NEGATIVE(value):
    try:
        return -value
    except TypeError:
        if type(value).__name__ == "list":
            return [-val for val in value]
        else:
            return None


def DIFFERENCE(val1, val2):
    return SUM(val1, NEGATIVE(val2))


def TIMEDIFFERENCE(val1, val2):
    if "Series" in [type(val1).__name__, type(val2).__name__]:
        return (pd.to_datetime(val1) - pd.to_datetime(val2)).astype("timedelta64[s]")
    else:
        return ((val1 - val2).days * (24 * 3600)) + (val1 - val2).seconds


def DIVIDE(val1, val2):
    return PRODUCT(val1, POWER(val2, -1))


def TO_DATETIME(str):
    if type(str).__name__ == "Series":
        return str.apply(lambda s: TO_DATETIME(s))
    return datetime.strptime(str, "%Y-%m-%d %H:%M:%S.%f")


def TO_DATE(str):
    if type(str).__name__ == "Series":
        return str.apply(lambda s: TO_DATE(s))
    return datetime.strptime(str, "%Y-%m-%d")


def TO_TIME(str):
    if type(str).__name__ == "Series":
        return str.apply(lambda s: TO_TIME(s))
    return datetime.strptime(str, "%H:%M:%S.%f")


def TO_BOOL(str):
    true_strs = ("yes", "true", "t", "1", "1.0")
    if type(str).__name__ == "Series":
        return str.isin(true_strs)
    return str.lower() in true_strs


def TO_NUMBER(value):
    if type(value).__name__ == "Series":
        return value.astype(float)
    return float(value)


def TO_STRING(value):
    if type(value).__name__ == "Series":
        return value.astype(str)
    return str(value)


def RANGE(start, increment, end=None, no_of_bins=None):
    data_type = None
    if type(start).__name__ == "date":
        increment = timedelta(days=increment)
    elif type(start).__name__ in ["datetime", "datetime64"]:
        start = start.astype(datetime)
        increment = timedelta(hours=increment)
        if end:
            if type(end).__name__ == "datetime64":
                end = end.astype(datetime)
    elif type(start).__name__ == "time":
        data_type = "time"
        start = start.hour * 60 + start.minute + (start.second / 60)
        if end:
            end = end.hour * 60 + end.minute + (end.second / 60)
    if no_of_bins:
        if end:
            calc_end = start + (increment * no_of_bins)
            if calc_end < end:
                end = calc_end
        else:
            end = start + (increment * no_of_bins)
    else:
        if not end:
            raise Exception("End nor no of bins are given")
    output = list(np.arange(start, end + increment, increment))
    if data_type:
        output = [x for x in output if x < 1440]
        output = [
            time(
                hour=int(element / 60),
                minute=int(element % 60),
                second=int((element % 1) * 60),
            )
            for element in output
        ]
    return output


def BINS(array, break_points):
    return pd.cut(array, break_points)


def SORT(values, ascending=True, by=None):
    if type(values).__name__ == "Series":
        return values.sort_values(ascending=ascending)
    elif type(values).__name__ == "DataFrame":
        return values.sort_values(by=by, ascending=ascending)
    elif type(values).__name__ == "list":
        return sorted(values, reverse=not ascending)


def LIMIT(values, limit):
    if type(values).__name__ == "list":
        return values[:limit]
    else:
        return values.iloc[limit, :]


def VARIANCE(values):
    return np.var(values, dtype=np.float64)


def STDEV(values):
    return pow(VARIANCE(values), 0.5)
