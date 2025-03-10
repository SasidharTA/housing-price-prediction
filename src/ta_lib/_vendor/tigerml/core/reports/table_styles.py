import numpy as np
import pandas as pd

# from bunch import Bunch
from tigerml.core.utils import DictObject

styles = DictObject({"width": "width", "height": "height"})

options = DictObject({})

preset_styles = DictObject(
    {
        "percent": {"num_format": "0.00%"},
        "time_format": {"num_format": "hh:mm:ss.000"},
        "date_format": {"num_format": "yyyy:mm:dd"},
        "datetime_format": {"num_format": "yyyy:mm:dd hh:mm:ss.000"},
        "align_left": {"align": "left"},
        "align_right": {"align": "right"},
        "align_center": {"align": "center"},
        "more_is_bad": {
            "type": "2_color_scale",
            "min_color": "#ffd9d9",
            "max_color": "#ee2626",
        },
        "less_is_bad": {
            "type": "2_color_scale",
            "min_color": "#ee2626",
            "max_color": "#ffd9d9",
        },
        "more_is_good": {
            "type": "2_color_scale",
            "min_color": "#caffc8",
            "max_color": "#168c12",
        },
        "less_is_good": {
            "type": "2_color_scale",
            "min_color": "#168c12",
            "max_color": "#caffc8",
        },
    }
)


def highlight_max(data, color="yellow"):
    """Highlight the maximum in a Series or DataFrame."""
    attr = "background-color: {}".format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else "" for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(
            np.where(is_max, attr, ""), index=data.index, columns=data.columns
        )


def highlight_min(data, color="green"):
    """Highlight the minimum in a Series or DataFrame."""
    attr = "background-color: {}".format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else "" for v in is_min]
    else:  # from .apply(axis=None)
        is_min = data == data.max().min()
        return pd.DataFrame(
            np.where(is_min, attr, ""), index=data.index, columns=data.columns
        )


def color_negative(val, color="red"):
    """Color negative.

    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = color if val < 0 else "black"
    return "color: %s" % color


def color_positive(val, color="green"):
    """Color positive.

    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = color if val > 0 else "black"
    return "color: %s" % color


def percentage_format(s):
    return "number-format: 0.00%".format()


def time_format(s):
    return "number-format: hh:mm:ss.000".format()


def date_format(s):
    return "number-format: yyyy:mm:dd".format()


def datetime_format(s):
    return "number-format: yyyy:mm:dd hh:mm:ss.000".format()


def set_commas(s):
    return "number-format: {:,}".format(s)


def left_align(s):
    return "text-align: {:<}".format(s)


def right_align(s):
    return "text-align: {:>}".format(s)


def center_align(s):
    return "text-align: {:=}".format(s)


def get_max_width(col, styler=None):
    col = col.astype(str)
    col_name_len = get_header_width(col, styler)
    max_col_length = 0
    if len(col) > 0:
        if styler is not None and col.name is not None:
            from tigerml.core.utils import flatten_list

            index = list(styler.data.columns).index(col.name)
            cell_values = [
                cell["display_value"]
                for cell in flatten_list(styler._translate()["body"])
                if "col{}".format(index) in cell["id"]
            ]
            max_col_length = max([len(str(val)) for val in cell_values])
        else:
            max_col_length = max(col.map(lambda c: len(c)))
    return max(max_col_length * 0.9, col_name_len)


def get_header_width(col, styler=None):
    max_width = 0
    if col.name:
        if isinstance(col.name, tuple) and styler is not None:
            levels = len(col.name)
            for level in range(0, levels):
                no_of_cols_under_level = len(
                    [
                        x
                        for x in styler.data.columns
                        if col.name[: level + 1] == x[: level + 1]
                    ]
                )
                max_width = max(
                    max_width, len(col.name[level]) * 0.9 / no_of_cols_under_level
                )
            return max_width
        else:
            return len(str(col.name)) * 0.9
    else:
        return max_width
