import pandas as pd
from pandas.io.formats.style import Styler


def validate_formatting_input(
    data, style, cols=None, rows=None, bool_map=None, index=None, header=None
):
    if not isinstance(style, dict) and not callable(style):
        raise Exception("style should be a dict or a function")
    if cols:
        if isinstance(cols, list):
            if [col for col in cols if col not in data]:
                raise Exception("some cols do not exist in data")
        else:
            raise Exception("cols should be list")
    if rows:
        if isinstance(rows, list):
            if [row for row in rows if row not in data.index] and [
                row for row in rows if not isinstance(row, int)
            ]:
                raise Exception("some rows do not exist in data")
        else:
            raise Exception("rows should be list")
    if bool_map:
        if isinstance(bool_map, pd.DataFrame):
            if bool_map.shape != data.shape:
                raise Exception("bool_map should be the same size as data")
        else:
            raise Exception("bool_map should a pandas DataFrame")
    if index is not None:
        if not isinstance(index, list) and not isinstance(index, bool):
            raise Exception(
                "index should either be boolean or list of levels (in case of multi-index)"
            )
    if header is not None:
        if not isinstance(header, list) and not isinstance(header, bool):
            raise Exception(
                "header should either be boolean or list of levels (in case of multi-index)"
            )
    return True


# no_conditional_formats = ['font_name', 'font_size', 'font_script', 'locked', 'hidden', 'diag_type', 'diag_border',
#                           'align', 'valign', 'rotation', 'text_wrap', 'reading_order', 'text_justlast',
#                           'center_across', 'indent', 'shrink']


class Table:
    """Table class."""

    def __init__(self, data, title="", datatable=True):
        self.name = title
        self.show_title = True
        self.name_style = []
        self.datatable = datatable
        if isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.MultiIndex):
                data.set_index(
                    pd.Index(data.index, name=data.index.names), inplace=True
                )
            self.data = data
            self.styler = self.data.style
        elif isinstance(data, Styler):
            if isinstance(data.index, pd.MultiIndex):
                self.data = data.data.set_index(
                    pd.Index(data.index, name=data.index.names)
                )
            else:
                self.data = data.data
            self.styler = data
        else:
            raise Exception("data should be a pandas DataFrame or a Styler.")
        self.cell_formatters = list()
        self.conditional_formatters = list()
        self.styler_rules = list()
        self.column_order = list(self.data.columns)
        self.params = {
            "na_rep": None,
            "float_format": None,
            "columns": None,
            "header": True,
            "index": True,
            "index_label": None,
            "merge_cells": True,
            "inf_rep": "inf",
        }
        self.header_height = len(self.data.columns.names)
        if (
            self.header_height > 1
        ):  # pandas adds one extra row in multi index columns to have the index name
            self.header_height += 1
        self.index_width = len(self.data.index.names)

    # def apply(self, func, axis=0, subset=None, *args, **kwargs):
    #     self.styler = self.styler.apply(func, axis=axis, subset=subset, *args, **kwargs)
    #     return self.styler
    #
    # def applymap(self, func, subset=None, **kwargs):
    #     self.styler = self.styler.applymap(func, subset=subset, **kwargs)
    #     return self.styler
    #
    # def format(self, formatter, subset=None, *args, **kwargs):
    #     self.styler = self.styler.format(formatter, subset=subset)
    #     return self.styler

    @property
    def height(self):
        """Returns height."""
        if self.data is not None:
            return (
                len(self.data)
                + (self.header_height if self.params["header"] else 0)
                + (1 if self.name else 0)
            )
        else:
            return None

    @property
    def width(self):
        """Returns width."""
        if self.data is not None:
            return len(self.data.columns) + (
                self.index_width if self.params["index"] else 0
            )
        else:
            return None

    def preview(self):
        """Returns styler."""
        return self.styler

    def apply_conditional_format(
        self,
        cols=None,
        rows=None,
        bool_map=None,
        index=None,
        header=None,
        col_wise=True,
        options=None,
        style={},
    ):
        """Applies conditional formatting."""
        validate_formatting_input(
            self.data,
            style={},
            cols=cols,
            rows=rows,
            bool_map=bool_map,
            index=index,
            header=header,
        )
        self.conditional_formatters.append(
            {
                "cols": cols,
                "rows": rows,
                "bool_map": bool_map,
                "options": options,
                "col_wise": col_wise,
                "style": style,
                "index": index,
                "header": header,
            }
        )
        return self

    def apply_cell_format(
        self,
        style,
        cols=None,
        rows=None,
        bool_map=None,
        index=None,
        header=None,
        options=None,
    ):
        """Applies cell formatting."""
        if isinstance(style, list):
            for ind_style in style:
                self.apply_cell_format(
                    ind_style,
                    cols=cols,
                    rows=rows,
                    bool_map=bool_map,
                    index=index,
                    header=header,
                    options=options,
                )
            return
        validate_formatting_input(
            self.data,
            style,
            cols=cols,
            rows=rows,
            bool_map=bool_map,
            index=index,
            header=header,
        )
        if isinstance(style, dict):
            if options or ("width" in style or "height" in style):
                new_style = {}
                if "width" in style:
                    new_style.update({"width": style.pop("width")})
                if "height" in style:
                    new_style.update({"height": style.pop("height")})
                self.cell_formatters.append(
                    {
                        "style": new_style,
                        "cols": cols,
                        "rows": rows,
                        "bool_map": bool_map,
                        "index": index,
                        "header": header,
                        "options": options,
                    }
                )
            if style:
                self.apply_conditional_format(
                    cols=cols,
                    rows=rows,
                    bool_map=bool_map,
                    options={"type": "blanks"},
                    style=style,
                    index=index,
                    header=header,
                )
                self.apply_conditional_format(
                    cols=cols,
                    rows=rows,
                    bool_map=bool_map,
                    options={"type": "no_blanks"},
                    style=style,
                    index=index,
                    header=header,
                )
        else:
            if cols:
                self.styler_rules.append({"func": style, "subset": cols})
                # self.styler.applymap(style, subset=cols)
            if rows:
                for row in rows:
                    self.styler_rules.append({"func": style, "subset": (row,)})
                    # self.styler.applymap(style, subset=(row,))
            if bool_map:
                self.styler_rules.append({"func": style, "map": bool_map})
                # self.styler[bool_map].applymap(style)
        return self

    def sort_columns(self, start=[], end=[], total_list=[]):
        """Sorts columns."""
        if set(start) & set(end):
            raise Exception("Same columns cannot occur in start and end lists")
        if total_list:
            if set(total_list) - set(self.column_order):
                raise Exception(
                    "Few columns in total_list do not exist in data - {}".format(
                        set(total_list) - set(self.column_order)
                    )
                )
            if len(total_list) < len(self.column_order):
                print(
                    "Not all columns are mentioned in total list. Appending the remaining columns at the end"
                )
                return self.sort_columns(start=total_list)
            self.column_order = total_list
            return self
        if start:
            if any([isinstance(col, tuple) for col in self.column_order]):
                new_start = []
                for val in start:
                    if [
                        col
                        for col in self.column_order
                        if val in col and isinstance(col, tuple)
                    ]:
                        new_start += [
                            col
                            for col in self.column_order
                            if val in col and isinstance(col, tuple)
                        ]
                    else:
                        new_start.append(val)
                start = new_start
            if set(start) - set(self.column_order):
                raise Exception(
                    "Few columns in start do not exist in data - {}".format(
                        set(start) - set(self.column_order)
                    )
                )
            self.column_order = start + [
                col for col in self.column_order if col not in start
            ]
        if end:
            if any([isinstance(col, tuple) for col in self.column_order]):
                new_end = []
                for val in end:
                    if [
                        col
                        for col in self.column_order
                        if val in col and isinstance(col, tuple)
                    ]:
                        new_end += [
                            col
                            for col in self.column_order
                            if val in col and isinstance(col, tuple)
                        ]
                    else:
                        new_end.append(val)
                end = new_end
            if set(end) - set(self.column_order):
                raise Exception(
                    "Few columns in start do not exist in data - {}".format(
                        set(end) - set(self.column_order)
                    )
                )
            self.column_order = [
                col for col in self.column_order if col not in end
            ] + end
        return self
