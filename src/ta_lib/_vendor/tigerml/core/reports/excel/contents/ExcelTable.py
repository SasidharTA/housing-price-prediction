import pandas as pd
from tigerml.core.utils import flatten_list

from ...contents import Table
from ..helpers import *


def excel_column_string(n):
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


# no_conditional_formats = ['font_name', 'font_size', 'font_script', 'locked',
#                           'hidden', 'diag_type', 'diag_border',
#                           'align', 'valign', 'rotation', 'text_wrap',
#                           'reading_order', 'text_justlast',
#                           'center_across', 'indent', 'shrink']


class ExcelTable(Table):
    """Excel table class."""

    @classmethod
    def from_parent(cls, parent):
        """Returns parent for Excel table class."""
        return cls(parent.styler)

    def set_params(
        self,
        na_rep=None,
        float_format=None,
        columns=None,
        header=None,
        index=True,
        index_label=None,
        merge_cells=True,
        inf_rep="inf",
        show_title=True,
    ):
        """Sets params for Excel table class."""
        for key in self.params:
            if eval(key) is not None:
                self.params[key] = eval(key)
        return self

    def save(self, worksheet, workbook, startrow=0, startcol=0):
        """Saves for Excel table class."""
        if self.name and self.show_title:
            write_title(self.name, worksheet, workbook, startrow, startcol, self.width)
            startrow += 1
        self.to_excel(workbook, worksheet.name, startrow=startrow, startcol=startcol)
        return workbook.get_worksheet_by_name(worksheet.name)

    def to_excel(self, workbook, sheet_name="Sheet1", startrow=0, startcol=0):
        """Converts to excel for Excel table class."""
        # import pdb
        # pdb.set_trace()
        excel_writer = pd.ExcelWriter(workbook.filename, engine="xlsxwriter")
        excel_writer.book = workbook
        excel_writer.sheets = workbook.sheetnames
        self.data = self.data[self.column_order]
        self.styler.data = self.styler.data[self.column_order]
        new_index, indexer = self.styler.columns.reindex(self.column_order)
        self.styler.columns = new_index
        for styler_rule in self.styler_rules:
            if "map" in styler_rule:
                styler = self.styler[styler_rule["map"]]
            else:
                styler = self.styler
            styler.applymap(
                styler_rule["func"],
                subset=styler_rule["subset"] if "subset" in styler_rule else None,
            )
        # self.params['header'] = False
        # self.params['index'] = False
        self.styler.to_excel(
            excel_writer,
            sheet_name=sheet_name,
            startrow=startrow,
            startcol=startcol,
            **self.params,
        )
        worksheet = excel_writer.sheets[sheet_name]
        # header_format = workbook.add_format({
        #     'bold': True,
        #     'text_wrap': True,
        #     'valign': 'top',
        #     'fg_color': '#D7E4BC',
        #     'border': 0})
        col_idx_format = workbook.add_format({'bold': True, 'border': 0})  # noqa

        for col_num, value in enumerate(self.styler.data.columns.values):
            try:
                worksheet.write(startrow, col_num + startcol + 1, value, col_idx_format)  # noqa
            except TypeError:  # TypeError: Unsupported type <class 'tuple'> in write()
                worksheet.write(startrow, col_num + startcol + 1, str(value), col_idx_format)  # noqa
        for row_num, value in enumerate(self.styler.data.index.values):
            worksheet.write(row_num + startrow + 1, startcol, value, col_idx_format)  # noqa

        if sheet_name == 'Index Sheet':
            for row_num, value in enumerate(self.styler.data['Sheet_No'].values):
                worksheet.write_url(row_num + startrow + 1, startcol + 1, f"internal:'{value}'!A1", string=value)  # noqa

        workbook = excel_writer.book
        if self.params["index"]:
            col_add = self.index_width
        else:
            col_add = 0
        if self.params["header"]:
            row_add = self.header_height
        else:
            row_add = 0
        worksheet = workbook.get_worksheet_by_name(sheet_name)
        if self.conditional_formatters:
            # import pdb
            # pdb.set_trace()
            for format_rule in self.conditional_formatters:
                if format_rule["style"]:
                    format_obj = workbook.add_format(format_rule["style"])
                    format_rule["options"].update({"format": format_obj})
                if format_rule["cols"] is not None:
                    for col in format_rule["cols"]:
                        col_index = list(self.data.columns).index(col) + col_add
                        worksheet.conditional_format(
                            row_add,
                            col_index,
                            row_add + len(self.data),
                            col_index,
                            options=format_rule["options"],
                        )
                if format_rule["rows"] is not None:
                    for row in format_rule["rows"]:
                        row_index = list(self.data.index).index(row) + row_add
                        worksheet.conditional_format(
                            row_index,
                            col_add,
                            row_index,
                            len(self.data.columns) + col_add,
                            options=format_rule["options"],
                        )
                if format_rule["index"] is not None:
                    # import pdb
                    # pdb.set_trace()
                    col_start_index = 0 + startcol
                    if format_rule["index"] is True:
                        col_range = range(
                            col_start_index,
                            col_start_index + len(self.data.index.names),
                        )
                    elif isinstance(format_rule["index"], list):
                        col_range = [
                            (col_start_index + level) for level in format_rule["index"]
                        ]
                    else:
                        continue
                    for col_index in col_range:
                        worksheet.conditional_format(
                            0,
                            col_index,
                            len(self.data),
                            col_index,
                            options=format_rule["options"],
                        )
                if format_rule["header"] is not None:
                    row_start_index = 0 + startrow
                    if format_rule["header"] is True:
                        row_range = range(
                            row_start_index,
                            row_start_index + len(self.data.columns.names),
                        )
                    elif isinstance(format_rule["header"], list):
                        row_range = [
                            (row_start_index + level) for level in format_rule["header"]
                        ]
                    else:
                        continue
                    for row_index in row_range:
                        worksheet.conditional_format(
                            row_index,
                            0,
                            row_index,
                            len(self.data.columns),
                            options=format_rule["options"],
                        )
                if format_rule["bool_map"] is not None:
                    # TODO: Finish this
                    if format_rule["col_wise"]:
                        for col in format_rule["cols"]:
                            col_index = list(self.data.columns).index(col) + col_add
                            worksheet.conditional_format(
                                1,
                                col_index,
                                len(self.data),
                                col_index,
                                options=format_rule["options"],
                            )
                    else:
                        for row in format_rule["rows"]:
                            row_index = list(self.data.index).index(row) + row_add
                            worksheet.conditional_format(
                                row_index,
                                0,
                                row_index,
                                len(self.data.columns),
                                options=format_rule["options"],
                            )
        col_rules = [rule for rule in self.cell_formatters if rule["cols"] is not None]
        row_rules = [rule for rule in self.cell_formatters if rule["rows"] is not None]
        bool_rules = [
            rule for rule in self.cell_formatters if rule["bool_map"] is not None
        ]
        index_rules = [rule for rule in self.cell_formatters if rule["index"]]
        header_rules = [rule for rule in self.cell_formatters if rule["header"]]
        if col_rules:
            for col in set(flatten_list([x["cols"] for x in col_rules])) & set(
                list(self.data.columns)
            ):
                all_rules_for_col = [x for x in col_rules if col in x["cols"]]
                combined_style = {}
                combined_options = {}
                for each_rule in all_rules_for_col:
                    combined_style.update(each_rule["style"])
                    if each_rule["options"] is not None:
                        combined_options.update(each_rule["options"])
                width_val, style = process_col_style(
                    combined_style, self.data[col], self.styler
                )
                col_index = list(self.data.columns).index(col) + col_add + startcol
                style_format = workbook.add_format(style)
                worksheet.set_column(
                    col_index,
                    col_index,
                    width=width_val,
                    cell_format=style_format,
                    options=combined_options,
                )
        if row_rules:
            row_vals = list(set(flatten_list([x["rows"] for x in row_rules])))
            # import pdb
            # pdb.set_trace()
            if str(self.data.index.dtype) != "int64":
                int_rows = [x for x in row_vals if isinstance(x, int)]
                int_row_names = [self.data.index[x] for x in int_rows]
                row_vals = set(
                    [x for x in row_vals if not isinstance(x, int)] + int_row_names
                ) & set(self.data.index)
            for row in row_vals:
                all_rules_for_row = [
                    x
                    for x in row_rules
                    if row in x["rows"] or list(self.data.index).index(row) in x["rows"]
                ]
                combined_style = {}
                combined_options = {}
                for each_rule in all_rules_for_row:
                    combined_style.update(each_rule["style"])
                    if each_rule["options"] is not None:
                        combined_options.update(each_rule["options"])
                height_val, style = process_row_style(
                    combined_style, self.data.loc[row], self.styler
                )
                row_index = list(self.data.index).index(row) + row_add + startrow
                style_format = workbook.add_format(style)
                worksheet.set_row(
                    row_index,
                    height=height_val,
                    cell_format=style_format,
                    options=combined_options,
                )
        if self.params["index"] and index_rules:
            # import pdb
            # pdb.set_trace()
            if not isinstance(self.data.index, pd.MultiIndex):
                width_val, style, options = create_index_format(
                    index_rules, self.data.index
                )
                col_index = 0 + startcol
                style_format = workbook.add_format(style)
                worksheet.set_column(
                    col_index,
                    col_index,
                    width=width_val,
                    cell_format=style_format,
                    options=options,
                )
            else:
                for level in range(0, len(self.data.index.names)):
                    all_rules_for_index = [
                        x
                        for x in index_rules
                        if level in x["index"] or x["index"] is True
                    ]
                    width_val, style, options = create_index_format(
                        all_rules_for_index, self.data.index.get_level_values(level)
                    )
                    col_index = level + startcol
                    style_format = workbook.add_format(style)
                    worksheet.set_column(
                        col_index,
                        col_index,
                        width=width_val,
                        cell_format=style_format,
                        options=options,
                    )
        if self.params["header"] and header_rules:
            # import pdb
            # pdb.set_trace()
            if not isinstance(self.data.columns, pd.MultiIndex):
                height_val, style, options = create_header_format(
                    header_rules, self.data.columns
                )
                row_index = 0 + startrow
                style_format = workbook.add_format(style)
                worksheet.set_row(
                    row_index,
                    height=height_val,
                    cell_format=style_format,
                    options=options,
                )
            else:
                for level in range(0, len(self.data.columns.names)):
                    all_rules_for_header = [
                        x
                        for x in header_rules
                        if level in x["header"] or x["header"] is True
                    ]
                    height_val, style, options = create_index_format(
                        all_rules_for_header, self.data.columns.get_level_values(level)
                    )
                    row_index = level + startrow
                    style_format = workbook.add_format(style)
                    worksheet.set_row(
                        row_index,
                        height=height_val,
                        cell_format=style_format,
                        options=options,
                    )
        if bool_rules:
            pass
            # TODO: Finish this
        # excel_writer.save()
        # excel_writer.close()
