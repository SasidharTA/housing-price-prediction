from ...contents.Component import Component, ComponentGroup
from ..helpers import write_title


class ExcelComponent(Component):
    """Excel component class."""

    def __init__(self, content, dashboard, group=None):
        # import pdb
        # pdb.set_trace()
        assert content.__module__.startswith(
            "tigerml.core.reports.excel"
        ) or content.__module__.startswith("xlsxwriter")
        super().__init__(content, parent=dashboard)
        self.dashboard = dashboard
        self.group = group
        self.left_col = 0
        self.top_row = 0

    def save_content(self, worksheet, workbook, top_row, left_col):
        """Saves content for Excel component class."""
        top_row = int(top_row)
        left_col = int(left_col)
        if self.content.__module__.startswith("tigerml.core.reports.excel"):
            from tigerml.core.reports.excel import ExcelText

            if isinstance(self.content, ExcelText):
                if self.group:
                    current_index = self.group.components.index(self)
                    try:
                        width = self.group.components[
                            current_index - self.group.columns
                        ].width
                    except IndexError:
                        try:
                            width = self.group.components[
                                current_index + self.group.columns
                            ].width
                        except IndexError:
                            width = None
                else:
                    width = (
                        self.on_above.width
                        if self.on_above
                        else self.on_below.width
                        if self.on_below
                        else None
                    )
                if width:
                    width = int(width)
                self.content.save(worksheet, workbook, top_row, left_col, width=width)
            else:
                self.content.save(worksheet, workbook, top_row, left_col)
            # else:
            # 	self.content.save(worksheet, top_row, left_col)
        else:
            from xlsxwriter.chart import Chart
            from xlsxwriter.drawing import Drawing

            if isinstance(self.content, Drawing):
                worksheet.insert_image(self.content)
            elif isinstance(self.content, Chart):
                worksheet.insert_chart(self.content)

    def save_to(self, worksheet, workbook):
        """Saves to Excel component class."""
        if self.on_left:
            self.left_col = int(
                round(
                    self.on_left.left_col
                    + self.on_left.width
                    + self.dashboard.hor_spacing
                )
            )
            self.top_row = self.on_left.top_row
        if self.on_above:
            self.top_row = int(
                round(
                    self.on_above.top_row
                    + self.on_above.height
                    + self.dashboard.ver_spacing
                )
            )
            self.left_col = self.on_above.left_col
        self.save_content(worksheet, workbook, self.top_row, self.left_col)


class ExcelComponentGroup(ComponentGroup):
    """Excel component group class."""

    def append(self, content):
        """Appends for Excel component group class."""
        self.components.append(ExcelComponent(content, self.dashboard, self))

    def save(self, worksheet, workbook, top_row, left_col):
        """Saves for Excel component group class."""
        if self.need_title_space:
            write_title(self.name, worksheet, workbook, top_row, left_col, self.width)
            top_row += 1
        for index, component in enumerate(self.components):
            row_num = int(index / self.columns)
            col_num = index % self.columns
            tr = top_row + sum([self.row_height(row) for row in range(0, row_num)])
            lc = left_col + sum([self.column_width(col) for col in range(0, col_num)])
            # import pdb
            # pdb.set_trace()
            if self.name:
                if component.content.name:
                    component.content.name = self.name + " - " + component.content.name
                elif isinstance(component.content, ComponentGroup):
                    component.content.name = self.name
            if isinstance(component.content, ComponentGroup):
                lc = left_col
            component.save_content(worksheet, workbook, tr, lc)
