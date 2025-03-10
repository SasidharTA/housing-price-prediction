import math


class Component:
    """Component class."""

    def __init__(self, content, parent):
        self.content = content
        self.parent = parent
        self.on_left = None
        self.on_above = None
        self.name = self.content.name

    # @property
    # def name(self):
    # 	return self.content.name

    @property
    def height(self):
        """Returns height."""
        if self.content.__module__.startswith("tigerml.core.reports.excel"):
            return self.content.height
        elif hasattr(self.content, "height"):
            return self.content.height
        else:
            return None

    @property
    def width(self):
        """Returns width."""
        if self.content.__module__.startswith("tigerml.core.reports.excel"):
            return self.content.width
        elif hasattr(self.content, "width"):
            return self.content.width / 100
        else:
            return None

    @property
    def on_right(self):
        """Returns module on right."""
        if [c for c in self.parent.components if c.on_left == self]:
            return [c for c in self.parent.components if c.on_left == self][0]
        return None

    @property
    def on_below(self):
        """Returns module below."""
        if [c for c in self.parent.components if c.on_above == self]:
            return [c for c in self.parent.components if c.on_above == self][0]
        return None


class ComponentGroup:
    """Component group class."""

    def __init__(self, dashboard, name="", columns=2):
        self.dashboard = dashboard
        self.columns = columns
        self.components = list()
        self.name = name
        self.show_titile = False

    def column_width(self, col_num):
        """Returns width."""
        components_in_column = [
            c
            for index, c in enumerate(self.components)
            if index % self.columns == col_num
        ]
        return max([c.width for c in components_in_column]) + self.dashboard.hor_spacing

    def column_height(self, col_num):
        """Returns height."""
        components_in_column = [
            c
            for index, c in enumerate(self.components)
            if index % self.columns == col_num
        ]
        return (
            sum([c.height for c in components_in_column])
            + (len(components_in_column) - 1) * self.dashboard.ver_spacing
        )

    def row_width(self, row_num):
        """Returns row width."""
        components_in_row = self.components[row_num : row_num + self.columns]
        return (
            sum([c.width for c in components_in_row])
            + (len(components_in_row) - 1) * self.dashboard.hor_spacing
        )

    def row_height(self, row_num):
        """Returns row height."""
        components_in_row = self.components[row_num : row_num + self.columns]
        return max([c.height for c in components_in_row]) + self.dashboard.ver_spacing

    @property
    def width(self):
        """Returns width."""
        no_of_rows = math.ceil(len(self.components) / float(self.columns))
        row_widths = [0]
        for row in range(0, no_of_rows):
            row_widths.append(self.row_width(row))
        return max(row_widths)

    @property
    def height(self):
        """Returns height."""
        column_heights = []
        for column in range(0, self.columns):
            column_heights.append(self.column_height(column))
        return max(column_heights) + (1 if self.need_title_space else 0)

    @property
    def need_title_space(self):
        """Returns name and title."""
        return self.name and self.show_titile
