from ...contents import Chart

SHOW_DATA_OPTIONS = {
    "left": "left",
    "right": "right",
    "top": "top",
    "bottom": "bottom",
    "new_sheet": "new_sheet",
    "hidden": "hidden",
}


class ExcelChart(Chart):
    """Excel chart class."""

    @classmethod
    def from_parent(cls, parent):
        """Excel chart class."""
        return cls(parent.plot)

    def __init__(self, type, data, show_data="right", options=None):
        self.data = data
        self.type = type
        self.options = options
        assert show_data in SHOW_DATA_OPTIONS
        self.show_data = show_data
        super().__init__()

    def set_show_data(self, value):
        """Sets show data for Excel chart class."""
        assert value in SHOW_DATA_OPTIONS
        self.show_data = SHOW_DATA_OPTIONS[value]

    def save(self, worksheet):
        """Saves for Excel chart class."""
        pass
