from ...contents import Chart


class PptChart(Chart):
    """Ppt chart class."""

    @classmethod
    def from_parent(cls, parent):
        """Returns parent class for Ppt chart class."""
        return cls(parent.plot)

    def save(self, placeholder, slide_obj):
        """Saves for Ppt chart class."""
        pass
