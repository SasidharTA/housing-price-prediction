from ...contents import Text


class PptText(Text):
    """Ppt text class."""

    @classmethod
    def from_parent(cls, parent):
        """Returns parent class for Ppt text class."""
        return cls(parent.text)

    def save(self, placeholder, slide_obj):
        """Saves for Ppt text class."""
        placeholder.text = self.text
