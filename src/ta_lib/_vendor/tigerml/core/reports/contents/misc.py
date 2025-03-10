class Text:
    """Text class."""

    def __init__(self, text, name="", width=1, height=1, format={}):
        self.text = text
        self.name = name
        self.text_width = int(width)
        self.text_height = height
        self.format = format

    @property
    def width(self):
        """Returns width."""
        return self.text_width

    @property
    def height(self):
        """Returns height."""
        return self.text_height + (1 if self.name else 0)


class BaseContent:
    """Base content class."""

    def __init__(self, content, name=""):
        self.content = content
        self.name = name

    @property
    def width(self):
        """Returns width."""
        return None

    @property
    def height(self):
        """Returns Height."""
        return None
