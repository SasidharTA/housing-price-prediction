from ...contents import BaseContent, Text


class HTMLText(Text):
    """Html text class."""

    @classmethod
    def from_parent(cls, parent):
        """Returns parent class for Html text class."""
        return cls(parent.text)

    def to_html(self, resource_path=""):
        """Converts to html for Html text class."""
        # html_str = ''
        # if self.name:
        # 	html_str += title_html(prettify_slug(self.name))
        text_html = "<p>{}</p>".format(self.text)
        return (
            '<div class="content text_content"><div class='
            '"content_inner">{}</div></div>'.format(text_html)
        )

    def save(self):
        """Saves for Html text class."""
        pass


class HTMLBase(BaseContent):
    """Html base class."""

    def to_html(self, resource_path=""):
        """Converts to html for Html base class."""
        return self.content
