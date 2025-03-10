import os
from datetime import datetime
from tigerml.core.utils import create_safe_filename

from ...contents import Image


class HTMLImage(Image):
    """Html image group class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, format="svg")

    @classmethod
    def from_parent(cls, parent):
        """Returns parent class for Html image group class."""
        return cls(parent.image_data)

    def to_html(self, resource_path=""):
        """Converts to html for Html image group class."""
        # html_str = ''
        # if self.name:
        # 	html_str += title_html(prettify_slug(self.name))
        if not self.name:
            self.name = create_safe_filename("image_at_{}".format(datetime.now()))
        # path = super().save(name=resource_path + '/' + self.name)
        # relative_path = os.path.split(resource_path)[-1] + '/' + self.name + '.png'
        # image_html = '<img src="{}">'.format(relative_path)
        image_html = self.image_data
        return (
            '<div class="content image_content"><div class='
            '"content_inner">{}</div></div>'.format(image_html)
        )

    def save(self, name=""):
        """Saves for Html image group class."""
        html_str = "<html><body>{}</body></html>".format(self.to_html())
        if not name:
            name = self.name
        name = name + (".html" if name[:-5] != ".html" else "")
        f = open(name, "w")
        f.write(html_str)
        f.close()
