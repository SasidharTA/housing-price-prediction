import pptx
from tigerml.core.utils import slugify

prs = pptx.Presentation()


class slide_layouts:
    """Slide layouts class initializer."""

    def __init__(self):
        """Slide layout class initializer."""
        for layout in prs.slide_layouts:
            setattr(self, slugify(layout.name), layout)
        self.list = [slugify(layout.name) for layout in prs.slide_layouts]


layouts = slide_layouts()
