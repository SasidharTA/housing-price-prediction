class ContentGroup:
    """Content group class."""

    def __init__(self):
        self.contents = list()

    @property
    def slides(self):
        """Returns all slides for Content group class."""
        from .Report import Section

        all_slides = []
        for content in self.contents:
            if isinstance(content, Section):
                all_slides += content.slides
            else:
                all_slides.append(content)
        return all_slides

    def add_slide(self, layout=None, title=""):
        """Adds slide for Content group class."""
        from .Slide import Slide

        slide = Slide(layout=layout, title=title)
        self.contents.append(slide)
        return slide

    def add_section(self, title=""):
        """Adds section for Content group class."""
        from .Report import Section

        section = Section(title=title)
        self.contents.append(section)
        return section

    def insert(self, content, pos=None):
        """Inserts for Content group class."""
        if pos:
            self.contents.insert(content, pos)
        else:
            self.contents.append(content)

    def get_parent_and_index(self, slide_index):
        """Gets parent and index for Content group class."""
        return 0

    def _insert_content(self, content, slide_index=None, content_index=None):
        """Inserts content for Content group class."""
        if content_index:
            self.contents.insert(content, content_index)
        elif slide_index:
            parent, index = self.get_parent_and_index(slide_index)
            parent._insert_content(content, content_index=index)
        else:
            self.contents.append(content)
        return self

    def insert_slide(self, slide, slide_index=None, content_index=None):
        """Inserts slide for Content group class."""
        return self._insert_content(slide, slide_index, content_index)

    def remove_slide(self, index):
        """Removes slide for Content group class."""
        pass

    def remove_slide_by_title(self, title):
        """Removes slide by title for Content group class."""
        pass
