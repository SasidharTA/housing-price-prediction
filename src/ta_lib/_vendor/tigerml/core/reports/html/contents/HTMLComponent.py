from tigerml.core.utils import prettify_slug, slugify

from ...contents.Component import Component, ComponentGroup
from ..helpers import title_html


class HTMLComponent(Component):
    """Html component class."""

    def __init__(self, content, dashboard, group=None):
        assert (
            content.__module__.startswith("tigerml.core.reports.html")
            or content.__class__.__name__ == "HTML"
        )
        super().__init__(content, parent=dashboard)
        self.dashboard = dashboard
        self.group = group
        self.show_title = True

    def _get_link(self, parent_name=""):
        """Gets link for Html component class."""
        if (
            isinstance(self.content, ComponentGroup)
            and self.content.components
            and (len(self.content.components) > 1 or self.content.components[0].name)
        ):
            if not [x for x in self.content.components if x.content.name]:
                link_str = '<a id="nav_to_{}" href="#{}">{}</a>'.format(
                    slugify(self.content.name),
                    slugify(self.content.name),
                    prettify_slug(self.name),
                )
            else:
                link_str = '<a id="nav_to_{}" class="closed">{}</a>'.format(
                    slugify(self.content.name), prettify_slug(self.name)
                )
                if not self.name:
                    return self.content._get_links(parent_name)[
                        4:-5
                    ]  # remove the <ul> links
                links_to_children = self.content._get_links()
                link_str = '<li class="group">{}{}</li>'.format(
                    link_str, links_to_children
                )
        else:
            link_str = '<a id="nav_to_{}" href="#{}">{}</a>'.format(
                slugify(self.content.name),
                slugify(self.content.name),
                prettify_slug(self.name),
            )
        return link_str

    def to_html(self, resource_path):
        """Converts to html for Html component class."""
        # print('Saving {}, {}'.format(self.name, self.content.name))
        from ..contents import HTMLImage

        content_html = ""
        if (
            self.content.name
            and self.show_title
            and not isinstance(self.content, HTMLComponentGroup)
        ):
            content_html += title_html(prettify_slug(self.content.name))
        if isinstance(self.content, HTMLImage) or isinstance(
            self.content, HTMLComponentGroup
        ):
            content_html += self.content.to_html(resource_path=resource_path)
        else:
            content_html += self.content.to_html()
        # print('Name changed to {}, {}'.format(self.name, self.content.name))
        return '<div class="component" id="{}">{}</div>'.format(
            slugify(self.content.name), content_html
        )

    def save(self, name=""):
        """Saves for Html component class."""
        html_str = "<html><body>{}</body></html>".format(
            self.to_html(resource_path=".")
        )
        if not name:
            name = self.name
        name = name + (".html" if name[-5:] != ".html" else "")
        f = open(name, "w")
        f.write(html_str)
        f.close()


class HTMLComponentGroup(ComponentGroup):
    """Html component group class."""

    def append(self, content):
        """Appends for Html component group class."""
        self.components.append(HTMLComponent(content, self.dashboard, self))

    def _get_links(self, parent_name=""):
        """Gets links for Html component group class."""
        link_html = ""
        if len(self.components) == 1:
            link_html = self.components[0]._get_link(parent_name)
        else:
            for component in self.components:
                link_html += "<li>{}</li>".format(component._get_link(parent_name))
        link_html = "<ul>{}</ul>".format(link_html)
        return link_html

    def to_html(self, resource_path):
        """Converts to html for Html component group class."""
        html_str = ""
        if len(self.components) == 1 and not isinstance(
            self.components[0].content, ComponentGroup
        ):
            if self.name and self.components[0].content.name:
                self.name += " - "
            self.name += (
                self.components[0].content.name
                if self.components[0].content.name
                else ""
            )
            self.components[0].show_title = False
        for component in self.components:
            if self.name:
                if component.content.name:
                    component.content.name = self.name + " - " + component.content.name
                elif isinstance(component.content, ComponentGroup):
                    component.content.name = self.name
            html_str += component.to_html(resource_path=resource_path)
        if self.name:
            html_str = title_html(prettify_slug(self.name)) + html_str
        columns = min(len(self.components), self.columns)
        return '<div class="component_group cols_{}">{}</div>'.format(
            str(columns), html_str
        )

    def save(self, name=""):
        """Saves for Html component group class."""
        html_str = "<html><body>{}</body></html>".format(
            self.to_html(resource_path=".")
        )
        if not name:
            name = self.name
        name = name + (".html" if name[-5:] != ".html" else "")
        f = open(name, "w")
        f.write(html_str)
        f.close()
