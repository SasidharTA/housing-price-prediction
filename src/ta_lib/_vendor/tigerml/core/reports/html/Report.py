import os
from tigerml.core.utils import get_extension_of_path, prettify_slug, slugify

from ..contents.Component import ComponentGroup
from .contents.HTMLComponent import HTMLComponent, HTMLComponentGroup
from .helpers import has_single_component, title_html

HERE = os.path.split(__file__)[0]

opp_direction = {"left": "right", "right": "left", "above": "below", "below": "above"}


class HTMLReport:
    """Html report class."""

    def __init__(self, name):
        self.name = name
        self.dashboards = list()

    def add_dashboard(self, name):
        """Adds dashboard for Html report class."""
        self.dashboards.append(HTMLDashboard(name))

    def append_dashboard(self, dashboard):
        """Appends dashboard for Html report class."""
        assert isinstance(dashboard, HTMLDashboard)
        self.dashboards.append(dashboard)
        return self

    def _get_nav(self):
        """Gets nav for Html report class."""
        nav_html = ""
        for dashboard in self.dashboards:
            nav_html += dashboard._get_nav()
        return nav_html

    def to_html(self, resource_path=""):
        """Converts to html for Html report class."""
        html_str = ""
        for dashboard in self.dashboards:
            html_str += dashboard.to_html(resource_path=resource_path)
        return html_str

    def save(self, path="", needs_folder=True):
        """Saves to html for Html report class."""
        import os

        html_str = ""
        if path:
            ext = get_extension_of_path(path)
            if ext:
                assert ext == ".html", "given extension of file is not html."
            else:
                path = os.path.join(path, self.name + ".html")
        else:
            path = self.name + ".html"
        if "/" in path:
            folder_path, file_name = path.rsplit("/", maxsplit=1)
        else:
            file_name = path
            folder_path = "."
        from tigerml.core.utils import check_or_create_path

        check_or_create_path(folder_path)
        if needs_folder:
            resource_path = folder_path + "/" + file_name[:-5] + "_files"
            check_or_create_path(resource_path)
        else:
            resource_path = ""
        for dashboard in self.dashboards:
            html_str += dashboard.to_html(resource_path=resource_path)
        html_file = open(path, "w", encoding="utf-8")
        from bokeh.resources import CDN, INLINE

        template_file = open(
            os.path.join(HERE, "report_resources", "template.html"), "r"
        )
        html_content = template_file.read()
        template_file.close()
        style_file = open(os.path.join(HERE, "report_resources", "style.css"), "r")
        custom_style = style_file.read()
        style_file.close()
        filesaver_script_file = open(
            os.path.join(HERE, "report_resources", "FileSaver.min.js"), "r"
        )
        custom_script = filesaver_script_file.read()
        script_file = open(os.path.join(HERE, "report_resources", "script.js"), "r")
        custom_script += script_file.read()
        script_file.close()
        filesaver_script_file.close()
        html_content = (
            html_content.replace("{{report_title}}", prettify_slug(self.name))
            .replace("{{report_head}}", INLINE.render())
            .replace("{{custom_style}}", custom_style)
            .replace("{{custom_script}}", custom_script)
            .replace("{{file_name}}", file_name)
            .replace("{{report_nav}}", self._get_nav())
            .replace("{{resource_path}}", resource_path)
            .replace("{{report_body}}", html_str)
        )
        html_file.write(html_content)
        html_file.close()


class HTMLDashboard:
    """Html dashboard class."""

    def __init__(self, name):
        self.name = name
        self.components = list()

    def add_component_group(self, name="", columns=2):
        """Adds component group for Html dashboard class."""
        cg = HTMLComponentGroup(self, name=name, columns=columns)
        self.append(cg)
        return cg

    def insert(self, content, rel_position="below", rel_component_id=None):
        """Inserts for Html dashboard class."""
        if [component for component in self.components if component.content == content]:
            Warning("{} is already in the dashboard".format(content))
        component = HTMLComponent(content, self)
        self.components.append(component)
        if rel_component_id:
            rel_component = self.get_component(rel_component_id)
        elif [
            c
            for c in self.components[:-1]
            if getattr(c, "on_{}".format(rel_position)) is None
        ]:
            rel_component = [
                c
                for c in self.components[:-1]
                if getattr(c, "on_{}".format(rel_position)) is None
            ][-1]
        else:
            rel_component = None
        if rel_component:
            if rel_position in ["left", "above"]:
                if getattr(rel_component, "on_{}".format(rel_position)):
                    existing_component = getattr(
                        rel_component, "on_{}".format(rel_position)
                    )
                    setattr(content, "on_{}".format(rel_position), existing_component)
                setattr(rel_component, "on_{}".format(rel_position), component)
            else:
                if getattr(rel_component, "on_{}".format(rel_position)):
                    existing_component = getattr(
                        rel_component, "on_{}".format(rel_position)
                    )
                    setattr(
                        existing_component,
                        "on_{}".format(opp_direction[rel_position]),
                        component,
                    )
                setattr(
                    component,
                    "on_{}".format(opp_direction[rel_position]),
                    rel_component,
                )
        return len(self.components)

    def append(self, content):
        """Appends for Html dashboard class."""
        return self.insert(content, rel_position="below", rel_component_id=-1)

    def get_component(self, position):
        """Gets component for Html dashboard class."""
        if position == 0:
            raise Exception("position should be greater than or less than 0")
        try:
            return self.components[position - 1]
        except IndexError:
            return None

    def _get_nav(self):
        """Gets nav for Html dashboard class."""
        link_str = '<a id="nav_to_{}" class="closed">{}</a>'.format(
            slugify(self.name), prettify_slug(self.name)
        )
        children_links = ""
        if len(self.components) == 1 and has_single_component(self.components[0]):
            links = "<li>{}</li>".format(
                link_str.replace('class="closed"', f'href="#{slugify(self.name)}"')
            )
        else:
            for component in self.components:
                parent_name = None
                if isinstance(component.content, ComponentGroup):
                    parent_name = self.name
                children_links += "<li>{}</li>".format(
                    component._get_link(parent_name=parent_name)
                )
            links = '<li class="group">{}<ul>{}</ul></li>'.format(
                link_str, children_links
            )
        return links

    def to_html(self, resource_path):
        """Converts to html for Html dashboard class."""
        html_str = ""
        if self.name:
            html_str += title_html(prettify_slug(self.name))
        for component in self.components:
            html_str += component.to_html(resource_path=resource_path)
        html_str = '<div class="dashboard" id="{}">{}</div>'.format(
            slugify(self.name), html_str
        )
        return html_str

    def save(self, html_file, resource_path):
        """Saves for Html dashboard class."""
        html_str = self.to_html(resource_path)
        html_file.write(html_str)
        return html_file
