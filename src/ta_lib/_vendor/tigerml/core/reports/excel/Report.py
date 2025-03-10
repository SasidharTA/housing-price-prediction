import xlsxwriter
from tigerml.core.utils import check_or_create_path

from .contents.ExcelComponent import ExcelComponent, ExcelComponentGroup

opp_direction = {"left": "right", "right": "left", "above": "below", "below": "above"}


class ExcelDashboard:
    """Excel dashboard class."""

    def __init__(self, name, hor_spacing=1, ver_spacing=2):
        self.name = name
        self.hor_spacing = hor_spacing
        self.ver_spacing = ver_spacing
        self.components = list()

    def add_component_group(self, name="", columns=2):
        """Adds component group."""
        cg = ExcelComponentGroup(self, name=name, columns=columns)
        self.append(cg)
        return cg

    def insert(self, content, rel_position="below", rel_component_id=None):
        """Excel dashboard class."""
        if [component for component in self.components if component.content == content]:
            Warning("{} is already in the dashboard".format(content))
        component = ExcelComponent(content, self)
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
        """Excel dashboard class."""
        return self.insert(content, rel_position="below", rel_component_id=-1)

    def get_component(self, position):
        """Excel dashboard class."""
        if position == 0:
            raise Exception("position should be greater than or less than 0")
        try:
            return self.components[position - 1]
        except IndexError:
            return None

    def to_excel(self, worksheet, workbook):
        """Excel dashboard class."""
        for component in self.components:
            component.save_to(worksheet, workbook)

    def save_to(self, workbook, sheet_name=None):
        """Excel dashboard class."""
        assert isinstance(sheet_name, str) or sheet_name is None
        if not sheet_name:
            sheet_name = self.name
            orig_sheet_name = sheet_name
            suffix = 1
            while sheet_name in workbook.sheetnames:
                sheet_name = orig_sheet_name + "_{}".format(suffix)
                suffix += 1
        from tigerml.core.utils import prettify_slug

        sheet_name = prettify_slug(sheet_name)
        if len(sheet_name) > 31:
            sheet_name = sheet_name.replace(" ", "")
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:30]
        worksheet = workbook.add_worksheet(prettify_slug(sheet_name))
        self.to_excel(worksheet, workbook)
        # return workbook


class ExcelReport:
    """Excel report class."""

    def __init__(self, filename, file_path=""):
        """Excel report class initializer."""
        if filename[:-5] != ".xlsx":
            filename += ".xlsx"
        if file_path and file_path[-1] != "/":
            file_path = file_path + "/"
        check_or_create_path(file_path)
        self.workbook = xlsxwriter.Workbook(file_path + filename)
        self.filename = filename
        self.dashboards = list()

    def add_dashboard(self, name=""):
        """Excel report class initializer."""
        if not name:
            name = "Sheet{}".format(len(self.dashboards) + 1)
        dashboard = ExcelDashboard(name)
        self.dashboards.append(dashboard)
        return dashboard

    def insert_dashboard(self, dashboard, pos):
        """Excel report class initializer."""
        self.dashboards.insert(pos, dashboard)

    def append_dashboard(self, dashboard):
        """Excel report class initializer."""
        self.dashboards.append(dashboard)

    def save(self):
        """Excel report class initializer."""
        # import pdb
        # pdb.set_trace()
        for dashboard in self.dashboards:
            dashboard.save_to(self.workbook)
        self.workbook.close()
