import pandas as pd
import tigerml.core.dataframe as td
import warnings
from bokeh.util.warnings import BokehUserWarning
from collections.abc import Iterable
from pandas.io.formats.style import Styler
from tigerml.core.dataframe.helpers import detigerify
from tigerml.core.reports.contents import Table

warnings.filterwarnings("ignore", category=BokehUserWarning)


def get_component_in_format(component, format):
    if format == "html":
        module = "html"
        prefix = "HTML"
    elif format == "xlsx":
        module = "excel"
        prefix = "Excel"
    elif format == "pptx":
        module = "ppt"
        prefix = "Ppt"
    else:
        raise Exception("Incorrect input for format")
    class_name = component.__class__.__name__
    child_class_name = prefix + class_name
    module_path = "tigerml.core.reports.{}".format(module)
    exec("from {} import {}".format(module_path, child_class_name))
    child_class = eval(child_class_name)
    child_obj = child_class.from_parent(component)
    for att in [a for a in dir(component) if a not in dir(child_obj)]:
        try:
            setattr(child_obj, att, getattr(component, att))
        except Exception:
            pass
    return child_obj


TABLE_CLASS = "table"
CHART_CLASS = "chart"
IMAGE_CLASS = "image"
TEXT_CLASS = "text"
COMPONENT_CLASS = "component"
CG_CLASS = "component_group"


def get_component_classes(format):
    import tigerml.core.reports as tr

    if format == "html":
        return {
            TABLE_CLASS: tr.html.HTMLTable,
            CHART_CLASS: tr.html.HTMLChart,
            IMAGE_CLASS: tr.html.HTMLImage,
            TEXT_CLASS: tr.html.HTMLText,
            COMPONENT_CLASS: tr.html.HTMLComponent,
            CG_CLASS: tr.html.HTMLComponentGroup,
        }
    elif format == "xlsx":
        import tigerml.core.reports.excel as tre

        return {
            TABLE_CLASS: tre.ExcelTable,
            CHART_CLASS: tre.ExcelChart,
            IMAGE_CLASS: tre.ExcelImage,
            TEXT_CLASS: tre.ExcelText,
            COMPONENT_CLASS: tre.ExcelComponent,
            CG_CLASS: tre.ExcelComponentGroup,
        }
    else:
        raise Exception("Given format not recognised - {}".format(format))


def enforce_iterable(input):
    from collections.abc import Iterable

    if not isinstance(input, Iterable):
        input = [input]
    return input


def create_components(contents, flatten=False, format="html"):
    import tigerml.core.reports as tr

    assert format in ["html", "xlsx"]
    CLASSES = get_component_classes(format)
    components = []
    needs_folder = False
    for content in enforce_iterable(contents):
        if isinstance(contents, dict):
            content_name = content
            content = contents[content_name]
        else:
            content_name = ""
        if isinstance(content, str):
            component = CLASSES[TEXT_CLASS](content, name=content_name)
        elif str(content.__class__.__module__).startswith(
            "tigerml.core.reports.contents"
        ):
            component = get_component_in_format(content, format=format)
        elif isinstance(content, CLASSES[CG_CLASS]) or isinstance(
            content, CLASSES[COMPONENT_CLASS]
        ):
            component = content
        elif content.__class__.__name__ == "HTML":
            if format == "html":
                component = tr.html.HTMLBase(content)
            else:
                Warning("Passed an HTML component for Excel report. Skipping it.")
                component = None
        elif isinstance(content, pd.DataFrame) or isinstance(content, Styler):
            component = CLASSES[TABLE_CLASS](content, title=content_name)
        elif isinstance(content, td.DataFrame):
            from tigerml.core.utils import compute_if_dask

            component = CLASSES[TABLE_CLASS](
                compute_if_dask(content)._data, title=content_name
            )
        elif (
            type(content).__module__.startswith("holoviews")
            or type(content).__module__.startswith("hvplot")
            or type(content).__module__.startswith("bokeh")
            or type(content).__module__.startswith("plotly")
        ):
            if format == "html":
                component = CLASSES[CHART_CLASS](content, name=content_name)
            else:
                component = CLASSES[IMAGE_CLASS](content, name=content_name)
                needs_folder = True
        elif isinstance(content, Iterable):
            if flatten:
                component = create_components(content, flatten=True, format=format)
                if format == "html":
                    needs_folder_local = component[1]
                    component = component[0]
                    needs_folder = needs_folder or needs_folder_local
            else:
                component = (
                    content_name,
                    create_components(content, flatten=False, format=format),
                )
                if format == "html":
                    needs_folder_local = component[1][1]
                    component = (component[0], component[1][0])
                    needs_folder = needs_folder or needs_folder_local
        elif content.__class__ in CLASSES.values():
            component = content
        else:
            component = CLASSES[IMAGE_CLASS](content, name=content_name)
            needs_folder = True
        # if isinstance(component, list):
        # 	components += component
        # else:
        if component is not None:
            components.append(component)
        if flatten:
            from tigerml.core.utils import flatten_list

            components = flatten_list(components)
    if format == "html":
        return components, needs_folder
    else:
        return components


def format_tables_in_report(report, title=""):
    """Format all tables in report dictionary."""
    if isinstance(report, dict):
        for key, sub_report in report.items():
            report[key] = format_tables_in_report(sub_report, title=key)
    elif isinstance(report, list):
        for key, sub_report in enumerate(report):
            report[key] = format_tables_in_report(sub_report)
    else:
        if isinstance(report, td.DataFrame):
            report = detigerify(report)
        if isinstance(report, pd.DataFrame):
            report = Table(report, title=title, datatable=False)
    return report
