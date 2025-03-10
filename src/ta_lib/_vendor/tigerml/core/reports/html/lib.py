import os
from tigerml.core.utils import time_now_readable

from .contents import HTMLText
from .Report import HTMLComponent, HTMLComponentGroup, HTMLDashboard, HTMLReport


def create_html_report(
    contents, columns=2, save=True, name="", path="", split_sheets=True
):
    if not name:
        name = "report_at_{}".format(time_now_readable())
    report = HTMLReport(name)
    assert isinstance(contents, dict), "contents should be in the form of a dict"
    if split_sheets:
        needs_folder = False
        for content in contents:
            content_name = content
            content = contents[content_name]
            if not isinstance(content, dict):
                content = {"": content}
            report_dict, needs_folder_local = create_html_dashboard(
                content, name=content_name, columns=columns
            )
            needs_folder = needs_folder or needs_folder_local
            report.append_dashboard(report_dict)
    else:
        report_dict, needs_folder = create_html_dashboard(
            contents, name=name, columns=columns
        )
        report.append_dashboard(report_dict)
    if save:
        report.save(path=path, needs_folder=needs_folder)
    else:
        return report


def create_html_dashboard(contents, name="", columns=2, flatten=False):
    dash = HTMLDashboard(name=name)
    # for content in contents:
    # 	content_name = content
    # 	content = contents[content_name]
    cg, needs_folder = create_component_group(
        contents, dash, columns=columns, flatten=flatten
    )
    dash.append(cg)
    return dash, needs_folder


def group_components(components, dashboard, name="", columns=2, flatten=False):
    if [component for component in components if isinstance(component, tuple)]:
        final_cg = HTMLComponentGroup(dashboard, name=name, columns=1)
        current_cg = None
        for component in components:
            if isinstance(component, tuple):
                # import pdb
                # pdb.set_trace()
                if current_cg:
                    import copy

                    old_cg = copy.copy(current_cg)
                    # old_cg.name = ''
                    # final_cg = HTMLComponentGroup(dashboard, name=name, columns=1)
                    final_cg.append(old_cg)
                new_cg = group_components(
                    component[1],
                    dashboard,
                    component[0],
                    columns=columns,
                    flatten=flatten,
                )
                final_cg.append(new_cg)
                current_cg = None
            else:
                if not current_cg:
                    current_cg = HTMLComponentGroup(dashboard, name="", columns=columns)
                current_cg.append(component)
        if current_cg:
            final_cg.append(current_cg)
    else:
        final_cg = HTMLComponentGroup(dashboard, name=name, columns=columns)
        for component in components:
            final_cg.append(component)

    # if final_cg != current_cg:
    # 	final_cg.append(current_cg)
    return final_cg


def create_component_group(contents, dashboard, name="", columns=2, flatten=False):
    from ..helpers import create_components

    needs_folder = False
    if isinstance(contents, str):
        components = [HTMLText(contents, name=dashboard.name)]
    else:
        components, needs_folder = create_components(
            contents, flatten=flatten, format="html"
        )
    # if len(contents) == 1:
    # 	columns = 1
    cg = group_components(
        components, dashboard, name=name, columns=columns, flatten=flatten
    )
    return cg, needs_folder


def _update_text_css(css_lines, text):
    """Update the CSS lines with the custom text content."""
    for i, line in enumerate(css_lines):
        if "--custom-content" in line:
            css_lines[i] = f"  --custom-content: '{text}';\n"
            break
    return css_lines


def _update_color_css(css_lines, color):
    """Update the CSS lines with the custom background color."""
    for i, line in enumerate(css_lines):
        if "--custom-background" in line:
            css_lines[i] = f"  --custom-background: {color};\n"
            break
    return css_lines


def report_configs(custom_text="Created with tigerml", custom_background="#FFA500"):
    """
    A function for managing custom CSS styles for HTML reports.

    This function provides the ability to set custom text content and background color
    for HTML reports's footer by updating a CSS file with the specified values.

    Parameters
    ----------
    custom_text : str, default="Created with tigerml"
        The custom text content to be injected into the CSS file.

    custom_background : str, default="#FFA500"
        The custom background color (in CSS format) to be injected into the CSS file.

    Examples
    --------
    Customize the report's footer with custom text and background color:

    >>> from tigerml.core.reports.html import report_configs

    >>> # Example 1: Change the report text to "Generated by MyReport"
    >>> report_configs(custom_text="Generated by MyReport")

    >>> # Example 2: Change the background color to red
    >>> report_configs(custom_background="#E74C3C")

    >>> # Example 3: Customize with a different text and color
    >>> report_configs(custom_text="Customized Text", custom_background="#3498DB")

    >>> # Example 4: Reset to default text and background color
    >>> report_configs()  # This sets it back to the defaults.

    >>> # Example 5: Change text and keep the default background color
    >>> report_configs(custom_text="New Custom Text")
    """
    css_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "report_resources/style.css")
    )

    with open(css_file_path, "r") as css_file:
        css_lines = css_file.readlines()

    # Set custom text and color
    updated_css_lines = _update_text_css(css_lines, custom_text)
    updated_css_lines = _update_color_css(updated_css_lines, custom_background)

    with open(css_file_path, "w") as css_file:
        css_file.writelines(updated_css_lines)
