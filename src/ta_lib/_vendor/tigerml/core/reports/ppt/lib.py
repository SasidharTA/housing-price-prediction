from tigerml.core.utils import time_now_readable

from .layouts import layouts
from .Report import PptReport, Section
from .Slide import Slide


def add_contents_to_group(contents, group):
    for content in contents:
        if isinstance(contents, dict):
            content_name = content
            content = contents[content_name]
        else:
            content_name = None
        if isinstance(content, list) or isinstance(content, dict):
            group.insert(create_ppt_section(content, content_name))
        else:
            group.insert(create_ppt_slide(content, content_name))
    return group


def create_ppt_report(contents, name="", path="", tiger_template=False):
    if not name:
        name = "report_at_{}".format(time_now_readable())
    report = PptReport(name)
    add_contents_to_group(contents, report)
    report.save(path=path, tiger_template=tiger_template)


def create_ppt_section(contents, name=""):
    section = Section(name)
    add_contents_to_group(contents, section)
    return section


def create_ppt_slide(content, name=""):
    slide = Slide(title=name, layout=layouts.title_and_content)
    slide.add_content(content)
    return slide
