from ..table_styles import styles


def process_col_style(style, data_col, styler=None):
    if styles.width in style:
        width = style[styles.width]
        del style[styles.width]
    else:
        width = None
    width_val = width
    if callable(width):
        width_val = width(data_col, styler)
    for key in [x for x in style.keys() if callable(style[x])]:
        style[key] = style[key](data_col)
    return width_val, style


def process_row_style(style, data_row, styler=None):
    if styles.height in style:
        height = style[styles.height]
        del style[styles.height]
    else:
        height = None
    height_val = height
    if callable(height):
        height_val = height(data_row, styler)
    for key in [x for x in style.keys() if callable(style[x])]:
        style[key] = style[key](data_row)
    return height_val, style


def create_index_format(index_rules, index_values):
    combined_style = {}
    combined_options = {}
    for each_rule in index_rules:
        combined_style.update(each_rule["style"])
        if each_rule["options"] is not None:
            combined_options.update(each_rule["options"])
    width_val, style = process_col_style(combined_style, index_values)
    return width_val, style, combined_options


def create_header_format(index_rules, index_values):
    combined_style = {}
    combined_options = {}
    for each_rule in index_rules:
        combined_style.update(each_rule["style"])
        if each_rule["options"] is not None:
            combined_options.update(each_rule["options"])
    width_val, style = process_row_style(combined_style, index_values)
    return width_val, style, combined_options


def write_title(text, worksheet, workbook, top_row, left_col, width):
    from tigerml.core.utils import prettify_slug

    from .contents import ExcelText

    return ExcelText(
        prettify_slug(text),
        width=width,
        height=1,
        format={"bold": True, "align": "center"},
    ).save(worksheet, workbook, top_row, left_col)
