import numpy as np
from bokeh.models import Legend, LegendItem, LinearAxis, Range1d
from bokeh.models.mappers import CategoricalColorMapper
from bokeh.models.transforms import Dodge
from bokeh.palettes import (
    Inferno256,
    Plasma256,
    Spectral6,
    Spectral11,
    diverging_palette,
    linear_palette,
)
from tigerml.core.utils import compute_if_dask, flatten_list


def get_max_y(element):
    if element.data.__class__.__name__ == "OrderedDict":
        max_ys = [get_max_y(element.data[key]) for key in element.data]
        max_ys = flatten_list(max_ys)
        max_y = max(max_ys)
    else:
        values = element.data[element.vdims[0].name]
        if hasattr(values, "fillna"):
            values = values.fillna(0)
        max_y = compute_if_dask(values.max())
    return max_y


def get_min_y(element):
    if element.data.__class__.__name__ == "OrderedDict":
        min_ys = [get_min_y(element.data[key]) for key in element.data]
        min_ys = flatten_list(min_ys)
        min_y = min(min_ys)
    else:
        values = element.data[element.vdims[0].name]
        if hasattr(values, "fillna"):
            values = values.fillna(0)
        min_y = compute_if_dask(values.min())
    if element.__class__.__name__ == "Bars" and min_y > 0:
        min_y = 0
    return min_y


def get_element_names(element):
    if element.data.__class__.__name__ == "OrderedDict":
        names = [get_element_names(element.data[key]) for key in element.data]
        names = flatten_list(names)
    else:
        names = [element.vdims[0].name]
    return names


def get_glyph_name(glyph):
    if glyph.glyph.__class__.__name__ in ["VBar", "Quad"]:
        return glyph.glyph.top
    else:
        return glyph.glyph.y


def get_pretty_glyph_name(glyph):
    if glyph.glyph.__class__.__name__ == "VBar":
        name = glyph.glyph.top
        return (
            name.replace("_left_parenthesis_", "(")
            .replace("left_parenthesis_", "(")
            .replace("_right_parenthesis_", ")")
            .replace("_right_parenthesis", ")")
        )
    elif glyph.glyph.__class__.__name__ == "Quad":
        return "Frequency"
    else:
        return glyph.glyph.y


def finalize_axis(plot, element, side, include_current, keep_0=False):
    # import pdb
    # pdb.set_trace()
    try:
        pl = plot.state
    except KeyError:
        return
    if side == "left":
        axes = [x for x in pl.left if x.__class__.__name__ == "LinearAxis"]
        if not axes:
            return
        axis_name = "default"
        axis_range = pl.y_range
    else:
        axes = [x for x in pl.right if x.__class__.__name__ == "LinearAxis"]
        if not axes:
            return
        axis_name = "twiny"
        axis_range = pl.extra_y_ranges["twiny"]
    max_ys = []
    min_ys = []
    current_name = []
    if include_current:
        max_ys.append(get_max_y(element))
        min_ys.append(get_min_y(element))
        current_name = get_element_names(element)
    glyphs = [glyph for glyph in pl.renderers if glyph.y_range_name == axis_name]
    if [
        glyph for glyph in glyphs if glyph.glyph.__class__.__name__ in ["VBar", "Quad"]
    ] or keep_0:
        min_ys.append(0)
        max_ys.append(0)
    if glyphs or include_current:
        axes[0].axis_label = ", ".join(
            list(set([get_pretty_glyph_name(glyph) for glyph in glyphs] + current_name))
        )
        max_y = (
            max(
                max_ys
                + [
                    np.nan_to_num(glyph.data_source.data[get_glyph_name(glyph)]).max()
                    for glyph in glyphs
                ]
            )
            * 1.01
        )
        min_y = (
            min(
                min_ys
                + [
                    np.nan_to_num(glyph.data_source.data[get_glyph_name(glyph)]).min()
                    for glyph in glyphs
                ]
            )
            * 0.99
        )
        axis_range.start = compute_if_dask(min_y)
        axis_range.end = compute_if_dask(max_y)
    else:
        exec("pl.{} = []".format(side))
        # axes = []
        # axes[0].axis_label = ''


def finalize_axes_right(plot, element):
    # import pdb
    # pdb.set_trace()
    finalize_axis(plot, element, "left", include_current=False)
    finalize_axis(plot, element, "right", include_current=True)


def finalize_axes_left(plot, element):
    # import pdb
    # pdb.set_trace()
    finalize_axis(plot, element, "left", include_current=True)
    finalize_axis(plot, element, "right", include_current=False)


def finalize_axes_right_keep_0(plot, element):
    # import pdb
    # pdb.set_trace()
    finalize_axis(plot, element, "left", include_current=False)
    finalize_axis(plot, element, "right", include_current=True, keep_0=True)


def finalize_axes_left_keep_0(plot, element):
    # import pdb
    # pdb.set_trace()
    finalize_axis(plot, element, "left", include_current=True, keep_0=True)
    finalize_axis(plot, element, "right", include_current=False)


def add_to_primary(plot, element):
    try:
        pl = plot.state
    except KeyError:
        return
    glyph = pl.renderers[-1]
    if element.data.__class__.__name__ == "OrderedDict":
        glyphs = pl.renderers[-len(element.data) :]
    else:
        glyphs = [glyph]
    for glyph in glyphs:
        glyph.y_range_name = "default"


def add_to_secondary(plot, element):
    """A hook to put data on secondary axis."""
    try:
        pl = plot.state
    except KeyError:
        return
    glyph = pl.renderers[-1]
    if element.data.__class__.__name__ == "OrderedDict":
        glyphs = pl.renderers[-len(element.data) :]
    else:
        glyphs = [glyph]
    # create secondary range and axis
    if "twiny" not in [t for t in pl.extra_y_ranges]:
        pl.extra_y_ranges = {"twiny": Range1d(start=0, end=10)}
        pl.add_layout(
            LinearAxis(y_range_name="twiny", axis_label=get_pretty_glyph_name(glyph)),
            "right",
        )
    for glyph in glyphs:
        glyph.y_range_name = "twiny"


def add_to_secondary_dynamicmap(plot, element):
    """A hook to put data on secondary axis."""
    try:
        pl = plot.state
    except KeyError:
        return
    series_name = str(element).split("]   ")[-1][1:-1]
    glyphs = [r for r in pl.renderers if get_glyph_name(r) == series_name]
    if "twiny" not in [t for t in pl.extra_y_ranges]:
        pl.extra_y_ranges = {"twiny": Range1d(start=0, end=10)}
        pl.add_layout(LinearAxis(y_range_name="twiny", axis_label=series_name), "right")
    for glyph in glyphs:
        glyph.y_range_name = "twiny"


def add_to_primary_dynamicmap(plot, element):
    try:
        pl = plot.state
    except KeyError:
        return
    series_name = str(element).split(" ")[-1][1:-1]
    glyphs = [r for r in pl.renderers if get_glyph_name(r) == series_name]
    for glyph in glyphs:
        glyph.y_range_name = "default"


def legend_policy(plot, element):
    plot.state.legend.click_policy = "hide"


def x_as_datetimeaxis(plot, element):
    """A hook to convert x axis as DateTimeAxis."""
    try:
        pl = plot.state
    except KeyError:
        return

    from bokeh.models import DatetimeTicker, DatetimeTickFormatter

    pl.xaxis.ticker = DatetimeTicker()
    pl.xaxis.formatter = DatetimeTickFormatter()


def y_as_datetimeaxis(plot, element, side):
    try:
        pl = plot.state
    except KeyError:
        return
    from bokeh.models import DatetimeTicker, DatetimeTickFormatter

    if (
        isinstance(pl.yaxis, list) and len(pl.yaxis) > 1
    ):  # 2nd condition is for scenario with only secondary axis
        if side == "left":
            pl.yaxis[0].ticker = DatetimeTicker()
            pl.yaxis[0].formatter = DatetimeTickFormatter()
        else:
            pl.yaxis[1].ticker = DatetimeTicker()
            pl.yaxis[1].formatter = DatetimeTickFormatter()
    else:
        pl.yaxis.ticker = DatetimeTicker()
        pl.yaxis.formatter = DatetimeTickFormatter()


def left_y_as_datetimeaxis(plot, element):
    y_as_datetimeaxis(plot, element, side="left")


def right_y_as_datetimeaxis(plot, element):
    y_as_datetimeaxis(plot, element, side="right")


def colorize_boxplot(color_axis_values, plot, element):
    try:
        pl = plot.state
    except KeyError:
        return
    glyph = pl.renderers

    color_axis = color_axis_values[0]
    if len(glyph) == 7:
        # Condition to ensure that the 2 new glyphs are not created multiple times
        # in the case of Dynamicmap (plots with split_by).
        # Typical hv.BoxWhisker will have only
        # 7 glyphs (Circle*1, Segment*4 and VBar*2)

        # matching the actual column name with bokeh's ColumnarDataSource column name
        # Eg: 'IMDB.Rating' becomes 'IMDB_full_stop_Rating'
        # and 'Week Day' becomes 'Week_Day'
        cds_col_names = list(pl.renderers[5].data_source.data.keys())
        color_axis_li = "_".join(e for e in color_axis if e.isalnum()).split("_")
        # color_axis = ''.join([name if all(val in name for val in color_axis_li)
        #                            else '' for name in cds_col_names])
        # 'Slot' matches with both 'Slot' and 'Channel Slot'
        match_list = [
            name for name in cds_col_names if all(val in name for val in color_axis_li)
        ]
        if len(match_list) == 1:
            color_axis = match_list[0]
        else:
            len_diff_list = [abs(len(color_axis_li) - len(name)) for name in match_list]
            color_axis = match_list[len_diff_list.index(min(len_diff_list))]

        # palette creation based on the no. of factor levels.
        # (Plasma256 colors are not visually very distinct,
        # but used when levels are more than 12)
        # level_1_factors = [str(x) for x in np.unique(
        #     glyph[5].data_source.data[color_axis]
        # ) if x not in [np.NaN, None]]
        level_1_factors = color_axis_values[
            1
        ]  # levels are to be passed from outside as the ColumnarDataSource
        # will not have the exhaustive list of factors in case of a Dynamicmap
        if len(level_1_factors) < 7:
            selected_palette = linear_palette(Spectral6, len(level_1_factors))
        elif len(level_1_factors) < 12:
            selected_palette = diverging_palette(
                Plasma256, Spectral11, len(level_1_factors)
            )
        else:
            selected_palette = diverging_palette(
                Plasma256, Inferno256, len(level_1_factors)
            )

        color_mapper = CategoricalColorMapper(
            factors=level_1_factors, palette=selected_palette, start=1, end=2
        )

        # Adding new glyph(s) on top of existing VBar(s)
        # (boxes) to render them with distinct colors
        # glyph[5] -> top half of every box
        # glyph[6] -> bottom half of every box
        pl.vbar(
            x=glyph[5].glyph.x,
            top=glyph[5].glyph.top,
            bottom=glyph[5].glyph.bottom,
            width=glyph[5].glyph.width,
            source=glyph[5].data_source,
            color=dict(field=glyph[5].glyph.x, transform=color_mapper),
            line_color="black",
            fill_alpha=0.8,
        )
        pl.vbar(
            x=glyph[6].glyph.x,
            top=glyph[6].glyph.top,
            bottom=glyph[6].glyph.bottom,
            width=glyph[6].glyph.width,
            source=glyph[6].data_source,
            color=dict(field=glyph[6].glyph.x, transform=color_mapper),
            line_color="black",
            fill_alpha=0.8,
            legend_field=color_axis,
        )
        # 'legend_group' (Python based) also functions similar to 'legend_field' (JavaScript based),
        # but the former should not be used in this case as it results in some color mismatch.
        # (Refer: https://docs.bokeh.org/en/latest/docs/user_guide/annotations.html#automatic-grouping-python)

    # References:
    # 1) For dynamic legend addition
    # https://discourse.bokeh.org/t/adding-a-legend-outside-the-plot-area-is-possible-even-with-auto-grouped-indirectly-created-legends/5595
    # 2) For factor_level specific color rendering
    # http://docs.bokeh.org/en/latest/docs/user_guide/categorical.html#nested-categories

    pl.xaxis.major_label_text_font_size = "0px"
    pl.legend.title = color_axis
    pl.legend.orientation = "vertical"
    if "," in pl.xaxis.axis_label:
        pl.xaxis.axis_label = pl.xaxis.axis_label.split(",")[0]
    elif "_levels" in pl.xaxis.axis_label:
        pl.xaxis.axis_label_text_font_size = "0px"
    pl.add_layout(
        pl.legend[0], "right"
    )  # adding the legend as a separate plot element to the right


def colorize_barplot(color_axis_values, plot, element):
    try:
        pl = plot.state
    except KeyError:
        return
    glyph = pl.renderers

    color_axis = color_axis_values[0]
    plot_orientation = color_axis_values[1]

    cds_col_names = list(glyph[0].data_source.data.keys())
    color_axis_li = "_".join(e for e in color_axis if e.isalnum()).split("_")
    match_list = [
        name for name in cds_col_names if all(val in name for val in color_axis_li)
    ]
    if len(match_list) == 1:
        color_axis = match_list[0]
    else:
        len_diff_list = [abs(len(color_axis_li) - len(name)) for name in match_list]
        color_axis = match_list[len_diff_list.index(min(len_diff_list))]

    if not pl.right or (pl.right and pl.right[0].__class__.__name__ != "Legend"):
        # creating a Legend and linking it with the VBar glyph
        li = LegendItem(label={"field": color_axis}, renderers=[glyph[0]])
        legend = Legend(items=[li])  # location=(0, -30)
        pl.add_layout(legend, "right")
        pl.plot_width += 50
        if plot_orientation == "inverted":
            pl.plot_height += 150

    if plot_orientation == "upright":
        pl.xaxis.major_label_text_font_size = "0px"
        pl.legend.title = color_axis
        pl.legend.orientation = "vertical"
        if "," in pl.xaxis.axis_label:
            pl.xaxis.axis_label = pl.xaxis.axis_label.split(",")[0]
        elif "_levels" in pl.xaxis.axis_label:
            pl.xaxis.axis_label_text_font_size = "0px"

    else:
        pl.yaxis.major_label_text_font_size = "0px"
        pl.legend.title = color_axis
        pl.legend.orientation = "vertical"
        if "," in pl.yaxis.axis_label:
            pl.yaxis.axis_label = pl.yaxis.axis_label.split(",")[0]
        elif "_levels" in pl.yaxis.axis_label:
            pl.yaxis.axis_label_text_font_size = "0px"


def dodge_barplot(x_axis_values, plot, element):
    try:
        pl = plot.state
    except KeyError:
        return
    renderers = pl.renderers

    axis_field = x_axis_values[0]
    plot_orientation = x_axis_values[1]

    if plot_orientation == "upright":
        glyphs_to_modify = []
        for renderer in renderers:
            glyph = renderer.glyph
            glyph_x_field = glyph.x["field"] if isinstance(glyph.x, dict) else glyph.x
            if "VBar" in str(glyph.__class__) and glyph_x_field == axis_field:
                glyphs_to_modify += [glyph]
        pl.x_range.factor_padding = 1
        pl.x_range.range_padding = 0.1
        dodge_pos = (-1 * 0.5 * len(glyphs_to_modify) * 0.25) + (0.25 * 0.5)
        for glyph in glyphs_to_modify:
            glyph.x = {
                "field": axis_field,
                "transform": Dodge(value=dodge_pos, range=pl.x_range),
            }
            glyph.width = 0.2
            dodge_pos += 0.25

    elif plot_orientation == "inverted":
        glyphs_to_modify = []
        for renderer in renderers:
            glyph = renderer.glyph
            glyph_y_field = glyph.y["field"] if isinstance(glyph.y, dict) else glyph.y
            if "HBar" in str(glyph.__class__) and glyph_y_field == axis_field:
                glyphs_to_modify += [glyph]
        pl.y_range.factor_padding = 1
        pl.y_range.range_padding = 0.1
        dodge_pos = (-1 * 0.5 * len(glyphs_to_modify) * 0.25) + 0.25 * 0.5
        for glyph in glyphs_to_modify:
            glyph.y = {
                "field": axis_field,
                "transform": Dodge(value=dodge_pos, range=pl.y_range),
            }
            glyph.height = 0.2
            dodge_pos += 0.25


def fix_overlay_legends(plot, element):
    try:
        pl = plot.state
    except KeyError:
        return

    if pl.legend:
        new_legends = []
        new_legends_dict = {}
        if len(pl.legend.items) > 1:
            for legend_item in pl.legend.items:
                if len(legend_item.renderers) > 1:
                    for renderer in legend_item.renderers:
                        if not legend_item.renderers.index(renderer) == 0:
                            glyph = renderer.glyph
                            glyph_type = glyph.__class__.__name__
                            if glyph_type in list(new_legends_dict.keys()):
                                new_legends_dict[glyph_type] += [LegendItem(label=legend_item.label, renderers=[renderer])]  # noqa
                            else:
                                new_legends_dict[glyph_type] = []
                                new_legends_dict[glyph_type] += [LegendItem(label=legend_item.label, renderers=[renderer])]  # noqa
                legend_item.renderers = [legend_item.renderers[0]]

        for glyph_type in list(new_legends_dict.keys()):
            new_legends += new_legends_dict[glyph_type]
        pl.legend.items += new_legends
