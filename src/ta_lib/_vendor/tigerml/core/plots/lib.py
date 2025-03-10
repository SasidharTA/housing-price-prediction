import holoviews as hv
from tigerml.core.dataframe.helpers import get_module
from tigerml.core.utils import DictObject

PLOT_TYPES = DictObject(
    {"bokeh": "bokeh", "plotly": "plotly", "matplotlib": "matplotlib"}
)


SAVE_FORMATS = DictObject(
    {
        "html": "html",
        # 'jpg': 'jpg',
        "png": "png",
    }
)


def hvPlot(data):
    module = get_module(data)
    if module == "tigerml":
        module = data.backend
        data = data._data
    if module == "dask":
        import hvplot.dask

        return data.hvplot
    else:
        import hvplot.pandas

        return data.hvplot


def get_plotter(data=None, x=None, y=None):
    import pandas as pd

    # meta_data = {}
    if data is None:
        assert (
            x is not None and y is not None
        ), "Need to pass x and y if data is not passed"
        data_series = pd.DataFrame.from_dict({"x": x, "y": y}).set_index("x")
        # meta_data = {'x': 'x', 'y': 'y'}
    else:
        assert x is None and y is None, "Cannot pass x and y if data is passed"
        if isinstance(data, dict):
            data_df = pd.DataFrame.from_dict(data)
        else:
            data_df = data
        data_series = data_df.set_index("x")
    return hvPlot(data_series)


def get_mpl_plot(hv_plot):
    mpl_renderer = hv.renderer("matplotlib")
    return mpl_renderer.get_plot(hv_plot).state


def vline_tooltip(plot):
    from bokeh.models import HoverTool

    if hasattr(plot, "tools"):
        for tool in [x for x in plot.tools if isinstance(x, HoverTool)]:
            tool.mode = "vline"
    elif hasattr(plot, "children"):
        for child in plot.children:
            vline_tooltip(child)


def set_wheel_zoom(plot):
    from bokeh.models import WheelZoomTool

    if hasattr(plot, "tools") and hasattr(plot, "toolbar"):
        wheel_zoom = [x for x in plot.tools if isinstance(x, WheelZoomTool)][0]
        plot.toolbar.active_scroll = wheel_zoom
    # elif hasattr(plot, 'toolbar'):
    # 	wheel_zoom = [x for x in plot.tools if isinstance(x, WheelZoomTool)][0]
    # 	plot.toolbar.active_scroll = wheel_zoom
    elif hasattr(plot, "children"):
        for child in plot.children:
            if isinstance(child, tuple):
                child = child[0]
            set_wheel_zoom(child)


def bokeh_enhancements(plot, line_plot=False):
    if hasattr(plot, "toolbar_location"):
        plot.toolbar_location = "left"
    set_wheel_zoom(plot)
    if line_plot:
        vline_tooltip(plot)


def get_bokeh_plot(hv_plot):
    line_plot = False
    if hv_plot.__class__.__name__ == "Curve" or (
        hasattr(hv_plot, "children") and "Curve" in hv_plot.children
    ):
        line_plot = True
    bokeh_renderer = hv.renderer("bokeh")
    bokeh_plot = bokeh_renderer.get_plot(hv_plot).state
    bokeh_enhancements(bokeh_plot, line_plot)
    return bokeh_plot


# def get_plotly_plot(hv_plot):
# 	from plotly.graph_objs import Figure
# 	return Figure(plotly_renderer.get_plot(hv_plot).state)


def save_plot(plot, name):
    extension = name.split(".")[-1]
    assert (
        extension in SAVE_FORMATS
    ), "Name should have one of these extensions - {}".format(SAVE_FORMATS.keys())
    if extension in ["jpg", "png"]:
        from tigerml.core.reports.contents import Image

        plot_image = Image(plot)
        plot_image.save(name=name)
    else:
        from tigerml.core.reports.html import HTMLChart

        plot_chart = HTMLChart(plot)
        plot_chart.save(name=name)


def get_plot_as(plot, type=PLOT_TYPES.bokeh):
    assert str(plot.__module__).startswith("hvplot") or str(plot.__module__).startswith(
        "holoviews"
    ), "get_plot_as accepts only hvPlot or holoviews plot. Got - {}".format(
        plot.__module__
    )
    assert (
        type in PLOT_TYPES
    ), "type of output should be one of bokeh, plotly or matplotlib"
    if type == PLOT_TYPES.bokeh:
        return get_bokeh_plot(plot)
    # elif type == PLOT_TYPES.plotly:
    # 	return get_plotly_plot(plot)
    elif type == PLOT_TYPES.matplotlib:
        return get_mpl_plot(plot)


def set_width(plot, width):
    width = int(round(width))
    plot_options = plot.opts.get().__dict__["kwargs"]
    if "width" in plot_options:
        current_width = plot_options["width"]
    else:
        current_width = 0
    width = max(width, current_width)
    return plot.opts(width=width)


def set_height(plot, height):
    height = int(round(height))
    plot_options = plot.opts.get().__dict__["kwargs"]
    if "height" in plot_options:
        current_height = plot_options["height"]
    else:
        current_height = 0
    height = max(height, current_height)
    return plot.opts(height=height)


def autosize_plot(plot):
    import holoviews as hv

    plot_options = plot.opts.get().__dict__["kwargs"]
    if (
        isinstance(plot, hv.Bars)
        or isinstance(plot, hv.BoxWhisker)
        or isinstance(plot, hv.Box)
        or isinstance(plot, hv.Violin)
    ):
        kdim_size = len(set(plot[plot.kdims[0]]))
        # if isinstance(plot, hv.BoxWhisker):
        # 	import pdb
        # 	pdb.set_trace()
        # multiplier = 20 if isinstance(plot, hv.Bars) else 2.5
        min_size = kdim_size * 20
        if "invert_axes" in plot_options and plot_options["invert_axes"]:
            plot = set_height(plot, min_size)
        else:
            if min_size > plot_options["width"]:
                plot.opts(invert_axes=True)
                plot = set_height(plot, min_size)
    elif isinstance(plot, hv.HeatMap):
        import math

        xdim_size = math.sqrt(plot[plot.kdims[0]].shape[0])
        ydim_size = math.sqrt(plot[plot.kdims[1]].shape[0])
        min_xsize = xdim_size * 20
        min_ysize = ydim_size * 20
        plot = set_width(plot, min_xsize)
        plot = set_height(plot, min_ysize)
    return plot
