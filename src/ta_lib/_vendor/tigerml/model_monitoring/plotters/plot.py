import holoviews as hv
import logging
import pandas as pd
from bokeh.models.formatters import NumeralTickFormatter
from hvplot import hvPlot
from tigerml.core.dataframe.dataframe import measure_time

_LOGGER = logging.getLogger(__name__)

hv.extension("bokeh", "matplotlib")


def get_heatmap(
    df,
    x_axis,
    y_axis,
    heatmap_value="psi",
    heatmap_title="",
):
    hvplot = hvPlot(df).heatmap(
        x=x_axis, y=y_axis, C=heatmap_value, title=heatmap_title
    )
    _LOGGER.info("Plotted heat map for the various metrics of df")
    return hvplot


def get_barchart(df, y_axis="variable", value_on="psi", by_dummy="", sort=True):

    if sort:
        df = df.sort_values(by=[value_on], ascending=True)

    if by_dummy != "":
        plot = hvPlot(df).barh(
            y_axis,
            value_on,
            by=by_dummy,
            title="",
            stacked=False,
            legend="bottom_right",
        )
    else:
        plot = hvPlot(df).barh(
            y_axis,
            value_on,
            title="",
            stacked=False,
            legend="bottom_right",
        )
    _LOGGER.info("Plotted bar chart on various metrics of df")
    return plot


def create_psi_plot(
    bins_or_categories: pd.Series, perc_base: pd.Series, perc_curr: pd.Series
):
    formatter = NumeralTickFormatter(format="0%")
    data = pd.DataFrame(
        {
            "bins_or_categories": bins_or_categories,
            "perc_base": perc_base,
            "perc_curr": perc_curr,
        },
    )
    data["bins_or_categories"] = data["bins_or_categories"].astype("str")
    plot = (
        hvPlot(data)
        .bar(
            x="bins_or_categories",
            y=["perc_base", "perc_curr"],
            title="Percenatge Distribution of Bins/Categories",
            xlabel="Bins Or Categories Left",
            ylabel="Percentage Presence",
        )
        .opts(multi_level=False, yformatter=formatter)
    )

    return plot
