from tigerml.core.utils.plots import get_plot_dict_for_dynamicmap  # noqa
from tigerml.core.utils.plots import holomap_input_from_dynamicmap  # noqa


class Chart:
    """Chart class."""

    def __new__(cls, input, name=""):
        """Returns parent class."""
        if (
            input.__class__.__name__ == "DynamicMap"
            and input.kdims
            and input.collate().__class__.__name__ != "DynamicMap"
        ):
            # When Dynamic map has a layout inside - converting it to HoloMap does not work. Should be shown as separate plots
            plot_dict = get_plot_dict_for_dynamicmap(input)
            from ..html import create_html_report

            report = create_html_report({"plots": plot_dict}, save=False)
            return report.dashboards[0].components[0].content
        return super().__new__(cls)

    def __init__(self, input, name=""):
        self.name = name
        if str(input.__module__).startswith("bokeh"):
            self.plot = input
        elif str(input.__module__).startswith("holoviews") or str(
            input.__module__
        ).startswith("hvplot"):
            if input.__class__.__name__ not in ["DynamicMap", "HoloMap"]:
                from tigerml.core.plots import autosize_plot, get_bokeh_plot

                input = autosize_plot(input)
                try:
                    self.plot = get_bokeh_plot(input)
                except Exception as e:
                    self.plot = "Could not generate. Error - {}".format(e)
            else:
                if (
                    input.__class__.__name__ == "DynamicMap" and input.kdims
                ):  # If dimensioned DynamicMap, convert to holomap
                    plot_dict = holomap_input_from_dynamicmap(input)
                    import holoviews as hv

                    input = hv.HoloMap(
                        plot_dict, kdims=[dim.name for dim in input.kdims]
                    )
                import holoviews as hv

                bokeh_renderer = hv.renderer("bokeh")
                plot_html = bokeh_renderer.static_html(input)
                head_tag = plot_html[
                    plot_html.index("<head>") + 6 : plot_html.index("</head>")
                ]
                script_tags = []
                while "<script" in head_tag:
                    tag = head_tag[
                        head_tag.index("<script") : head_tag.index("</script>") + 9
                    ]
                    script_tags += [tag]
                    head_tag = head_tag.replace(tag, "")
                script = "".join([x for x in script_tags if "<script src=" not in x])
                # from bokeh.embed import components
                # script, html = components(model)
                html_body = plot_html[plot_html.index("</head>") + 7 :]
                self.plot = (
                    script
                    + html_body[
                        html_body.index("<body>") + 6 : html_body.index("</body>")
                    ]
                )
        elif str(input.__module__).startswith("plotly"):
            self.plot = input
        else:
            raise Exception("Unsupported format for Chart - {}".format(type(input)))

    @property
    def chart_module(self):
        """Returns chart module."""
        return (
            "str" if isinstance(self.plot, str) else self.plot.__module__.split(".")[0]
        )

    @property
    def height(self):
        """Returns height."""
        return self.plot.height

    @property
    def width(self):
        """Returns width."""
        return self.plot.width
