from ...contents import Chart


class HTMLChart(Chart):
    """Html chart class."""

    @classmethod
    def from_parent(cls, parent):
        """Returns parent class for Html chart class."""
        return cls(parent.plot)

    def to_html(self, resource_path=""):
        """Converts to html for Html chart class."""
        # html_str = ''
        # if self.name:
        # 	html_str += title_html(prettify_slug(self.name))
        if isinstance(self.plot, str):
            plot_html = self.plot
        else:
            if self.chart_module == "bokeh":
                from bokeh.embed import components

                script, div = components(self.plot)
                script += """<script src="https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.0.1.min.js" crossorigin="anonymous"></script>"""
                plot_html = script + div
            elif self.chart_module == "plotly":
                from plotly.io import to_html

                plot_html = to_html(self.plot)
            else:
                plot_html = (
                    "Could not generate. Error - Given plot "
                    "input is neither bokeh nor plotly."
                )
        return (
            '<div class="content chart_content"><div class='
            '"content_inner">{}</div></div>'.format(plot_html)
        )

    def save(self, name=""):
        """Saves for Html chart class."""
        if not name:
            name = self.name
        name = name + (".html" if name[-5:] != ".html" else "")
        if self.chart_module == "bokeh":
            from bokeh.io import save
            from bokeh.resources import CDN, INLINE

            save(self.plot, name, resources=CDN, title=name.rsplit(".", maxsplit=1)[0])
        elif self.chart_module == "plotly":
            from plotly.io import write_html

            write_html(self.plot, name)
