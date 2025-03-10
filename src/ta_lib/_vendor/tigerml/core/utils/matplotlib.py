import matplotlib.backends.backend_pdf
import pandas as pd
import plotly.offline as po
import plotly.tools as tls


def save_figures_as_html(figures, file_path):
    """Saves figures as html.

    Saves a collection of matplotlib Figures into a HTML with
    plotly graphs at the file_path location if the path exists.

    Parameters
    ----------
    figures: python list or pandas Series or matplotlib Figure

    file_path: str

    Returns
    -------
    boolean: returns True if the function is successful, False otherwise
    """
    if type(figures).__name__ in ["list", "Series"]:
        final_fig = tls.make_subplots(rows=len(figures), cols=1)
        final_fig["layout"].update(
            height=500 * len(figures),
            title=file_path.split(".html")[0],
            showlegend=False,
        )
        counter = 1
        for index, fig in enumerate(figures):
            if not fig.get_axes():
                continue
            try:
                plotly_fig = tls.mpl_to_plotly(fig, resize=True, strip_style=True)
            except Exception as e:
                print(
                    "Could not convert to plot at index {} to plotly figure.".format(
                        index
                    ),
                    "Error message: {}".format(e.message),
                )
                continue
            final_fig.add_traces(plotly_fig.data, [counter], [1])
            counter += 1
        po.plot(final_fig, filename=file_path)
    elif (
        type(figures).__module__ == "matplotlib.figure"
        and type(figures).__name__ == "Figure"
    ):
        save_figures_as_html([figures], file_path)
    else:
        print(
            "figures should either be a list or a series or",
            "a matplotlib figure object.",
        )
        return False


def save_figures_as_pdf(figures, file_path):
    """Saves figures as pdf.

    Saves a collection of matplotlib Figures into a PDF at the
    file_path location if the path exists.

    Parameters
    ----------
    figures: python list or pandas Series or matplotlib Figure

    file_path: str

    Returns
    -------
    boolean: returns True if the function is successful, False otherwise
    """
    if type(figures).__name__ in ["list", "Series"]:
        pdf = matplotlib.backends.backend_pdf.PdfPages(file_path)
        for fig in figures:
            if not fig.get_axes():
                continue
            fig.tight_layout()
            pdf.savefig(fig)
        pdf.close()
        return True
    elif (
        type(figures).__module__ == "matplotlib.figure"
        and type(figures).__name__ == "Figure"
    ):
        save_figures_as_pdf([figures], file_path)
    else:
        print(
            "figures should either be a list or a series",
            "or a matplotlib figure object.",
        )
        return False


def save_plot(plot, file_path):
    file_name = file_path.split("/")[-1]
    extension = file_name.split(".")[-1]
    if hasattr(plot, "__name__") and plot.__name__ == "matplotlib.pyplot":
        plots = [plot.figure(num) for num in plot.get_fignums()]
    elif isinstance(plot, dict):
        plots = list(plot.values())
    elif isinstance(plot, pd.Series):
        plots = plot.to_list()
    elif isinstance(plot, list):
        plots = plot
    else:
        raise Exception("Incorrect input to save_plot")
    if extension == "pdf":
        result = save_figures_as_pdf(plots, file_path)
    elif extension == "html":
        result = save_figures_as_html(plots, file_path)
    else:
        print(
            "The file extension in file_path is invalid.",
            "Should be either .pdf or .html",
        )
        return False
    if result:
        print("Saved {} at {}".format(file_name, file_path))
        return True
    else:
        print("Could not save the plot")
        return False
