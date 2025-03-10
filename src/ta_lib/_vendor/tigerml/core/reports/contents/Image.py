import io
from datetime import datetime  # noqa
from tigerml.core.utils import create_safe_filename  # noqa


class Image:
    """Image class."""

    def __init__(self, input, name="", format="png"):
        if isinstance(input, str):
            file_path = input
            if not name:
                if "/" in file_path:
                    name = file_path.rsplit("/", maxsplit=1)[-1].split(".")[0]
                else:
                    name = file_path.split(".")[0]
            from PIL import Image

            self.image_data = Image.open(file_path)
            # self._width, self._height = self.image_data.size
        elif str(input.__module__).startswith("matplotlib"):
            import matplotlib.pyplot as plt

            if isinstance(input, plt.Figure):
                mpl_fig = input
            elif input.__name__ == "matplotlib.pyplot":
                mpl_fig = input.gcf()
            else:
                raise Exception("Unsupported matplotlib object for Image creation")
            if not name and mpl_fig._suptitle:
                name = mpl_fig._suptitle.get_text()
            if format == "svg":
                from io import StringIO

                image_data = StringIO()
                try:
                    mpl_fig.savefig(image_data, format=format)
                    self.image_data = "<svg" + image_data.getvalue().split("<svg")[1]
                except Exception as e:
                    msg = "COULD NOT SAVE PLOT - {}. ERROR - {}".format(name, e)
                    print(msg)
                    self.image_data = msg
            else:
                image_data = io.BytesIO()
                from PIL import Image

                try:
                    mpl_fig.savefig(image_data, format=format)
                    self.image_data = Image.open(image_data)
                except Exception as e:
                    print("COULD NOT SAVE PLOT - {}. ERROR - {}".format(name, e))
                    self.image_data = Image.new(mode="RGB", size=(1, 1))
        elif str(input.__module__).startswith("bokeh"):
            from bokeh.plotting import Figure

            if isinstance(input, Figure):
                from bokeh.io.export import get_screenshot_as_png

                self.image_data = get_screenshot_as_png(input)
            else:
                raise Exception("Unsupported bokeh input for Image creation")
        elif str(input.__module__).startswith("plotly"):
            image_data = io.BytesIO(input.to_image(format="png"))
            from PIL import Image

            self.image_data = Image.open(image_data)
        elif str(input.__module__).startswith("holoviews") or str(
            input.__module__
        ).startswith("hvplot"):
            from tigerml.core.plots import (
                autosize_plot,
                get_bokeh_plot,
                get_mpl_plot,
            )

            input = autosize_plot(input)
            bokeh_plot = get_bokeh_plot(input)
            # mpl_fig = get_mpl_plot(input)
            # if not name and mpl_fig._suptitle:
            # 	name = mpl_fig._suptitle.get_text()
            # image_data = io.BytesIO()
            from PIL import Image

            try:
                # mpl_fig.savefig(image_data, format='png')
                # self.image_data = Image.open(image_data)
                from bokeh.io.export import get_screenshot_as_png
                from selenium import webdriver
                from webdriver_manager.chrome import (  # pip install webdriver-manager
                    ChromeDriverManager,
                )

                option = webdriver.ChromeOptions()
                option.add_argument("headless")
                driver = webdriver.Chrome(ChromeDriverManager().install(), options=option)  # noqa
                self.image_data = get_screenshot_as_png(bokeh_plot, driver=driver)
            except Exception as e:
                print("COULD NOT SAVE PLOT - {}. ERROR - {}".format(name, e))
                self.image_data = Image.new(mode="RGB", size=(1, 1))
        elif str(input.__module__).startswith("PIL."):
            self.image_data = input
        else:
            raise Exception(
                "Unsupported input for Image creation - {}".format(type(input))
            )
        if format != "svg":
            self._width, self._height = self.image_data.size
        self.name = create_safe_filename(name)

    @property
    def height(self):
        """Returns height."""
        return (self._height / 20) + (1 if self.name else 0)

    @property
    def width(self):
        """Returns width."""
        return self._width / 70

    def save(self, name=""):
        """Saves for Image class."""
        # import pdb
        # pdb.set_trace()
        default_extension = ".png"
        if not name:
            name = self.name
        name = name + (default_extension if name[:-4] != default_extension else "")
        self.image_data.save(name)
        return name
