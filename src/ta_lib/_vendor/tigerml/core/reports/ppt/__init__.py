try:
    import pptx
except ModuleNotFoundError as e:
    Warning(
        "Please install python-pptx for Powerpoint "
        "reports. - conda install python-pptx"
    )
    raise e

from .lib import create_ppt_report
from .Report import PptReport, Section
from .Slide import Slide, SlideComponent
