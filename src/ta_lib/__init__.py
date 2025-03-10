# Add vendored libs to the path
import os
import sys

from .version import version

__version__ = version


_HERE = os.path.abspath(os.path.dirname(__file__))
_VENDOR_PATH = os.path.join(_HERE, "_vendor")
if os.path.exists(_VENDOR_PATH):
    sys.path.append(_VENDOR_PATH)
