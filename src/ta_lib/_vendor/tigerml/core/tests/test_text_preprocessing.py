import string
from random import choice

import pandas as pd
from hypothesis import given, settings
from hypothesis.strategies import composite, integers, sampled_from

from tigerml.core.preprocessing.text import string_cleaning


# ---------------------------------------------------------------------------------
# ---------- composite strategy to generate String series ----------------
@composite
def data(draw):
    """Creates dataset."""
    string_options = [
        *[f"St_r %#$^Num{str(e)}" for e in list(range(1, 5))],
        *[choice(string.ascii_uppercase) + " *&^caps" for i in range(1, 10)],
        *[
            choice(string.ascii_uppercase) + "  " + choice(string.ascii_lowercase)
            for i in range(1, 3)
        ],
        *[
            choice(string.ascii_letters) + choice(string.punctuation)
            for i in range(1, 10)
        ],
    ]
    return pd.Series(string_options)


# ---------------------------------------------------------------------------------
# ------------------------------ string_cleaning testing --------------------------
@settings(max_examples=1, deadline=None)
@given(series=data())
def test_string_cleaning(series):

    special_chars_to_keep = "._,$&"
    string_series = string_cleaning(series)

    regex = r"^[^\s]+([\s]{0,1}[A-Za-z0-9" + special_chars_to_keep + "]*)$"
    assert (string_series.str.match(regex)).all()

    string_series = string_cleaning(series, strip=False)
    regex = r"^([A-Za-z0-9\s" + special_chars_to_keep + "]*)$"
    assert (string_series.str.match(regex)).all()

    special_chars_to_keep = "_"
    regex = (
        r"^[a-z0-9"
        + special_chars_to_keep
        + r"]+([\s]{0,1}[a-z0-9"
        + special_chars_to_keep
        + r"]*)$"
    )
    # regex2=r"^[^\s]+([\s]{0,1}[a-z0-9" + special_chars_to_keep + "]*)$"
    string_series = string_cleaning(series, special_chars_to_keep=".", lower=True)
    assert (string_series.str.match(regex)).all()
