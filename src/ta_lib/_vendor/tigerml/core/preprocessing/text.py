def string_cleaning(
    string_series,
    special_chars_to_keep="._,$&",
    remove_chars_in_braces=True,
    strip=True,
    lower=False,
):
    """Function to clean strings. Removes special character, characters between square/round braces, multiple spaces, leading/tailing spaces.

    Parameters
    ----------
        string_series          pandas string Series
        special_chars_to_keep  string having special characters that have to
                                be kept
        remove_chars_in_braces Logical if to keep strings in braces.
                               e.g: "Packages (8oz)" will be "Packages"
        strip                  True(default), if False it will not remove
                               extra/leading/tailing spaces
        lower                  False(default), if True it will convert all
                               characters to lowercase

    Returns
    -------
    Pandas series.
    """
    # FIXME: Handle Key Error Runtime Exception
    try:
        if lower:
            # Convert names to lowercase
            string_series = string_series.str.lower()
        if remove_chars_in_braces:
            # Remove characters between square and round braces
            string_series = string_series.str.replace(r"\(.*\)|\[.*\]", "", regex=True)
        else:
            # Add braces to special character list, so that they will not be
            # removed further
            special_chars_to_keep = special_chars_to_keep + "()[]"
        if special_chars_to_keep:
            # Keep only alphanumeric character and some special
            # characters(.,_-&)
            reg_str = "[^\\w" + "\\".join(list(special_chars_to_keep)) + " ]"
            string_series = string_series.str.replace(reg_str, "", regex=True)
        if strip:
            # Remove multiple spaces
            string_series = string_series.str.replace(r"\s+", " ", regex=True)
            # Remove leading and trailing spaces
            string_series = string_series.str.strip()
        return string_series
    except AttributeError:
        print("Variable datatype is not string")
    except KeyError:
        print("Variable name mismatch")


def string_diff(A, B):
    """Function to give difference of two strings (A-B).

    Takes two strings as input.
    Returns string which is difference of two.
    """
    try:
        # Removes words from string A which are present in string B
        final_string = A.replace(B, "")
        return final_string
    except TypeError as e:
        print(e)
    except AttributeError as e:
        print(e)
