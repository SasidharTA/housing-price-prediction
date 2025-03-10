import matplotlib.colors

more_is_good_2colors = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["white", "green"]
)
less_is_good_2colors = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["green", "white"]
)
more_is_bad_2colors = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["white", "red"]
)
less_is_bad_2colors = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["red", "white"]
)

more_is_good_3colors = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["red", "white", "green"]
)
less_is_good_3colors = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["green", "white", "red"]
)
