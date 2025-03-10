import holoviews as hv
import numpy as np


def is_datashaded(plot):
    callback = None
    if "_callback_param_value" in plot.__dict__:
        callback = plot.__dict__["_callback_param_value"]
    if (
        callback
        and hasattr(callback, "operation")
        and callback.operation.__name__ == "dynspread"
    ):
        return True
    return False


def holomap_input_from_dynamicmap(plot):
    plot_dict = {}
    import itertools

    combinations = sorted(
        list(
            set(
                [
                    x
                    for x in itertools.product(*[dim.values for dim in plot.kdims])
                    if len(set(x)) == len(plot.kdims)
                ]
            )
        )
    )
    # groupers = [dim.name for dim in plot.kdims]
    sample_data = None
    for comb in combinations:
        comb_plot = plot.get(comb)
        if comb_plot is None:
            if sample_data is None:
                sample_data = plot[[x for x in plot.data.keys()][1]].data
                # x = plot[[x for x in plot.data.keys()][1]].kdims[0].name
                y = plot[[x for x in plot.data.keys()][1]].vdims[0].name
                sample_data[y] = np.NaN
            comb_plot = hv.Curve(sample_data)
        plot_dict[comb] = comb_plot
    return plot_dict


def get_plot_dict_for_dynamicmap(plot):
    plot_dict = {}
    import itertools

    combinations = sorted(
        list(
            set(
                [
                    tuple(set(x))
                    for x in itertools.product(*[dim.values for dim in plot.kdims])
                    if len(set(x)) == len(plot.kdims)
                ]
            )
        )
    )
    for comb in combinations:
        dict = plot_dict
        for ind, key in enumerate(comb):
            if ind + 1 != len(comb):  # not the last key in combination
                if key not in dict:
                    dict[key] = {}
                dict = dict[key]
            else:
                dict[key] = plot.get(comb)
    return plot_dict
