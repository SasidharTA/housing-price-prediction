import featuretools as ft
import featuretools_tsfresh_primitives as tsfresh_primitives
import inspect
from tigerml.core.utils import flatten_list

tsfresh_classes = [
    x[1] for x in inspect.getmembers(tsfresh_primitives, inspect.isclass)
]
tsfresh_prims = [
    prim
    for prim in tsfresh_classes
    if [param.name for param in inspect.signature(prim.__init__).parameters.values()]
    == ["self"]
]
tsfresh_classes = [
    prim
    for prim in tsfresh_classes
    if [param.name for param in inspect.signature(prim.__init__).parameters.values()]
    != ["self"]
]


def import_and_load(classes):
    objects = list()
    for cl in classes:
        exec(f"from {cl.__module__} import {cl.__name__}")
        # exec(f'{cl.__name__.lower()} = {cl.__name__}()')
        objects.append(eval(f"{cl.__name__}()"))
    return objects


common_primitves = [
    "mean",
    "median",
    "maximum",
    "minimum",
    "standarddeviation",
    "variance",
]
tsfresh_primitives = [
    x for x in tsfresh_prims if x.__name__.lower() not in common_primitves
]

all_primitives = (
    list(ft.primitives.get_aggregation_primitives().values()) + tsfresh_prims
)
general_primitives = [
    prim
    for prim in all_primitives
    if any(
        [
            "variable" == input_type.__name__.lower()
            or "index" == input_type.__name__.lower()
            for input_type in flatten_list(prim.input_types)
        ]
    )
]


def _get_filtered(types, array=all_primitives):
    filtered_primitives = [
        prim
        for prim in array
        if any(
            [
                any([type in input_type.__name__.lower() for type in types])
                for input_type in flatten_list(prim.input_types)
            ]
        )
    ]
    return filtered_primitives


datetime_aggregators = import_and_load(
    _get_filtered(["datetime"]) + general_primitives
) + _get_filtered(["datetime"], tsfresh_classes)
numeric_aggregators = import_and_load(
    _get_filtered(["numeric"]) + general_primitives
) + _get_filtered(["numeric"], tsfresh_classes)
bool_aggregators = import_and_load(
    _get_filtered(["bool"]) + general_primitives
) + _get_filtered(["bool"], tsfresh_classes)
categorical_aggregators = import_and_load(
    _get_filtered(["categorical", "discrete"]) + general_primitives
) + _get_filtered(["categorical", "discrete"], tsfresh_classes)
