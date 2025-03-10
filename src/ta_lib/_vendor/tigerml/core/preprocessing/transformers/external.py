import featuretools as ft
from tigerml.core.utils import flatten_list


def import_and_load(classes):
    objects = list()
    for cl in classes:
        exec(f"from {cl.__module__} import {cl.__name__}")
        # exec(f'{cl.__name__.lower()} = {cl.__name__}()')
        objects.append(eval(f"{cl.__name__}()"))
    return objects


all_primitives = list(ft.primitives.get_transform_primitives().values())
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


def _get_filtered(types):
    filtered_primitives = [
        prim
        for prim in all_primitives
        if any(
            [
                any([type in input_type.__name__.lower() for type in types])
                for input_type in flatten_list(prim.input_types)
            ]
        )
    ]
    return filtered_primitives


datetime_transformers = import_and_load(
    _get_filtered(["datetime"]) + general_primitives
)
numeric_transformers = import_and_load(_get_filtered(["numeric"]) + general_primitives)
bool_transformers = import_and_load(_get_filtered(["bool"]) + general_primitives)
categorical_transformers = import_and_load(
    _get_filtered(["categorical", "discrete"]) + general_primitives
)
text_transformers = import_and_load(_get_filtered(["text"]) + general_primitives)
location_transformers = import_and_load(_get_filtered(["latlong"]) + general_primitives)
