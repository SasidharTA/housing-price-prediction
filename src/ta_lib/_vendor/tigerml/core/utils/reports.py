import os


def get_extension_of_path(path):
    _, ext = os.path.splitext(os.path.basename(path))
    return ext


def append_file_to_path(path, file_name):
    if get_extension_of_path(path):
        return path
    else:
        return os.path.join(path, file_name)


def convert_to_tuples(keys_list, report_dict):

    for node_key in keys_list:
        try:
            eval_str = "report_dict[" + "][".join(str(node_key)[1:-1].split(", ")) + "]"  # noqa
            dict_to_combine = eval(eval_str)
            tuple_list = []
            for key_ in dict_to_combine.keys():
                if isinstance(dict_to_combine[key_], list):
                    tuple_list += dict_to_combine[key_]
                else:
                    tuple_list += [dict_to_combine[key_]]
            exec(eval_str + "= tuple(tuple_list)")

        except (KeyError, AttributeError):
            pass
