import os
import pandas as pd

# import xlrd


def get_files_in_dir(
    dir_path, pattern=None, recursive=False, extensions=("xls", "xlsx", "csv", "tsv")
):
    if recursive:
        # Generate list of file by recursive search inside directory and
        # its subdirectories
        file_list = pd.Series(
            [
                os.path.join(path, names)
                for path, subdirs, files in os.walk(dir_path)
                for names in files
            ]
        )
    else:
        # Take all files' name in the list of the directory
        file_list = pd.Series(os.listdir(dir_path))
        # Join file name with their path
        file_list = dir_path + "\\" + file_list
    file_list = file_list[file_list.str.endswith(extensions)].reset_index(drop=True)
    if pattern:
        # Removing file path from list and then matching with the regex
        # pattern provided and keeping only matched file name
        file_list = file_list[
            file_list.str.replace(r".*\\", "", regex=True).str.match(pattern)
        ].reset_index(drop=True)
    return file_list


def read_files_in_dir(
    dir_path, pattern=None, recursive=False, extensions=("xls", "xlsx", "csv", "tsv")
):
    """
    Function to rowbind all the files(csv/tsv/xlsx/xls) in the directory. Takes directory path as input. Returns dataframe as output.

    Parameters
    ----------
        dir_path  : path of directory
        pattern   : None(default) regex pattern to match the filename (not considerding path)
        recursive : False(default); if True, recursively find all the matching files in subdirectories
    """
    try:
        file_list = get_files_in_dir(dir_path, pattern, recursive, extensions)

        df = pd.DataFrame()
        for file in file_list[file_list.str.endswith("csv")]:
            # Read csv file and append iteratively
            # FIXME: Parser Error exception
            temp = pd.read_csv(file, engine="python")
            temp["__source"] = file
            df = pd.concat([df, temp], sort=False)
        for file in file_list[file_list.str.endswith("tsv")]:
            # Read tsv file and append iteratively
            temp = pd.read_csv(file, delimiter="\t")
            temp["__source"] = file
            df = pd.concat([df, temp], sort=False)
        for file in file_list[file_list.str.endswith(("xls", "xlsx"))]:
            # Read xlsx/xls file iteratively
            try:
                # TODO: sheet_names unused
                temp_excel = pd.read_excel(file, sheet_name=None)
            except MemoryError:
                print("File size too big")
            except PermissionError:
                print("Please close the file to get read permissions.")
            # except xlrd.XLRDError:
            #     print(file, 'is unsupported or corrupt file')
            else:
                for sheet in temp_excel.items():
                    # Read every sheet from file and append to master dataframe
                    temp = sheet[1]
                    temp["__source"] = file + sheet[0]
                    df = pd.concat([df, temp], sort=False)
        return df
    except Exception as e:
        print(e.args[1])


def check_or_create_path(file_path):
    # import pdb
    # pdb.set_trace()
    if file_path:
        if file_path[-1] == "/":
            file_path = file_path[:-1]
    else:
        file_path = "."
    folders_to_create = []
    path_to_check = file_path
    while not os.path.exists(path_to_check):
        if "/" in path_to_check:
            path_to_check, folder_name = path_to_check.rsplit("/", maxsplit=1)
        else:
            folder_name = path_to_check
            path_to_check = "."
        if folder_name:
            folders_to_create = [folder_name] + folders_to_create
    if folders_to_create:
        current_path = path_to_check
        for folder_name in folders_to_create:
            create_path = current_path + "/" + folder_name
            try:
                os.mkdir(create_path)
            except FileExistsError:
                pass
            current_path = create_path
    return True
