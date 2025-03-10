import numpy as np
import pandas as pd


def store_df_in_db(
    df: pd.DataFrame, db_conn, table_name: str, additional_columns: dict = {}
):
    """
    This function stores a pandas DataFrame to a Table.

    You can also give additional_columns dict. This dict should be of the format {"<<col_name:str>>":<<col_value>>}.
    Some good additional_columns to give are the model's MetaData info genrated by using MetaData().setup_metadata() func.

    Parameters
    ----------
    df: pd.DataFrame
        df which you want to store in DB.
    db_conn: SQLAlchemy Engine / SQLAlchemy DB Connection
        Provide the db_conn so that this function can interface with the database for storing.
    table_name: str
        The name of the DB table where you want to store the df.
        If table exists in DB, the df is appended to the table.
    additional_columns: dict
        This dict should be of the format {"<<col_name: str>>":<<col_value>>}.
        Some good additional_columns to give are the model's MetaData info genrated by using MetaData().setup_metadata() func.
    """
    if isinstance(df, pd.DataFrame):
        df_copy = df.copy()
        if additional_columns != {}:
            for k, v in additional_columns.items():
                df_copy[k] = v

        # This is a pandas inbuilt method to match the column with best possible dtype and convert it to that dtype
        df_copy = df_copy.convert_dtypes()

        # Convert Object columns to str
        obj_cols = df_copy.select_dtypes(
            exclude=[np.datetime64, "string", "int64", "int", "float", "float64"]
        ).columns.values.tolist()
        df_copy[obj_cols] = df_copy[obj_cols].applymap(lambda x: str(x))

        # Repeat again after obj col conversion.
        # This is a pandas inbuilt method to match the column with best possible dtype and convert it to that dtype
        df_copy = df_copy.convert_dtypes()

        df_copy.to_sql(
            name=table_name,
            con=db_conn,
            if_exists="append",
            index=False,
        )
        return df_copy
    else:
        return df
