import os
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine


def get_db_engine(connection_string):
    """
    This function returns a SQLAlchemy DB Engine, given a connection string.

    Parameters
    ----------
    connection_string: str
        A SQLAlchemy style db connection string. See https://docs.sqlalchemy.org/en/14/core/engines.html for more details.

    Returns
    -------
    engine: MockConnection
        SQLAlchemy DB engine which can be used to execute CRUD operations on DB
    """
    engine = create_engine(connection_string)
    return engine


def get_db_conn_pool(connection_string, pool_size):
    """
    This function returns a DB Connection Pool.

    Using DB Connection pool, helps reduce load on DB.

    Parameters
    ----------
    connection_string: str
        A SQLAlchemy style db connection string. See https://docs.sqlalchemy.org/en/14/core/engines.html for more details.
    pool_size: int
        The size of connection pool

    Returns
    -------
    conn_pool: Connection
        SQLAlchemy DB connection pool with specified pool_size.
    """
    engine = create_engine(connection_string, pool_size=pool_size)
    conn_pool = engine.connect()
    return conn_pool
