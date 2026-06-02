"""SQLite database helpers for rank-displacement scans and benchmarks."""

import os
import sqlite3


def connect_db(path=None, default_name='rank_scan.db'):
    """Open a SQLite database connection.

    Args:
        path: explicit path, or None to use default_name in cwd
        default_name: filename to use if path is None

    Returns:
        sqlite3.Connection
    """
    if path is None:
        path = os.path.join(os.getcwd(), default_name)
    return sqlite3.connect(path)
