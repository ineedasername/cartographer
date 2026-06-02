"""LanceDB helpers for the token vocabulary index."""


def open_lance(path='tokens.lance'):
    """Open a LanceDB connection to the token index.

    Args:
        path: path to the LanceDB directory

    Returns:
        lancedb.DBConnection
    """
    import lancedb
    return lancedb.connect(path)
