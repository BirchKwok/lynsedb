import threading


def collection_repr(collection):
    """
    Get the string representation of a collection.

    Parameters:
        collection (Collection): The collection to represent.

    Returns:
        str: The string representation of the collection.
    """
    return (f'{collection.name}CollectionInstance(\n'
            f'    database="{collection._database_name}", \n'
            f'    collection="{collection._collection_name}", \n'
            f'    shape={collection.shape}'
            f'\n)')


# Use threading.local to create thread-local storage
thread_local = threading.local()
