

def copy_doc(target_method, source_method):
    """Copy the docstring from the source method to the target method.

    Parameters:
        target_method: The target method.
        source_method: The source method.
    """
    target_method.__func__.__doc__ = source_method.__doc__
