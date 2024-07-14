import importlib


def load_module(module: str):
    """
    Method to load a module specified as string.

    Parameters
    ----------
    module: str
        The module to load.

    Returns
    -------
    The loaded module.
    """

    # split the string into module name and class name
    module_name, class_name = module.rsplit('.', 1)

    # import the module dynamically
    module = importlib.import_module(module_name)

    # access the class object using getattr
    object = getattr(module, class_name)

    return object
