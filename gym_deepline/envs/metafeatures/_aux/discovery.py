import importlib
import inspect
import pkgutil
import sys


def discover_components(package, directory, base_class):
    """Discover implementations of a base class in a package.

    Parameters
    ----------
    package : str
        Package name
    directory : str
        Directory of the package to which is inspected.
    base_class : object
        Base class of objects to discover

    Returns
    -------
    list : all subclasses of `base_class` inside `directory`
    """

    components = list()

    for module_loader, module_name, ispkg in pkgutil.iter_modules(
            [directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(base_class, obj):
                    classifier = obj
                    components.append(classifier)

    return components