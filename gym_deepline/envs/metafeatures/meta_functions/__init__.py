import os

from gym_deepline.envs.metafeatures._aux.discovery import discover_components
from gym_deepline.envs.metafeatures.meta_functions.base import MetaFunction


meta_functions_directory = os.path.split(__file__)[0]
meta_functions = discover_components(__package__,
                                     meta_functions_directory,
                                     MetaFunction)