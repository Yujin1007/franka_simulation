__version__ = "0.0.0.2"

import os
import franka_valve

PACKAGE_DIR = os.path.dirname(franka_valve.__file__)

def check_valid_object(object):
    valid_values = ["handle", "valve"]

    if object not in valid_values:
        raise ValueError(f"Invalid object. Valid values are {valid_values}")