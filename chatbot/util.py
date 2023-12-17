# util.py
# File for utilities functions

from yaml import safe_load as yaml_load
from pickle import load as pickle_load

from typing import Any


def yaml_from_file(filename: str) -> Any:
    print(f'Loading YAML object from file "{filename}"...')

    with open(filename, 'rb') as file_obj:
        return yaml_load(file_obj)


def pickle_from_file(filename: str) -> Any:
    print(f'Loading Pickle object from file "{filename}"...')

    with open(filename, 'rb') as file_obj:
        return pickle_load(file_obj)
