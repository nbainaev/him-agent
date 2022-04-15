#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from ast import literal_eval
from typing import Iterable, Any, Optional

# TODO: rename file to `config` and add top-level description comment


# Register config-related conventional constants here. Start them with `_` as non-importable!
_TYPE_KEY = '_type_'
_TO_BE_INDUCED_VALUE = 'TBI'


def filtered(d: dict, keys_to_remove: Iterable[str], depth: int) -> dict:
    """
    Return a shallow copy of the provided dictionary without the items
    that match `keys_to_remove`.

    The `depth == 1` means filtering `d` itself,
        `depth == 2` — with its dict immediate descendants
        and so on.
    """
    if not isinstance(d, dict) or depth <= 0:
        return d

    return {
        k: filtered(v, keys_to_remove, depth - 1)
        for k, v in d.items()
        if k not in keys_to_remove
    }


# sadly, cannot specify the correct type hint here, which is tuple[dict, Optional[Any], ...]
def extracted(d: dict, *keys: str) -> tuple:
    """
    Return a copy of the dictionary without specified keys and each extracted value
    (or None if a specified key was absent).

    Examples
    --------
    >>> extracted({'a': 1, 'b': 2, 'c': 3}, 'a', 'c')
    ({'b': 2}, 1, 3)
    """
    values = tuple([d.get(k, None) for k in keys])
    filtered_dict = filtered(d, keys, depth=1)
    return (filtered_dict, ) + values


def extracted_type(config: dict) -> tuple[dict, Optional[str]]:
    return extracted(config, _TYPE_KEY)


def split_arg(arg: str) -> tuple[str, str]:
    # "--key=value" --> ["--key", "value"]
    key_path, value = arg.split('=', maxsplit=1)

    # "--key" --> "key"
    key_path = key_path.removeprefix('--')

    # TODO: modify the func to parse_arg str -> (key_path: list, value: Any)
    return key_path, value


def parse_str(s: str) -> Any:
    """Parse string value to the most appropriate type."""

    # noinspection PyShadowingNames
    def boolify(s):
        if s in ['True', 'true']:
            return True
        if s in ['False', 'false']:
            return False
        raise ValueError('Not Boolean Value!')

    # NB: try/except is widely accepted pythonic way to parse things

    # NB: order here is important
    for caster in (boolify, int, float, literal_eval):
        try:
            return caster(s)
        except ValueError:
            pass
    return s
