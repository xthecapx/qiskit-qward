from typing import Any, Dict, Union


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Union[int, bool, str, list, Any]]:
    """
    Recursively flattens a nested dictionary using dot notation for keys.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested flattening
        sep: Separator to use between keys

    Returns:
        Flattened dictionary with dot-notation keys

    Example:
        {'a': {'b': 1}} -> {'a.b': 1}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
