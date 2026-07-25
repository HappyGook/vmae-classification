import json
import os

import numpy as np


def to_serializable(obj):
    """Recursively convert numpy values and tuples into JSON-friendly values."""
    if isinstance(obj, dict):
        return {
            key: to_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            to_serializable(value)
            for value in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def safe_name(*parts):
    """Build a filesystem-safe filename from arbitrary parts."""
    raw = "_".join(str(part) for part in parts if part is not None)
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in raw
    )


def save_json(data, save_dir, filename):
    """Serialize data to `save_dir / filename`, adding `.json` if needed."""
    os.makedirs(save_dir, exist_ok=True)

    if not filename.endswith(".json"):
        filename += ".json"

    path = os.path.join(save_dir, filename)

    with open(path, "w") as file:
        json.dump(to_serializable(data), file, indent=2)

    return path


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)