import json
from pathlib import Path


def load_dict_from_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
