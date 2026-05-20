import pickle
from typing import Any


def load_pickle(path: str) -> Any:
    """Load and return a Python object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str) -> None:
    """Serialize and save a Python object to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
