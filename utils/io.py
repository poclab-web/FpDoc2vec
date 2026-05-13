import pickle


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
