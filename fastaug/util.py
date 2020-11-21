import importlib_resources


def load_resource(path: str):
    return importlib_resources.files("fastaug") / "resources" / path
    