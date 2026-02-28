version = "unknown"

def set_version(vers: str):
    global version
    version = vers

def get_version() -> str:
    return version