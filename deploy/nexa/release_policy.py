"""File policy shared by Nexa packaging and promotion."""
import fnmatch
import hashlib


FORBIDDEN = (".env", ".env.*", "*.env", "*.glb", "*.db", "*.sqlite*")


def check_entry(relative):
    if any(fnmatch.fnmatchcase(relative.name.lower(), pattern) for pattern in FORBIDDEN):
        raise ValueError(f"forbidden release entry: {relative}")


def digest(path, check_content):
    checksum = hashlib.sha256()
    needle = b"/home/kza"
    tail = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if check_content and needle in tail + chunk:
                raise ValueError(f"private host path in {path}")
            tail = chunk[-(len(needle) - 1):]
            checksum.update(chunk)
    return checksum.hexdigest()
