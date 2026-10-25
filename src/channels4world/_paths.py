"""Paths taken from metadata are relative POSIX paths that must stay inside a root.

Writes go to ``<target>.part`` created exclusively (never through an existing file or
link) and are then renamed. Existing symbolic links inside the destination are refused
rather than followed. (Concurrent tampering with the destination while writing is out of
scope.)
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from ._errors import FormatError


def safe_relpath(rel: str) -> str:
    """Return ``rel`` if it is a plain relative POSIX path without '..', else raise."""
    if (not isinstance(rel, str) or not rel or "\\" in rel or "\x00" in rel or PurePosixPath(rel).is_absolute()
            or any(part in ("", ".", "..") for part in rel.split("/"))):
        raise FormatError(f"unsafe path {rel!r} in dataset metadata")
    return rel


def safe_name(name: str) -> str:
    """A single path component (scene IDs, group names)."""
    if "/" in str(name):
        raise FormatError(f"unsafe name {name!r} in dataset metadata")
    return safe_relpath(name)


def target(dest: Path, rel: str) -> Path:
    """``dest/rel`` after checking that no existing component below ``dest`` is a symlink."""
    safe_relpath(rel)
    cur = dest
    for part in PurePosixPath(rel).parts:
        cur = cur / part
        if cur.is_symlink():
            raise FormatError(f"{cur} is a symbolic link; refusing to write through it")
    return cur


def fresh_temp(path: Path) -> Path:
    """Create the parent directories and return an unused ``.part`` path (stale ones removed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    return tmp


def write_file(dest: Path, rel: str, body: bytes) -> Path:
    path = target(dest, rel)
    tmp = fresh_temp(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return path
