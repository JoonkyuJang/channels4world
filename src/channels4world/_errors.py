"""Public exceptions. Each carries a plain message saying what happened and what to do."""

from __future__ import annotations


class C4WError(Exception):
    """Base class for channels4world errors."""


class UnknownIDError(C4WError, KeyError):
    """An ID (split, scene, frequency, base station, UE) does not exist."""

    def __str__(self) -> str:  # KeyError would otherwise print the repr of the message
        return str(self.args[0]) if self.args else ""


class NotPublishedError(C4WError, LookupError):
    """The requested component is not part of this release or subset."""


class NotDownloadedError(C4WError, FileNotFoundError):
    """Published data whose file is not present locally."""

    def __init__(self, message: str, paths: list[str]) -> None:
        super().__init__(message)
        self.paths = list(paths)

    def __str__(self) -> str:
        return str(self.args[0])


class IntegrityError(C4WError, ValueError):
    """Bytes or arrays differ from what the release published."""


class FormatError(C4WError, ValueError):
    """The directory is not a readable channels4world root (or is incomplete)."""


class InsufficientSpaceError(C4WError, OSError):
    """The destination does not have enough free space."""

    def __str__(self) -> str:
        return str(self.args[0])
