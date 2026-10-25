"""Selection recipes: a small JSON file that records a selection, so it can be shared and replayed.

A recipe stores what was asked for (tier, split, profiles, scene IDs, carriers, BS IDs and
components) and which release it refers to. It stores no file paths or commands: the
recipient opens their own copy of the catalog and plans downloads from there, so the same
recipe works for the release root and for Tiny, and for any files already downloaded.

    {
      "format": "channels4world.selection",
      "format_version": 1,
      "release": "1.0.0",
      "revision": "v1.0.0",
      "selection": {"tier": "small", "profiles": ["urban"], "components": ["core", "positions"]}
    }

``revision`` (optional) is the Hugging Face revision the catalog came from; ``c4w missing
--recipe`` uses it to pin the printed download command.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import _contract as C

FORMAT = "channels4world.selection"
FORMAT_VERSION = 1
MAX_BYTES = 8_000_000
MAX_SCENE_IDS = 100_000
SELECTION_FIELDS = ("tier", "split", "profiles", "scene_ids", "frequencies_hz", "bs_ids", "components")
_TOP_FIELDS = ("format", "format_version", "release", "revision", "selection")
_COMPONENTS = ("core", "detail", "positions", "geometry")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,127}")


class RecipeError(ValueError):
    """The recipe is not a valid Channels4World selection recipe."""


_MAX_INT = 2**53 - 1  # the largest integer every JSON reader (JavaScript included) keeps exactly


def _int(value: Any, label: str) -> int:
    if type(value) is not int:  # bool and float are refused (3.5e9 parses as a float)
        raise RecipeError(f"{label} must be integers, not {type(value).__name__}")
    if not 0 <= value <= _MAX_INT:
        raise RecipeError(f"{label} must be non-negative integers below 2**53")
    return value


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise RecipeError(f"{label} must be plain names (letters, digits, '_', '.', '-'), not {value!r}")
    return value


def _names(value: Any, label: str, limit: int = 64) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RecipeError(f"{label} must be a non-empty list")
    if len(value) > limit:
        raise RecipeError(f"{label} has {len(value)} entries; at most {limit} are allowed")
    return list(dict.fromkeys(_name(v, label) for v in value))


def validate(recipe: Any) -> dict[str, Any]:
    """Check a parsed recipe and return its normalized form. Raises :class:`RecipeError`."""
    if not isinstance(recipe, Mapping):
        raise RecipeError("a recipe is a JSON object")
    unknown = [k for k in recipe if k not in _TOP_FIELDS]
    if unknown:
        raise RecipeError(f"unknown recipe field(s): {', '.join(map(str, unknown))}")
    if recipe.get("format") != FORMAT:
        raise RecipeError(f"not a selection recipe (format must be {FORMAT!r})")
    version = recipe.get("format_version")
    if type(version) is not int or version != FORMAT_VERSION:
        raise RecipeError(f"recipe format_version {version!r} is not supported (expected {FORMAT_VERSION})")
    out: dict[str, Any] = {"format": FORMAT, "format_version": FORMAT_VERSION,
                           "release": _name(recipe.get("release"), "release")}
    revision = recipe.get("revision")
    if revision is not None:
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision) or ".." in revision:
            raise RecipeError(f"revision must be a branch, tag or commit name, not {revision!r}")
        out["revision"] = revision
    sel = recipe.get("selection")
    if not isinstance(sel, Mapping):
        raise RecipeError("selection must be an object")
    unknown = [k for k in sel if k not in SELECTION_FIELDS]
    if unknown:
        raise RecipeError(f"unknown selection field(s): {', '.join(map(str, unknown))} "
                          f"(supported: {', '.join(SELECTION_FIELDS)})")
    norm: dict[str, Any] = {}
    for key in ("tier", "split"):
        if sel.get(key) is not None:
            norm[key] = _name(sel[key], key)
    if sel.get("profiles") is not None:
        norm["profiles"] = _names(sel["profiles"], "profiles")
    if sel.get("scene_ids") is not None:
        norm["scene_ids"] = _names(sel["scene_ids"], "scene_ids", MAX_SCENE_IDS)
    for key, label in (("frequencies_hz", "frequencies_hz"), ("bs_ids", "bs_ids")):
        if sel.get(key) is not None:
            values = sel[key]
            if not isinstance(values, list) or not values:
                raise RecipeError(f"{label} must be a non-empty list")
            if len(values) > 1024:
                raise RecipeError(f"{label} has too many entries")
            norm[key] = sorted({_int(v, label) for v in values})
    comps = sel.get("components")
    if comps is None:
        raise RecipeError("selection.components is required (e.g. [\"core\", \"positions\"])")
    comps = _names(comps, "components", len(_COMPONENTS))
    bad = [c for c in comps if c not in _COMPONENTS]
    if bad:
        raise RecipeError(f"unknown component(s) {bad}; choose from {list(_COMPONENTS)}")
    if "detail" in comps and "core" not in comps:
        comps.append("core")  # detail is always read with its core record
    norm["components"] = [c for c in _COMPONENTS if c in comps]
    out["selection"] = norm
    return out


def loads(text: str | bytes) -> dict[str, Any]:
    """Parse and validate recipe JSON (duplicate keys and NaN/Infinity are rejected)."""
    if len(text) > MAX_BYTES:
        raise RecipeError(f"recipe is larger than {MAX_BYTES} bytes")
    try:
        parsed = C.strict_json(text)
    except ValueError as exc:
        raise RecipeError(f"recipe is not valid JSON: {exc}") from None
    return validate(parsed)


def load(source: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """A validated recipe from a file path or an already parsed mapping."""
    if isinstance(source, Mapping):
        return validate(source)
    path = Path(source)
    if path.stat().st_size > MAX_BYTES:
        raise RecipeError(f"{path} is larger than {MAX_BYTES} bytes")
    return loads(path.read_bytes())


def dumps(recipe: Mapping[str, Any]) -> str:
    """Recipe JSON with a stable key order."""
    r = validate(recipe)
    ordered = {k: r[k] for k in _TOP_FIELDS if k in r}
    ordered["selection"] = {k: r["selection"][k] for k in SELECTION_FIELDS if k in r["selection"]}
    return json.dumps(ordered, indent=2, ensure_ascii=True) + "\n"


def select_kwargs(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Keyword arguments for :meth:`Dataset.select` (validated recipe in, nothing read)."""
    s = recipe["selection"]
    return {"tier": s.get("tier"), "split": s.get("split"), "profile": s.get("profiles"),
            "scene_ids": s.get("scene_ids"), "frequencies_hz": s.get("frequencies_hz"),
            "bs_ids": s.get("bs_ids"), "components": tuple(s["components"])}
