"""Local preparation: subset extraction and NPY cache conversion.

Both write a normal channels4world root that reopens with ``c4w.open``. Files are written
under temporary names and renamed; the root file is written last, so an interrupted run
never looks complete. ``check_space`` and ``verify`` are independent of any confirmation.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import _contract as C
from ._errors import FormatError, InsufficientSpaceError, NotDownloadedError
from ._paths import fresh_temp, safe_name, safe_relpath, target, write_file
from ._root import FILES_SCHEMA, FORMAT, FORMAT_VERSION, RECORDS_SCHEMA, ROOT_FILE, SCENES_SCHEMA, TABLE_DIR, Root
from .dataset import Selection, _fmt_bytes


@dataclass
class Plan:
    """What a preparation step will write (returned to an optional ``confirm`` callback)."""

    action: str
    dest: Path
    scenes: int
    records: int
    files: int
    bytes: int
    free_bytes: int
    unpublished: int
    skipped_existing: int = 0
    items: list[dict[str, Any]] = field(default_factory=list, repr=False)
    meta: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def describe(self) -> str:
        what = "loose files" if self.action == "extract" else "NPY cache"
        lines = [f"Selected   {self.scenes} scenes · {self.records} channel records",
                 f"Writes     {self.files} files · {_fmt_bytes(self.bytes)} ({what}; filesystem overhead not included)",
                 f"Free       {_fmt_bytes(self.free_bytes)} at {self.dest}"]
        if self.skipped_existing:
            lines.append(f"Existing   {self.skipped_existing} matching files will be kept")
        if self.unpublished:
            lines.append(f"Note       {self.unpublished} requested items are not published and will be skipped")
        return "\n".join(lines)


def _free(dest: Path) -> int:
    p = dest
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def _check_dest(dest: Path, allow_existing: bool, hint: str = "") -> None:
    if dest.exists() and any(dest.iterdir()) and not allow_existing:
        raise FileExistsError(f"{dest} is not empty; choose a new directory{hint}")


def _write_atomic(dest: Path, rel: str, body: bytes) -> None:
    write_file(dest, rel, body)


def _table_bytes(rows: list[dict[str, Any]], schema: pa.Schema) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def _collect(sel: Selection) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, int]:
    """Items (published records to write) and metadata files to copy."""
    r = sel.dataset._root
    items: list[dict[str, Any]] = []
    groups: list[str] = []
    scenes: set[str] = set()
    records = unpublished = 0
    missing: list[str] = []
    for item in sel._plan():
        scenes.add(item.scene.id)
        if item.paths is None:
            unpublished += 1
            continue
        g = item.scene._group
        if g.name not in groups:
            groups.append(g.name)
        for p in item.paths:
            if not (r.path / p).exists():
                missing.append(p)
        if item.component == "core":
            records += 1
        m = g.members[item.record_id]
        items.append({"group": g.name, "record_id": item.record_id, "component": item.component,
                      "frequency_hz": item.hz, "member_path": m["member_path"], "bytes": m["member_bytes"],
                      "sha256": m["member_sha256"], "scene_id": item.scene.id})
    unpublished += len(sel._absent())
    if not items:
        raise ValueError("the selection contains nothing to write (check its filters, or what local() kept)")
    for g in groups:  # group metadata is always part of what preparation copies: check it up front
        missing += [p for p in r.metadata_paths(g) if not (r.path / p).exists()]
    if missing:
        uniq = sorted(set(missing))
        raise NotDownloadedError(f"{len(uniq)} selected file(s) are not downloaded, e.g. {uniq[0]}. "
                                 "Download them (see selection.missing()) or use selection.local().", uniq)
    meta = [f for g in groups for f in r.group_files[g] if f["kind"] in ("members", "snapshot", "catalog")]
    meta = [f for f in meta if f["kind"] != "catalog" or (r.path / f["path"]).exists()]
    meta += [f for f in r.files if f["kind"] == "schema"]
    return items, meta, len(scenes), records, unpublished


def _copy_meta(src: Path, dest: Path, meta: list[dict[str, Any]]) -> None:
    for f in meta:
        body = (src / safe_relpath(f["path"])).read_bytes()
        if f["sha256"] and C.sha256(body) != f["sha256"]:
            from ._errors import IntegrityError
            raise IntegrityError(f"{f['path']} differs from the source root")
        _write_atomic(dest, f["path"], body)


def _write_root(sel: Selection, dest: Path, kind: str, payload: str, file_rows: list[dict[str, Any]],
                record_ids: list[str]) -> None:
    r = sel.dataset._root
    scene_ids = set(sel.scene_ids)
    scenes = [{"scene_id": s.scene_id, "split": s.split, "profile": s.profile, "tier": s.tier, "group": s.group,
               "group_order": s.order[0], "scene_order": s.order[1]} for s in r.scenes.values() if s.scene_id in scene_ids]
    tables = {}
    for name, rows, schema in (("scenes", scenes, SCENES_SCHEMA), ("files", file_rows, FILES_SCHEMA),
                               ("records", [{"record_id": x} for x in record_ids], RECORDS_SCHEMA)):
        body = _table_bytes(rows, schema)
        rel = f"{TABLE_DIR}/{name}.parquet"
        _write_atomic(dest, rel, body)
        tables[name] = {"path": rel, "sha256": C.sha256(body), "bytes": len(body)}
    from . import __version__
    # Carriers actually written, so selecting an absent carrier fails instead of matching nothing.
    present_hz = {f["frequency_hz"] for f in file_rows if f["kind"] in ("payload", "cache_index")
                  and f["component"] in ("core", "detail") and f["frequency_hz"] is not None}
    frequencies = [hz for hz in r.frequencies_hz if hz in present_hz]
    manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "kind": kind, "payload": payload,
                "collection_id": r.collection_id, "release": r.release, "tiers": r.tiers,
                "frequencies_hz": frequencies, "hf_repo": None, "hf_path": None, "tables": tables,
                "source": {"release": r.release, "kind": r.kind, "root_sha256": r.manifest_sha256},
                "created_by": f"channels4world {__version__}"}
    _write_atomic(dest, ROOT_FILE, json.dumps(manifest, indent=1, sort_keys=True).encode())


def _meta_rows(meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: f[k] for k in FILES_SCHEMA.names} for f in meta]


# =============================================================================== extract
def _check_extension(dest: Path, src: Root, new_ids: set[str]) -> None:
    """``skip_existing`` resumes or extends an earlier extraction; it never drops what ``dest`` holds.

    The new root lists exactly the new selection, so the selection must include every record
    the existing subset lists (for example, select both carriers to add a second one).
    """
    prev = Root(dest)
    if prev.kind != "subset" or prev.collection_id != src.collection_id:
        raise FileExistsError(f"{dest} holds a different {prev.kind} ({prev.collection_id}); choose a new directory")
    dropped = (prev.included or set()) - new_ids
    if dropped:
        raise FileExistsError(
            f"{dest} already holds {len(dropped)} record(s) outside this selection, which skip_existing would "
            "drop from the subset. Select everything it already holds plus what you want to add (for example "
            "both carriers), or extract into a new directory.")


def extract(selection: Selection, dest: str | os.PathLike[str], *, verify: str = "basic",
            check_space: bool = True, skip_existing: bool = False,
            confirm: Callable[[Plan], bool] | None = None) -> Path:
    """Copy the selected published files (unchanged) to their published loose paths under ``dest``.

    The result reopens with ``c4w.open(dest)``; its record list defines what it contains.
    """
    C.check_verify(verify)
    dest = Path(dest).expanduser().resolve()
    _check_dest(dest, skip_existing, " or pass skip_existing=True")
    src = selection.dataset._root
    if src.kind == "cache":
        raise FormatError("extract copies published NPZ/ZIP files; select from the release or a subset, not a cache")
    items, meta, n_scenes, n_records, unpublished = _collect(selection)
    if skip_existing and (dest / ROOT_FILE).is_file():
        _check_extension(dest, src, {it["record_id"] for it in items})
    todo, kept = [], 0
    for it in items:
        existing = target(dest, it["member_path"])
        if skip_existing and existing.is_file() and existing.stat().st_size == it["bytes"] and (
                verify == "none" or C.sha256(existing.read_bytes()) == it["sha256"]):
            kept += 1
        else:
            todo.append(it)
    nbytes = sum(it["bytes"] for it in todo) + sum(f["bytes"] for f in meta)
    plan = Plan("extract", dest, n_scenes, n_records, len(todo) + len(meta), nbytes, _free(dest), unpublished,
                kept, todo, meta)
    if check_space and plan.bytes > plan.free_bytes:
        raise InsufficientSpaceError(f"extraction needs {_fmt_bytes(plan.bytes)} but only "
                                     f"{_fmt_bytes(plan.free_bytes)} is free at {dest} (check_space=False to skip)")
    if confirm is not None and not confirm(plan):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    root_file = dest / ROOT_FILE
    if root_file.exists():
        root_file.unlink()  # the root is rewritten last
    for it in todo:
        g = src.group(it["group"])
        if it["component"] in ("geometry",) or verify == "none":
            body = g.read_member(it["record_id"], verify)
        else:
            body = src.read_bytes(src.locate(g, it["record_id"]), verify)
            if verify != "none":
                row = g.snapshot.row(it["record_id"])
                C.decode_npz(body, row, g.snapshot.companion, verify)  # basic: hash + structure; full: + science
        _write_atomic(dest, it["member_path"], body)
    _copy_meta(src.path, dest, meta)
    rows = _meta_rows(meta) + [
        {"path": it["member_path"], "bytes": it["bytes"], "sha256": it["sha256"], "group": it["group"],
         "kind": "payload", "component": it["component"], "frequency_hz": it["frequency_hz"],
         "record_id": it["record_id"]} for it in items]
    _write_root(selection, dest, "subset", "loose", rows, [it["record_id"] for it in items])
    return dest


# =============================================================================== cache
def _npy_bytes(shape: tuple[int, ...], dtype: str) -> int:
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize + 128


def to_cache(selection: Selection, dest: str | os.PathLike[str], *, verify: str = "basic",
             check_space: bool = True, confirm: Callable[[Plan], bool] | None = None) -> Path:
    """Convert the selection to memory-mappable NPY arrays (values, dtypes, IDs and ragged
    structure unchanged). Geometry, if selected, is copied as its original ZIP.

    ``verify="full"`` also checks every written record against the published array hashes.
    """
    C.check_verify(verify)
    dest = Path(dest).expanduser().resolve()
    _check_dest(dest, False)
    src = selection.dataset._root
    items, meta, n_scenes, n_records, unpublished = _collect(selection)
    parts: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
    geometry = []
    for it in items:
        if it["component"] == "geometry":
            geometry.append(it)
        else:
            parts.setdefault((it["group"], it["component"], it["frequency_hz"]), []).append(it)
    nbytes = sum(f["bytes"] for f in meta) + sum(it["bytes"] for it in geometry)
    nfiles = len(meta) + len(geometry)
    for (gname, comp, _hz), its in parts.items():
        snap = src.group(gname).snapshot
        totals: dict[str, int] = {n: 0 for n in C.FIELDS[comp]}
        for it in its:
            for n, sh in C.shapes(snap.row(it["record_id"])["record"]).items():
                totals[n] += sh[0]
        for n, dtype in C.FIELDS[comp].items():
            shape = (totals[n], 3) if n in ("interaction_vertex", "bs_positions", "ue_positions") else (totals[n],)
            nbytes += _npy_bytes(shape, dtype)
        nfiles += len(C.FIELDS[comp]) + 1
    plan = Plan("cache", dest, n_scenes, n_records, nfiles, nbytes, _free(dest), unpublished, 0, items, meta)
    if check_space and plan.bytes > plan.free_bytes:
        raise InsufficientSpaceError(f"the cache needs {_fmt_bytes(plan.bytes)} but only "
                                     f"{_fmt_bytes(plan.free_bytes)} is free at {dest} (check_space=False to skip)")
    if confirm is not None and not confirm(plan):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for (gname, comp, hz), its in parts.items():
        g = src.group(gname)
        snap = g.snapshot
        part_dir = f"cache/{safe_name(gname)}/{comp}" + ("" if hz is None else f"-{hz}")
        row_of = {it["record_id"]: snap.row(it["record_id"]) for it in its}
        totals = {n: 0 for n in C.FIELDS[comp]}
        for rid in row_of:
            for n, sh in C.shapes(row_of[rid]["record"]).items():
                totals[n] += sh[0]
        maps, fields = {}, {}
        for n, dtype in C.FIELDS[comp].items():
            shape = (totals[n], 3) if n in ("interaction_vertex", "bs_positions", "ue_positions") else (totals[n],)
            rel = f"{part_dir}/{n}.npy"
            maps[n] = np.lib.format.open_memmap(fresh_temp(target(dest, rel)), mode="w+", dtype=np.dtype(dtype),
                                                shape=shape)
            fields[n] = rel
        pos = {n: 0 for n in maps}
        for it in its:
            arrays = g.read_numeric(it["record_id"], verify)
            for n, a in arrays.items():
                maps[n][pos[n]:pos[n] + len(a)] = a
                pos[n] += len(a)
        for n, m in maps.items():
            m.flush()
            del m
        maps.clear()
        for n, rel in fields.items():
            os.replace(dest / (rel + ".part"), dest / rel)
        if verify == "full":
            check = {n: np.load(dest / rel, mmap_mode="r") for n, rel in fields.items()}
            start = {n: 0 for n in check}
            for it in its:
                row = row_of[it["record_id"]]
                rec = {}
                for n, sh in C.shapes(row["record"]).items():
                    rec[n] = check[n][start[n]:start[n] + sh[0]]
                    start[n] += sh[0]
                C.check_full(row, rec, snap.companion)
        index = json.dumps({"component": comp, "frequency_hz": hz, "records": [it["record_id"] for it in its],
                            "fields": fields}, indent=1).encode()
        _write_atomic(dest, f"{part_dir}/index.json", index)
        rows.append({"path": f"{part_dir}/index.json", "bytes": len(index), "sha256": C.sha256(index), "group": gname,
                     "kind": "cache_index", "component": comp, "frequency_hz": hz, "record_id": None})
        for n, rel in fields.items():
            rows.append({"path": rel, "bytes": (dest / rel).stat().st_size, "sha256": None, "group": gname,
                         "kind": "cache_array", "component": comp, "frequency_hz": hz, "record_id": None})
    for it in geometry:
        body = src.group(it["group"]).read_member(it["record_id"], verify)
        _write_atomic(dest, it["member_path"], body)
        rows.append({"path": it["member_path"], "bytes": it["bytes"], "sha256": it["sha256"], "group": it["group"],
                     "kind": "payload", "component": "geometry", "frequency_hz": None, "record_id": it["record_id"]})
    _copy_meta(src.path, dest, meta)
    _write_root(selection, dest, "cache", "npy", _meta_rows(meta) + rows, [it["record_id"] for it in items])
    return dest
