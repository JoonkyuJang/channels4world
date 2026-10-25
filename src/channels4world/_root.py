"""Release/subset/cache roots, lazy per-group metadata, and payload access.

A root is a directory containing ``c4w-root.json`` (small manifest, written last) and
``c4w-root/`` tables:

* ``scenes.parquet``  one row per scene: scene_id, split, profile, tier, group, group_order, scene_order
* ``files.parquet``   one row per file this root describes: path, bytes, sha256, group, kind,
                      component, frequency_hz, record_id
* ``records.parquet`` subsets/caches only: the included record IDs

Group metadata (``members.parquet``, ``snapshot.json``) is checked against the files table
whenever it is loaded. ``verify`` only controls checks on payload bytes and arrays.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import _contract as C
from ._errors import FormatError, IntegrityError, NotDownloadedError, NotPublishedError
from ._paths import safe_name, safe_relpath

ROOT_FILE = "c4w-root.json"
TABLE_DIR = "c4w-root"
FORMAT = "channels4world.root"
FORMAT_VERSION = 1
GROUP_CACHE = 4

SCENES_SCHEMA = pa.schema([
    ("scene_id", pa.string()), ("split", pa.string()), ("profile", pa.string()), ("tier", pa.string()),
    ("group", pa.string()), ("group_order", pa.int64()), ("scene_order", pa.int64())])
FILES_SCHEMA = pa.schema([
    ("path", pa.string()), ("bytes", pa.int64()), ("sha256", pa.string()), ("group", pa.string()),
    ("kind", pa.string()), ("component", pa.string()), ("frequency_hz", pa.int64()), ("record_id", pa.string())])
RECORDS_SCHEMA = pa.schema([("record_id", pa.string())])
# kinds: members, snapshot, catalog, schema, payload, cache_index, cache_array


class SceneInfo(NamedTuple):
    scene_id: str
    split: str
    profile: str
    tier: str
    group: str
    order: tuple[int, int]


@dataclass(frozen=True)
class Location:
    kind: str          # "tar" | "file"
    path: str          # relative to the root
    offset: int
    size: int
    published_bytes: int | None


def _read_table(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path, use_threads=False).to_pylist()


class Root:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest_path = self.path / ROOT_FILE
        if not manifest_path.is_file():
            raise FormatError(
                f"{self.path} has no {ROOT_FILE}. Point c4w.open at a downloaded release, Tiny, an extracted "
                "subset or a cache; an interrupted extraction or conversion leaves no root file.")
        body = manifest_path.read_bytes()
        self.manifest_sha256 = C.sha256(body)
        m = json.loads(body)
        if m.get("format") != FORMAT:
            raise FormatError(f"{manifest_path} is not a channels4world root")
        if m.get("format_version") != FORMAT_VERSION:
            raise FormatError(f"root format version {m.get('format_version')} is not supported by this library; "
                              "upgrade channels4world")
        self.manifest = m
        self.kind: str = m["kind"]
        self.collection_id: str = m["collection_id"]
        self.release: str = m["release"]
        self.tiers: list[str] = list(m["tiers"])
        self.frequencies_hz: list[int] = [int(f) for f in m["frequencies_hz"]]
        self.hf_repo: str | None = m.get("hf_repo")
        # Repository-relative folder of this root as published ("" = top level); None when this
        # copy is not a published root (e.g. a user-made subset or cache).
        self.hf_path: str | None = m.get("hf_path")
        if self.hf_path:
            safe_relpath(self.hf_path)

        tables = {name: self._table(name, ref) for name, ref in m["tables"].items()}
        for r in tables["scenes"]:
            safe_name(r["scene_id"])
            safe_name(r["group"])
        for f in tables["files"]:
            safe_relpath(f["path"])
        self.scenes: dict[str, SceneInfo] = {}
        for r in tables["scenes"]:
            self.scenes[r["scene_id"]] = SceneInfo(r["scene_id"], r["split"], r["profile"], r["tier"], r["group"],
                                                   (r["group_order"], r["scene_order"]))
        self.scene_order = sorted(self.scenes, key=lambda s: self.scenes[s].order)
        self.splits: list[str] = list(dict.fromkeys(self.scenes[s].split for s in self.scene_order))
        self.files = tables["files"]
        self.by_path = {f["path"]: f for f in self.files}
        self.by_record = {f["record_id"]: f for f in self.files if f["record_id"] and f["kind"] == "payload"}
        self.group_files: dict[str, list[dict[str, Any]]] = {}
        for f in self.files:
            if f["group"]:
                self.group_files.setdefault(f["group"], []).append(f)
        self.included: set[str] | None = (
            {r["record_id"] for r in tables["records"]} if "records" in tables else None)
        self._groups: OrderedDict[str, Group] = OrderedDict()
        self._companion: C.Companion | None = None
        # Hashes of snapshots that passed full validation in this process. Reloading an evicted
        # group re-checks the bytes against the root but skips re-validating identical rows.
        self._validated_snapshots: set[str] = set()

    def _table(self, name: str, ref: dict[str, Any]) -> list[dict[str, Any]]:
        p = self.path / safe_relpath(ref["path"])
        if not p.is_file():
            raise FormatError(f"root table {ref['path']} is missing")
        if C.sha256(p.read_bytes()) != ref["sha256"]:
            raise IntegrityError(f"root table {ref['path']} differs from {ROOT_FILE}")
        return _read_table(p)

    # ------------------------------------------------------------------ groups
    def group(self, name: str) -> "Group":
        g = self._groups.get(name)
        if g is None:
            g = Group(self, name)
            self._groups[name] = g
            while len(self._groups) > GROUP_CACHE:
                self._groups.popitem(last=False)
        else:
            self._groups.move_to_end(name)
        return g

    def group_meta_file(self, group: str, kind: str) -> dict[str, Any]:
        for f in self.group_files.get(group, ()):
            if f["kind"] == kind:
                return f
        raise FormatError(f"the root lists no {kind} file for group {group}")

    def frequencies_of_group(self, group: str) -> list[int]:
        return sorted({f["frequency_hz"] for f in self.group_files.get(group, ())
                       if f["kind"] in ("payload", "cache_index") and f["component"] in ("core", "detail")
                       and f["frequency_hz"] is not None})

    def metadata_paths(self, group: str) -> list[str]:
        """Metadata a selection over this group always requires: members, snapshot, shared schemas."""
        return ([self.group_meta_file(group, k)["path"] for k in ("members", "snapshot")]
                + [f["path"] for f in self.files if f["kind"] == "schema"])

    def is_included(self, record_id: str) -> bool:
        return self.included is None or record_id in self.included

    # ------------------------------------------------------------------ locations
    def locate(self, group: "Group", record_id: str) -> Location:
        loose = self.by_record.get(record_id)
        if loose is not None:
            return Location("file", loose["path"], 0, loose["bytes"], loose["bytes"])
        member = group.members.get(record_id)
        if member is None or not self.is_included(record_id):
            raise NotPublishedError("record is not part of this release or subset")
        obj = self.by_path.get(member["object_path"])
        if obj is None:
            raise NotPublishedError("record's archive is not part of this root")
        return Location("tar", member["object_path"], member["data_offset"], member["member_bytes"], obj["bytes"])

    def read_bytes(self, loc: Location, verify: str) -> bytes:
        p = self.path / loc.path
        try:
            if verify != "none" and loc.kind == "tar" and loc.published_bytes is not None:
                size = p.stat().st_size
                if size != loc.published_bytes:
                    raise IntegrityError(f"{loc.path} has {size} bytes, the release published "
                                         f"{loc.published_bytes} (incomplete download?)")
            with p.open("rb") as fh:
                fh.seek(loc.offset)
                body = fh.read(loc.size)
        except FileNotFoundError:
            raise NotDownloadedError(f"{loc.path} is not downloaded", [loc.path]) from None
        if len(body) != loc.size:
            raise IntegrityError(f"short read from {loc.path}")
        return body


class Group:
    """One release group's metadata, loaded lazily and checked against the root."""

    def __init__(self, root: Root, name: str) -> None:
        self.root = root
        self.name = name
        self._members: dict[str, dict[str, Any]] | None = None
        self._snapshot: C.Snapshot | None = None
        self._parts: dict[tuple[str, int | None], CachePart | None] = {}
        self._ident: dict[str, tuple[str, int | None]] | None = None
        self._by_scene: dict[str, dict[tuple[str, int | None, int | None], str]] | None = None

    def _meta_bytes(self, kind: str) -> bytes:
        f = self.root.group_meta_file(self.name, kind)
        p = self.root.path / f["path"]
        if not p.is_file():
            raise NotDownloadedError(f"group metadata {f['path']} is not downloaded", [f["path"]])
        body = p.read_bytes()
        if C.sha256(body) != f["sha256"]:
            raise IntegrityError(f"{f['path']} differs from the release root")
        return body

    @property
    def members(self) -> dict[str, dict[str, Any]]:
        if self._members is None:
            import io
            t = pq.read_table(io.BytesIO(self._meta_bytes("members")), use_threads=False,
                              columns=["record_id", "object_path", "member_path", "data_offset",
                                       "member_bytes", "member_sha256"])
            rows = t.to_pylist()
            for r in rows:
                safe_relpath(r["object_path"])
                safe_relpath(r["member_path"])
            self._members = {r["record_id"]: r for r in rows}
        return self._members

    @property
    def snapshot(self) -> C.Snapshot:
        if self._snapshot is None:
            absent = [x["path"] for x in self.root.files if x["kind"] == "schema" and not (self.root.path / x["path"]).is_file()]
            if absent:
                raise NotDownloadedError(f"shared schema file(s) not downloaded: {', '.join(absent)}", absent)
            f = self.root.group_meta_file(self.name, "snapshot")
            seen = self.root._validated_snapshots
            self._snapshot = C.Snapshot(self._meta_bytes("snapshot"), f["sha256"], self.root.path,
                                        companion=self.root._companion, trusted=f["sha256"] in seen)
            seen.add(f["sha256"])
            self.root._companion = self._snapshot.companion
        return self._snapshot

    def records_of_scene(self, scene_id: str) -> dict[tuple[str, int | None, int | None], str]:
        """(component, frequency_hz, bs_index) -> record_id for included numeric records."""
        if self._by_scene is None:
            idx: dict[str, dict[tuple[str, int | None, int | None], str]] = {}
            for rid, comp, sid, hz, bs in self.snapshot.identities:
                if self.root.is_included(rid):
                    idx.setdefault(sid, {})[(comp, hz, bs)] = rid
            self._by_scene = idx
        return self._by_scene.get(scene_id, {})

    # ------------------------------------------------------------------ cache parts
    def identity(self, record_id: str) -> tuple[str, int | None]:
        """(component, frequency_hz) of an included numeric record."""
        if self._ident is None:
            self._ident = {rid: (comp, hz) for rid, comp, _sid, hz, _bs in self.snapshot.identities}
        return self._ident[record_id]

    def _part_rows(self, comp: str, hz: int | None) -> list[dict[str, Any]]:
        return [f for f in self.root.group_files.get(self.name, ())
                if f["kind"] in ("cache_index", "cache_array") and f["component"] == comp and f["frequency_hz"] == hz]

    def part_paths(self, record_id: str) -> list[str] | None:
        """Files of the cache part holding ``record_id`` (from the root tables; nothing is read)."""
        rows = self._part_rows(*self.identity(record_id))
        return [f["path"] for f in rows] or None

    def part(self, record_id: str) -> "CachePart | None":
        """The cache part for this record's component and carrier, loading only that part's index."""
        key = self.identity(record_id)
        if key not in self._parts:
            index = [f for f in self._part_rows(*key) if f["kind"] == "cache_index"]
            self._parts[key] = CachePart(self, index[0]) if index else None  # stored only if construction succeeds
        part = self._parts[key]
        return part if part is not None and record_id in part else None

    # ------------------------------------------------------------------ reading
    def read_numeric(self, record_id: str, verify: str) -> dict[str, np.ndarray]:
        row = self.snapshot.row(record_id)
        part = self.part(record_id) if self.root.kind == "cache" else None
        if part is not None:
            arrays = part.arrays(record_id, row)
            if verify == "full":
                C.check_full(row, arrays, self.snapshot.companion)
            elif verify == "basic":
                C.check_basic_arrays(row, arrays)
            return arrays
        payload = self.root.read_bytes(self.root.locate(self, record_id), verify)
        return C.decode_npz(payload, row, self.snapshot.companion, verify)

    def read_member(self, record_id: str, verify: str) -> bytes:
        """Raw published member bytes (used for geometry ZIPs)."""
        loc = self.root.locate(self, record_id)
        body = self.root.read_bytes(loc, verify)
        if verify != "none" and C.sha256(body) != self.members[record_id]["member_sha256"]:
            raise IntegrityError(f"{loc.path}: bytes differ from the published member")
        return body


class CachePart:
    """One cache part: concatenated NPY arrays for the selected records of one source archive."""

    def __init__(self, group: Group, index_file: dict[str, Any]) -> None:
        self.group = group
        self.index_path = index_file["path"]
        p = group.root.path / index_file["path"]
        try:
            body = p.read_bytes()
        except FileNotFoundError:
            raise NotDownloadedError(f"cache index {index_file['path']} is missing", [index_file["path"]]) from None
        if index_file["sha256"] and C.sha256(body) != index_file["sha256"]:
            raise IntegrityError(f"{index_file['path']} differs from the cache root")
        idx = json.loads(body)
        self.component: str = idx["component"]
        self.record_ids: list[str] = list(idx["records"])
        self.fields: dict[str, str] = dict(idx["fields"])
        self._ordinal = {rid: i for i, rid in enumerate(self.record_ids)}
        self.paths = [index_file["path"], *self.fields.values()]
        self._maps: dict[str, np.ndarray] | None = None
        self._starts: dict[str, np.ndarray] | None = None

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._ordinal

    def _open(self) -> None:
        if self._maps is not None:
            return
        snap = self.group.snapshot
        lengths = {name: [] for name in C.FIELDS[self.component]}
        for rid in self.record_ids:
            for name, shape in C.shapes(snap.row(rid)["record"]).items():
                lengths[name].append(shape[0])
        maps, starts = {}, {}
        for name, rel in self.fields.items():
            path = self.group.root.path / rel
            try:
                m = np.load(path, mmap_mode="r", allow_pickle=False)
            except FileNotFoundError:
                raise NotDownloadedError(f"cache array {rel} is missing", [rel]) from None
            total = int(sum(lengths[name]))
            if m.dtype.str != C.FIELDS[self.component][name] or m.shape[0] != total:
                raise IntegrityError(f"cache array {rel} does not match its index")
            maps[name] = m
            starts[name] = np.concatenate([[0], np.cumsum(lengths[name], dtype=np.int64)])
        self._maps, self._starts = maps, starts

    def arrays(self, record_id: str, row: dict[str, Any]) -> dict[str, np.ndarray]:
        self._open()
        assert self._maps is not None and self._starts is not None
        i = self._ordinal[record_id]
        return {name: m[self._starts[name][i]:self._starts[name][i + 1]] for name, m in self._maps.items()}
