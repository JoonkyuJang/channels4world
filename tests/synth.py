"""Small synthetic Channels4World release in the published format:
uncompressed USTAR TARs of compressed v2 NPZ members, per-group catalogs and snapshot, shared
public schema documents, and a release root (``write_release_root``). All records satisfy the
published contracts unless asked otherwise.

Archive paths default to a nested layout; ``layout="release"`` uses the published paths. The
library locates archives through the catalogs, so tests read both alike.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from channels4world import _contract as C
from channels4world._paths import safe_relpath
from channels4world._root import FILES_SCHEMA, FORMAT, FORMAT_VERSION, ROOT_FILE, SCENES_SCHEMA, TABLE_DIR, Root
from channels4world.prepare import _table_bytes, _write_atomic

SCHEMA = Path(__file__).parent / "data" / "schema"
CID = "channels4world-test"
FREQS = (3_500_000_000, 7_000_000_000)
N_BS, N_UES = 3, 24
GROUPS = [  # archive names follow the confirmed release template; "inc" is the group's increment
    {"name": "g-a", "split": "train", "profile": "metropolitan", "scenes": ["a0", "a1"], "detail": True, "inc": "small"},
    {"name": "g-b", "split": "train", "profile": "suburban", "scenes": ["b0", "b1"], "detail": False, "inc": "medium"},
    {"name": "g-c", "split": "test", "profile": "urban", "scenes": ["c0"], "detail": True, "inc": "small"},
]
REALISTIC = {"g-a": "train-metro-g00000", "g-b": "train-suburban-g00001", "g-c": "test-urban-g00002"}
INC_DIR = {"small": "inc1-small", "medium": "inc2-medium", "large": "inc3-large", "full": "inc4-full"}
TIERS = {"a0": "tiny", "a1": "small", "b0": "medium", "b1": "full", "c0": "small"}


def _parquet(path: Path, rows: list[dict], schema: pa.Schema | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pylist(rows, schema=schema) if schema is not None else pa.Table.from_pylist(rows)
    pq.write_table(t, path, compression="zstd")


def _npz(arrays: dict, header: dict) -> bytes:
    out = io.BytesIO()
    np.savez_compressed(out, **arrays, __meta__=np.array(C.canonical(header).decode()))
    return out.getvalue()


def core_and_detail(rng: np.random.Generator, *, zero_links: bool = False, empty_link: bool = False):
    if zero_links:
        ue = np.array([], dtype=np.int32)
    else:
        ue = np.sort(rng.choice(N_UES, size=int(rng.integers(2, N_UES // 2)), replace=False)).astype(np.int32)
    counts = rng.integers(1, 5, size=len(ue))
    if empty_link and len(counts):
        counts[0] = 0
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    paths = int(off[-1])
    rank = np.concatenate([np.sort(rng.choice(16, size=int(c), replace=False)) for c in counts]
                          + [np.array([], dtype=np.int64)]).astype(np.uint8)
    n_int = rng.integers(0, 4, size=paths)
    types = [rng.choice([1, 2, 4, 8], size=int(n)) for n in n_int]
    flags = np.array([int(np.bitwise_or.reduce(1 << t.astype(np.int64))) if len(t) else 0 for t in types],
                     dtype=np.uint16)
    f32 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    core = {
        "link_ue_index": ue, "link_path_offsets": off, "path_rank": rank,
        "a_real": f32(rng.normal(0, 1e-4, paths)), "a_imag": f32(rng.normal(0, 1e-4, paths)),
        "tau": f32(rng.uniform(1e-7, 2e-6, paths)),
        "theta_t": f32(rng.uniform(0, np.pi, paths)), "phi_t": f32(rng.uniform(-np.pi, np.pi, paths)),
        "theta_r": f32(rng.uniform(0, np.pi, paths)), "phi_r": f32(rng.uniform(-np.pi, np.pi, paths)),
        "interaction_count": n_int.astype(np.uint8), "interaction_flags": flags,
    }
    pio = np.concatenate([[0], np.cumsum(n_int)]).astype(np.int64)
    total = int(pio[-1])
    detail = {
        "path_interaction_offsets": pio,
        "interaction_rank": np.concatenate([np.sort(rng.choice(10, size=int(n), replace=False)) for n in n_int]
                                           + [np.array([], dtype=np.int64)]).astype(np.uint8),
        "interaction_type": (np.concatenate(types + [np.array([], dtype=np.int64)])).astype(np.int8),
        "interaction_vertex": f32(rng.uniform(-50, 50, (total, 3))),
    }
    return core, detail


def make_release(root: Path, *, seed: int = 0, bad_core: tuple[str, int, int] | None = None,
                 zero_link: tuple[str, int, int] | None = None, hf_repo: str = "org/Channels4World",
                 hf_path: str = "", layout: str = "nested", realistic_names: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SCHEMA, root / "schema", dirs_exist_ok=True)
    conv = (root / "schema/conventions.json").read_bytes()
    cref = {"ref": "schema/conventions.json", "sha256": C.sha256(conv)}
    band = {3_500_000_000: "3.5ghz", 7_000_000_000: "7ghz"}
    for g in GROUPS:
        name, split = (REALISTIC[g["name"]] if realistic_names else g["name"]), g["split"]
        inc = g["inc"]
        if layout == "nested":    # payload/main/<band|shared>/<component>/inc-<inc>/<split>/<file>
            def obj(top, comp, fname, inc=inc, split=split):
                return f"payload/main/{top}/{comp}/inc-{inc}/{split}/{fname}"
        elif layout != "release":
            raise ValueError(f"unknown layout {layout!r}")
        else:                     # published: payload/<band|scene>/<component>/inc<N>-<inc>/<file>
            def obj(top, comp, fname, inc=inc):
                top = "scene" if top == "shared" else top
                return f"payload/{top}/{comp}/{INC_DIR[inc]}/{fname}"
        objects = {"positions": obj("shared", "positions", f"{name}-positions.tar"),
                   "geometry": obj("shared", "geometry", f"{name}-geometry.tar")}
        for hz in FREQS:
            for comp in (("core", "detail") if g["detail"] else ("core",)):
                objects[f"{comp}-{hz}"] = obj(band[hz], comp, f"{name}-{band[hz]}-{comp}.tar")
        members: dict[str, list[tuple[str, str, bytes]]] = {k: [] for k in objects}
        rows = []
        for sid in g["scenes"]:
            pos = {"bs_positions": rng.uniform(-100, 100, (N_BS, 3)).astype(np.float32),
                   "ue_positions": rng.uniform(-100, 100, (N_UES, 3)).astype(np.float32)}
            ph = {"schema": "wireless.npz-meta", "version": 2, "conventions": cref, "info": {},
                  "identity": {"collection_id": CID, "scene_id": sid, "component": "positions",
                               "frequency_hz": None, "bs_index": None},
                  "dimensions": {"bs": N_BS, "ues": N_UES}}
            pman = C.numeric_manifest("positions", pos)
            prid = C.logical_id(ph["identity"])
            pbody = _npz(pos, ph)
            members["positions"].append((prid, f"records/scene={sid}/positions.npz", pbody))
            rows.append({"record_id": prid, "record": ph, "numeric_content": pman,
                         "content_sha256": C.content_digest(pman), "member_sha256": C.sha256(pbody)})
            pin = {"record_id": prid, "content_sha256": C.content_digest(pman), "conventions_sha256": cref["sha256"],
                   "bs": N_BS, "ues": N_UES}
            gz = io.BytesIO()
            with zipfile.ZipFile(gz, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("meshes/ground.obj", f"# {sid}\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
                z.writestr("scene.xml", '<scene version="2.1.0"><shape type="obj">'
                                        '<string name="filename" value="meshes/ground.obj"/></shape></scene>')
                z.writestr("scene.json", json.dumps({"entrypoint": "scene.xml", "schema": "wireless.scene-geometry"}))
            members["geometry"].append((C.geometry_record_id(CID, sid), f"records/scene={sid}/geometry.zip",
                                        gz.getvalue()))
            for hz in FREQS:
                for bs in range(N_BS):
                    key = (sid, hz, bs)
                    core, det = core_and_detail(rng, zero_links=key == zero_link, empty_link=key == bad_core)
                    ident = {"collection_id": CID, "scene_id": sid, "component": "core", "frequency_hz": hz,
                             "bs_index": bs}
                    axis = C.path_axis(core)
                    ch = {"schema": "wireless.npz-meta", "version": 2, "conventions": cref, "info": {},
                          "identity": ident, "dimensions": {"links": len(core["link_ue_index"]),
                                                            "paths": len(core["path_rank"])},
                          "path_axis_sha256": axis, "dependencies": {"positions": pin}}
                    crid = C.logical_id(ident)
                    cbody = _npz(core, ch)
                    cman = C.numeric_manifest("core", core)
                    members[f"core-{hz}"].append((crid, C_path(sid, hz, "core", bs), cbody))
                    rows.append({"record_id": crid, "record": ch, "numeric_content": cman,
                                 "content_sha256": C.content_digest(cman), "member_sha256": C.sha256(cbody)})
                    if g["detail"]:
                        dident = dict(ident, component="detail")
                        dh = {"schema": "wireless.npz-meta", "version": 2, "conventions": cref, "info": {},
                              "identity": dident,
                              "dimensions": {"paths": len(core["path_rank"]),
                                             "interactions": len(det["interaction_type"])},
                              "path_axis_sha256": axis,
                              "dependencies": {"core_axis": {"record_id": crid, "path_axis_sha256": axis,
                                                             "conventions_sha256": cref["sha256"], "positions": pin}}}
                        dbody = _npz(det, dh)
                        dman = C.numeric_manifest("detail", det)
                        members[f"detail-{hz}"].append((C.logical_id(dident), C_path(sid, hz, "detail", bs), dbody))
                        rows.append({"record_id": C.logical_id(dident), "record": dh, "numeric_content": dman,
                                     "content_sha256": C.content_digest(dman), "member_sha256": C.sha256(dbody)})
        member_rows, object_rows = [], []
        for key, rel in objects.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as fh, tarfile.open(fileobj=fh, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for rid, mpath, body in members[key]:
                    info = tarfile.TarInfo(mpath)
                    info.size, info.mode, info.mtime = len(body), 0o644, 0
                    off = tar.offset + 512
                    tar.addfile(info, io.BytesIO(body))
                    member_rows.append({"record_id": rid, "object_path": rel, "member_path": mpath, "data_offset": off,
                                        "member_bytes": len(body), "member_sha256": C.sha256(body)})
            body = path.read_bytes()
            object_rows.append({"object_path": rel, "object_bytes": len(body), "object_sha256": C.sha256(body)})
        gdir = root / "groups" / name
        ms = pa.schema([(k, pa.int64() if k in ("data_offset", "member_bytes") else pa.string(), False)
                        for k in ("record_id", "object_path", "member_path", "data_offset", "member_bytes",
                                  "member_sha256")])
        _parquet(gdir / "catalog/members.parquet", member_rows, ms)
        _parquet(gdir / "catalog/objects.parquet", object_rows)
        _parquet(gdir / "catalog/scenes.parquet", [{"scene_id": s, "scene_order": i, "split": split,
                                                    "profile": g["profile"]} for i, s in enumerate(g["scenes"])])
        snap = C.canonical({"schema": "wireless.numeric-snapshot", "version": 1, "conventions": cref, "records": rows})
        (gdir / "snapshot.json").write_bytes(snap)
    write_release_root(root, tiers=TIERS, release="test-1", hf_repo=hf_repo, hf_path=hf_path)
    return root


TIER_ORDER = ("tiny", "small", "medium", "large", "full")


def write_release_root(root_dir: Path, *, tiers: dict[str, str], release: str, hf_repo: str | None = None,
                       hf_path: str | None = None) -> Path:
    """Write the release root (``c4w-root.json`` and its scenes and files tables) for the
    synthetic groups under ``root_dir/groups``, in the published root format. Test fixture."""
    root = Path(root_dir).resolve()
    names = sorted(p.name for p in (root / "groups").iterdir() if (p / "snapshot.json").is_file())
    scene_rows, file_rows, collection, frequencies, companion = [], [], set(), set(), None
    for gi, name in enumerate(names):
        gdir = f"groups/{name}"
        snap_body = (root / gdir / "snapshot.json").read_bytes()
        snap = C.Snapshot(snap_body, C.sha256(snap_body), root, companion=companion)
        companion = snap.companion
        kinds: dict[str, tuple[str, int | None]] = {}
        for rid, comp, _sid, hz, _bs in snap.identities:
            kinds[rid] = (comp, hz)
            collection.add(snap.row(rid)["record"]["identity"]["collection_id"])
            if hz is not None:
                frequencies.add(hz)
        scenes = pq.read_table(root / gdir / "catalog/scenes.parquet").to_pylist()
        for s in scenes:
            scene_rows.append({"scene_id": s["scene_id"], "split": s["split"], "profile": s["profile"],
                               "tier": tiers[s["scene_id"]], "group": name, "group_order": gi,
                               "scene_order": s["scene_order"]})
        (cid,) = collection
        for s in scenes:
            kinds[C.geometry_record_id(cid, s["scene_id"])] = ("geometry", None)
        obj_kind = {m["object_path"]: kinds[m["record_id"]]
                    for m in pq.read_table(root / gdir / "catalog/members.parquet").to_pylist()}
        for f in sorted((root / gdir).rglob("*")):
            if f.is_file():
                body = f.read_bytes()
                kind = ("snapshot" if f.name == "snapshot.json" else "members" if f.name == "members.parquet"
                        else "catalog")
                file_rows.append({"path": f.relative_to(root).as_posix(), "bytes": len(body),
                                  "sha256": C.sha256(body), "group": name, "kind": kind, "component": None,
                                  "frequency_hz": None, "record_id": None})
        for o in pq.read_table(root / gdir / "catalog/objects.parquet").to_pylist():
            comp, hz = obj_kind[o["object_path"]]
            file_rows.append({"path": o["object_path"], "bytes": o["object_bytes"], "sha256": o["object_sha256"],
                              "group": name, "kind": "payload", "component": comp, "frequency_hz": hz,
                              "record_id": None})
    for f in sorted((root / "schema").glob("*.json")):
        body = f.read_bytes()
        file_rows.append({"path": f"schema/{f.name}", "bytes": len(body), "sha256": C.sha256(body), "group": None,
                          "kind": "schema", "component": None, "frequency_hz": None, "record_id": None})
    tables = {}
    for tname, rows, schema in (("scenes", scene_rows, SCENES_SCHEMA), ("files", file_rows, FILES_SCHEMA)):
        body = _table_bytes(rows, schema)
        rel = f"{TABLE_DIR}/{tname}.parquet"
        _write_atomic(root, rel, body)
        tables[tname] = {"path": rel, "sha256": C.sha256(body), "bytes": len(body)}
    manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "kind": "release", "payload": "tar",
                "collection_id": collection.pop(), "release": release, "tiers": list(TIER_ORDER),
                "frequencies_hz": sorted(frequencies), "hf_repo": hf_repo, "hf_path": hf_path, "tables": tables,
                "source": None}
    _write_atomic(root, ROOT_FILE, json.dumps(manifest, indent=1, sort_keys=True).encode())
    return root / ROOT_FILE


def set_root_publication(root_dir: Path, *, hf_repo: str, hf_path: str) -> Path:
    """Record where a root (for example an extracted Tiny) sits in a Hugging Face repository,
    as a published root does. Manifest only. Test fixture."""
    if hf_path:
        safe_relpath(hf_path)
    root = Root(root_dir)
    manifest = dict(root.manifest, hf_repo=hf_repo, hf_path=hf_path)
    _write_atomic(root.path, ROOT_FILE, json.dumps(manifest, indent=1, sort_keys=True).encode())
    return root.path / ROOT_FILE


def C_path(sid: str, hz: int, comp: str, bs: int) -> str:
    return f"records/scene={sid}/band_hz={hz}/component={comp}/bs{bs:05d}.npz"
