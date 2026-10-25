"""Channels4World v2 record contract: identities, digests and checks for published records.

Scientific predicates, header rules, signatures and digests follow the published schema
documents (``schema/``). Implementation notes:

* ``Snapshot`` keeps an immutable per-record index after its construction-time validation,
  instead of re-parsing the whole snapshot on every lookup.
* Decoding is split into ``verify`` levels (``none`` / ``basic`` / ``full``); ``full`` runs
  every check.
* ``validate_detail`` is vectorized (tests compare it with a straightforward loop).
* Arrays are returned as read-only NumPy arrays without an extra byte copy.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ._errors import FormatError, IntegrityError

VERIFY_LEVELS = ("none", "basic", "full")

FIELDS: dict[str, dict[str, str]] = {
    "core": {
        "link_ue_index": "<i4",
        "link_path_offsets": "<i8",
        "path_rank": "|u1",
        **{k: "<f4" for k in ("a_real", "a_imag", "tau", "theta_t", "phi_t", "theta_r", "phi_r")},
        "interaction_count": "|u1",
        "interaction_flags": "<u2",
    },
    "detail": {
        "path_interaction_offsets": "<i8",
        "interaction_rank": "|u1",
        "interaction_type": "|i1",
        "interaction_vertex": "<f4",
    },
    "positions": {"bs_positions": "<f4", "ue_positions": "<f4"},
}


def check_verify(verify: str) -> str:
    if verify not in VERIFY_LEVELS:
        raise ValueError(f"verify must be one of {VERIFY_LEVELS}, not {verify!r}")
    return verify


# ----------------------------------------------------------------------------- JSON / IDs
def strict_json(payload: bytes | str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError("nonstandard JSON constant")

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def logical_id(identity: dict[str, Any]) -> str:
    """Published record ID: SHA-256 of the canonical identity JSON."""
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                   allow_nan=False).encode()
    ).hexdigest()


def geometry_record_id(collection_id: str, scene_id: str) -> str:
    return logical_id({"bs_index": None, "collection_id": collection_id, "component": "geometry",
                       "frequency_hz": None, "scene_id": scene_id})


def _hash(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256")


def _integer(value: Any) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("expected nonnegative integer")


def path_axis(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"path-axis-v1\0")
    for name in ("link_ue_index", "link_path_offsets", "path_rank"):
        a = arrays[name]
        header = json.dumps({"field": name, "dtype": a.dtype.str, "shape": list(a.shape)},
                            sort_keys=True, separators=(",", ":")).encode()
        body = a.tobytes(order="C")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(body).to_bytes(8, "little"))
        digest.update(body)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- scientific rules
ANGLE_TOLERANCE_RAD = 1e-6
THETA_INTERVAL_F32 = (np.float32(-ANGLE_TOLERANCE_RAD), np.float32(math.pi + ANGLE_TOLERANCE_RAD))
PHI_INTERVAL_F32 = (np.float32(-math.pi - ANGLE_TOLERANCE_RAD), np.float32(math.pi + ANGLE_TOLERANCE_RAD))
RELEASE_PROFILE_SCHEMA = "channels4world.release-profile"
_FINITE = {
    "core": ["a_imag", "a_real", "phi_r", "phi_t", "tau", "theta_r", "theta_t"],
    "detail": ["interaction_vertex"],
    "positions": ["bs_positions", "ue_positions"],
}


@dataclass(frozen=True)
class ReleasePolicy:
    """Named release retention rules (the release-profile document); scientific validation only."""

    name: str
    sha256: str
    top_k: int
    nonempty_links: bool

    @classmethod
    def from_declaration(cls, body: bytes) -> "ReleasePolicy":
        if not isinstance(body, bytes) or len(body) > 65536:
            raise ValueError("release profile declaration bytes")
        p = strict_json(body)
        g = p["generic_interpretation"]
        rules = p["release_rules"]
        angles = g["angles"]
        if (
            p["schema"] != RELEASE_PROFILE_SCHEMA
            or type(p["version"]) is not int
            or p["version"] != 1
            or not isinstance(p["name"], str)
            or g["finite_float_fields"] != _FINITE
            or g["delay"]["field"] != "tau"
            or g["delay"]["minimum_seconds"] != 0
            or g["delay"]["inclusive"] is not True
            or angles["tolerance_rad"] != ANGLE_TOLERANCE_RAD
            or angles["theta_interval"] != ["-tolerance", "pi+tolerance"]
            or angles["phi_interval"] != ["-pi-tolerance", "pi+tolerance"]
            or g["interaction_codes"] != [1, 2, 4, 8]
            or rules["top_k"] != 16
            or rules["max_paths_per_link"] != rules["top_k"]
            or rules["path_rank_exclusive_upper"] != rules["top_k"]
            or rules["retained_links"] != "nonempty"
        ):
            raise ValueError("unsupported release profile declaration")
        return cls(p["name"], sha256(body), rules["top_k"], True)


def _rows(a: Any) -> int:
    return int(a.shape[0]) if a.ndim else -1


def _segments_strictly_increase(values: Any, offsets: Any) -> bool:
    n = len(values)
    if n < 2:
        return True
    starts = np.zeros(n, dtype=bool)
    lengths = np.diff(np.asarray(offsets, dtype=np.int64))
    starts[np.asarray(offsets[:-1], dtype=np.int64)[lengths > 0]] = True
    v = np.asarray(values, dtype=np.int64)
    return not bool(np.any((v[1:] <= v[:-1]) & ~starts[1:]))


def _csr(offsets: Any, endpoint: int) -> None:
    if (
        len(offsets) == 0
        or offsets[0] != 0
        or offsets[-1] != endpoint
        or np.any(offsets < 0)
        or np.any(offsets > endpoint)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("CSR boundaries")


def validate_numeric_arrays(component: str, arrays: Mapping[str, Any], *, ues: int | None = None,
                            bs_index: int | None = None, bs_count: int | None = None) -> None:
    """Generic scientific predicates for a decoded numeric record."""
    if component not in FIELDS or set(arrays) != set(FIELDS[component]):
        raise ValueError("exact declared numeric fields required")
    for name, a in arrays.items():
        if not isinstance(a, np.ndarray) or a.dtype.str != FIELDS[component][name]:
            raise ValueError("declared numeric dtype mismatch")
    if any(a.dtype.kind == "f" and not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite numeric field")
    if component == "positions":
        leading = {n: (_rows(a), 3) for n, a in arrays.items()}
    elif component == "core":
        paths = _rows(arrays["path_rank"])
        links = _rows(arrays["link_ue_index"])
        leading = {n: (links + 1,) if n == "link_path_offsets" else (links,) if n == "link_ue_index"
                   else (paths,) for n in arrays}
    else:
        ints = _rows(arrays["interaction_type"])
        paths = _rows(arrays["path_interaction_offsets"]) - 1
        leading = {n: (paths + 1,) if n == "path_interaction_offsets" else (ints, 3)
                   if n == "interaction_vertex" else (ints,) for n in arrays}
    if any(a.shape != leading[n] for n, a in arrays.items()):
        raise ValueError("numeric dimensions mismatch")
    if component == "core":
        if ues is None or bs_index is None or bs_count is None:
            raise ValueError("core validation requires positions axes")
        off = arrays["link_path_offsets"]
        ranks = arrays["path_rank"]
        ue = arrays["link_ue_index"].astype(np.int64)
        _csr(off, len(ranks))
        if len(off) != len(ue) + 1:
            raise ValueError("CSR boundaries")
        if np.any(ue < 0) or np.any(ue >= ues) or np.any(ue[1:] <= ue[:-1]) or not 0 <= bs_index < bs_count:
            raise ValueError("UE/BS association")
        if not _segments_strictly_increase(ranks, off):
            raise ValueError("source rank association")
        if np.any(arrays["tau"] < 0):
            raise ValueError("negative delay")
        lo, hi = THETA_INTERVAL_F32
        for n in ("theta_t", "theta_r"):
            if np.any(arrays[n] < lo) or np.any(arrays[n] > hi):
                raise ValueError("zenith boundary")
        lo, hi = PHI_INTERVAL_F32
        for n in ("phi_t", "phi_r"):
            if np.any(arrays[n] < lo) or np.any(arrays[n] > hi):
                raise ValueError("azimuth boundary")
        if np.any(arrays["interaction_flags"] & np.uint16(~278 & 65535)) or np.any(
            (arrays["interaction_count"] == 0) != (arrays["interaction_flags"] == 0)
        ):
            raise ValueError("core interaction flags")
    elif component == "detail":
        off = arrays["path_interaction_offsets"]
        _csr(off, len(arrays["interaction_type"]))
        if not np.isin(arrays["interaction_type"], [1, 2, 4, 8]).all():
            raise ValueError("interaction code")
        if not _segments_strictly_increase(arrays["interaction_rank"], off):
            raise ValueError("interaction order")


def validate_release_policy(component: str, arrays: Mapping[str, Any], policy: ReleasePolicy) -> None:
    if component != "core":
        return
    lengths = np.diff(np.asarray(arrays["link_path_offsets"], dtype=np.int64))
    ranks = arrays["path_rank"]
    if policy.nonempty_links and np.any(lengths == 0):
        raise ValueError("release profile retained-empty link")
    if np.any(lengths > policy.top_k) or (ranks.size and int(ranks.max()) >= policy.top_k):
        raise ValueError("release profile top-k")


def numeric_manifest(component: str, arrays: Mapping[str, Any]) -> dict[str, Any]:
    if component not in FIELDS or set(arrays) != set(FIELDS[component]):
        raise ValueError("exact declared numeric fields required")
    entries = []
    for name in sorted(arrays):
        a = arrays[name]
        if not isinstance(a, np.ndarray) or a.dtype.str != FIELDS[component][name]:
            raise ValueError("declared numeric dtype mismatch")
        entries.append({"name": name, "dtype": a.dtype.str, "shape": list(a.shape),
                        "c_order_sha256": sha256(a.tobytes(order="C"))})
    return {"schema": "wireless.numeric-content", "version": 1, "component": component, "arrays": entries}


def content_digest(manifest: dict[str, Any]) -> str:
    if (
        set(manifest) != {"schema", "version", "component", "arrays"}
        or manifest["schema"] != "wireless.numeric-content"
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
        or manifest["component"] not in FIELDS
    ):
        raise ValueError("unsupported numeric manifest")
    names = []
    for entry in manifest["arrays"]:
        if set(entry) != {"name", "dtype", "shape", "c_order_sha256"}:
            raise ValueError("numeric signature fields")
        names.append(entry["name"])
        if entry["dtype"] != FIELDS[manifest["component"]].get(entry["name"]):
            raise ValueError("numeric signature dtype")
        if not isinstance(entry["shape"], list):
            raise ValueError("signature shape")
        for value in entry["shape"]:
            _integer(value)
        _hash(entry["c_order_sha256"])
    if names != sorted(FIELDS[manifest["component"]]):
        raise ValueError("exact sorted signature membership required")
    return sha256(b"wireless.numeric-content.v1\0" + canonical(manifest))


def validate_detail(detail: Mapping[str, Any], core: Mapping[str, Any]) -> None:
    """Interaction counts and flags of a detail record agree with its core record (vectorized)."""
    offsets = np.asarray(detail["path_interaction_offsets"], dtype=np.int64)
    counts = np.diff(offsets)
    if counts.shape != core["interaction_count"].shape or not np.array_equal(
            counts, core["interaction_count"].astype(np.int64)):
        raise ValueError("interaction count association mismatch")
    types = np.asarray(detail["interaction_type"], dtype=np.int64)
    if types.size and (types.min() < 0 or types.max() > 15):
        raise ValueError("interaction code")
    flags = np.zeros(len(counts), dtype=np.int64)
    nonempty = counts > 0
    if types.size:
        flags[nonempty] = np.bitwise_or.reduceat(np.left_shift(1, types), offsets[:-1][nonempty])
    if not np.array_equal(flags, core["interaction_flags"].astype(np.int64)):
        raise ValueError("interaction flags association mismatch")


# ----------------------------------------------------------------------------- interpretation docs
class Companion:
    """Resolve the hash-pinned interpretation documents once; hold immutable bytes."""

    def __init__(self, metadata_root: Path, reference: dict[str, Any]) -> None:
        root = Path(metadata_root).resolve(strict=True)

        def resolve(ref: dict[str, Any]) -> bytes:
            if set(ref) != {"ref", "sha256"}:
                raise ValueError("companion reference fields")
            _hash(ref["sha256"])
            name = ref["ref"]
            if not isinstance(name, str) or not name or "\\" in name:
                raise ValueError("relative companion reference required")
            if PurePosixPath(name).is_absolute() or any(x in ("", ".", "..") for x in name.split("/")):
                raise ValueError("unsafe companion reference")
            # The reference is checked as written (relative, no "..") and may be a symbolic
            # link, as in the Hugging Face cache; its content is pinned by SHA-256 below.
            target = root / name
            if not target.is_file():
                raise FileNotFoundError(f"interpretation document {name} not found")
            with target.open("rb") as stream:
                body = stream.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024 or sha256(body) != ref["sha256"]:
                raise ValueError("companion size/hash mismatch")
            return body

        self._reference = canonical(reference)
        self._conventions = resolve(reference)
        conventions = strict_json(self._conventions)
        if (conventions["schema"] != "wireless.channel-conventions" or type(conventions["version"]) is not int
                or conventions["version"] != 2):
            raise ValueError("unsupported conventions")
        if (conventions["angles"].get("acceptance_tolerance_rad") != ANGLE_TOLERANCE_RAD
                or conventions["angles"].get("acceptance_comparison")
                != "float32_values_vs_float32_rounded_closed_interval_endpoints"):
            raise ValueError("unsupported angle acceptance tolerance")
        self._policy = resolve(conventions["release_profile"])
        self.release_policy = ReleasePolicy.from_declaration(self._policy)
        self._arrays = resolve(conventions["arrays"])
        self._codes = resolve(conventions["interaction_codes"])
        arrays = strict_json(self._arrays)
        if (arrays["schema"] != "wireless.arrays" or arrays["metadata_version"] != 2
                or arrays["metadata_schema"] != "wireless.npz-meta" or arrays["metadata_key"] != "__meta__"
                or arrays["fields"] != FIELDS or arrays["capabilities"] != {"retained_empty_links": True}):
            raise ValueError("unsupported arrays contract")
        expected_masks = {"1": 2, "2": 4, "4": 16, "8": 256}
        if (conventions["interaction_flags"]["masks"] != expected_masks
                or conventions["interaction_flags"]["encoding"] != "bitwise_OR_of_1_shifted_left_by_code"):
            raise ValueError("unsupported flag interpretation")
        codebook = strict_json(self._codes)
        if codebook["codes"] != {"0": "none", "1": "specular_reflection", "2": "diffuse_reflection",
                                 "4": "refraction", "8": "diffraction"}:
            raise ValueError("unsupported interaction codebook")
        if (
            conventions["coordinates"] != {"frame": "scene_cartesian", "georeferencing": "not_declared",
                                           "handedness": "right_handed", "position_unit": "metre", "z_axis": "up"}
            or conventions["delay"] != {"field": "tau", "meaning": "propagation_delay_not_delay_spread",
                                        "unit": "second"}
            or conventions["angles"]["unit"] != "radian"
            or conventions["angles"]["theta"] != "zenith_from_positive_z"
            or conventions["angles"]["phi"] != "azimuth_from_positive_x_toward_positive_y"
            or conventions["angles"]["t"] != "departure_at_BS"
            or conventions["angles"]["r"] != "arrival_at_UE"
            or conventions["coefficient"]["unit"] != "dimensionless"
            or conventions["coefficient"]["numeric_representation"] != "a_real_plus_i_a_imag"
            or conventions["interaction_order"] != "BS_to_UE"
            or conventions["retention"]["top_k"] != 16
            or self.release_policy.top_k != conventions["retention"]["top_k"]
        ):
            raise ValueError("unsupported scientific interpretation")

    @property
    def reference(self) -> dict[str, Any]:
        return strict_json(self._reference)

    @property
    def conventions(self) -> dict[str, Any]:
        return strict_json(self._conventions)


def _positions_pin(pin: dict[str, Any]) -> None:
    if set(pin) != {"record_id", "content_sha256", "conventions_sha256", "bs", "ues"}:
        raise ValueError("positions dependency fields")
    for key in ("record_id", "content_sha256", "conventions_sha256"):
        _hash(pin[key])
    for key in ("bs", "ues"):
        _integer(pin[key])


def validate_header(header: dict[str, Any], companion: Companion) -> None:
    component = header["identity"]["component"]
    if component not in FIELDS:
        raise ValueError("unsupported component")
    fields = {"schema", "version", "identity", "conventions", "dimensions", "info"}
    if component != "positions":
        fields |= {"path_axis_sha256", "dependencies"}
        _hash(header["path_axis_sha256"])
    if (set(header) != fields or header["schema"] != "wireless.npz-meta" or type(header["version"]) is not int
            or header["version"] != 2 or header["conventions"] != companion.reference
            or not isinstance(header["info"], dict)):
        raise ValueError("v2 header/interpretation mismatch")
    identity = header["identity"]
    if set(identity) != {"collection_id", "scene_id", "component", "frequency_hz", "bs_index"}:
        raise ValueError("identity fields")
    for key in ("collection_id", "scene_id"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValueError("identity string")
    if component == "positions":
        if identity["frequency_hz"] is not None or identity["bs_index"] is not None:
            raise ValueError("positions identity must be shared")
    else:
        _integer(identity["frequency_hz"])
        _integer(identity["bs_index"])
        if identity["frequency_hz"] == 0:
            raise ValueError("frequency must be positive")
    dims = {"core": {"links", "paths"}, "detail": {"paths", "interactions"}, "positions": {"bs", "ues"}}
    if set(header["dimensions"]) != dims[component]:
        raise ValueError("dimension fields")
    for value in header["dimensions"].values():
        _integer(value)
    dep: dict[str, Any] = {}
    if component == "core":
        if set(header["dependencies"]) != {"positions"}:
            raise ValueError("core requires positions binding")
        _positions_pin(header["dependencies"]["positions"])
    elif component == "detail":
        if set(header["dependencies"]) != {"core_axis"}:
            raise ValueError("detail requires core association")
        dep = header["dependencies"]["core_axis"]
        if set(dep) != {"record_id", "path_axis_sha256", "conventions_sha256", "positions"}:
            raise ValueError("core association fields")
        for key in ("record_id", "path_axis_sha256", "conventions_sha256"):
            _hash(dep[key])
        _positions_pin(dep["positions"])
        if dep["path_axis_sha256"] != header["path_axis_sha256"]:
            raise ValueError("detail dependency axis mismatch")
        if (dep["record_id"] != logical_id(dict(identity, component="core"))
                or dep["conventions_sha256"] != companion.reference["sha256"]):
            raise ValueError("core dependency identity/interpretation mismatch")
    if component != "positions":
        pin = header["dependencies"]["positions"] if component == "core" else dep["positions"]
        expected_identity = dict(identity, component="positions", frequency_hz=None, bs_index=None)
        if (pin["record_id"] != logical_id(expected_identity)
                or pin["conventions_sha256"] != companion.reference["sha256"]):
            raise ValueError("positions dependency identity/interpretation mismatch")


def shapes(header: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    component = header["identity"]["component"]
    d = header["dimensions"]
    if component == "positions":
        return {"bs_positions": (d["bs"], 3), "ue_positions": (d["ues"], 3)}
    if component == "core":
        return {k: (d["links"] if k == "link_ue_index" else d["links"] + 1 if k == "link_path_offsets"
                    else d["paths"],) for k in FIELDS[component]}
    return {k: (d["paths"] + 1,) if k == "path_interaction_offsets" else (d["interactions"], 3)
            if k == "interaction_vertex" else (d["interactions"],) for k in FIELDS[component]}


# ----------------------------------------------------------------------------- snapshot
class Snapshot:
    """Hash-pinned per-group record catalog.

    Construction validates every row against the published contract. Afterwards each
    row is kept as immutable canonical bytes keyed by record ID, so lookups never re-parse
    the whole snapshot and callers can never mutate cached state.

    ``trusted=True`` skips the per-row validation. Use it only for bytes whose hash matches a
    snapshot that already passed full construction with the same interpretation documents
    (the root does this when a group is reloaded after cache eviction).
    """

    def __init__(self, payload: bytes, expected_sha256: str, metadata_root: Path,
                 companion: Companion | None = None, trusted: bool = False) -> None:
        _hash(expected_sha256)
        if not isinstance(payload, bytes) or sha256(payload) != expected_sha256:
            raise IntegrityError("snapshot hash differs from the release root")
        catalog = strict_json(payload)
        if (set(catalog) != {"schema", "version", "conventions", "records"}
                or catalog["schema"] != "wireless.numeric-snapshot" or type(catalog["version"]) is not int
                or catalog["version"] != 1):
            raise FormatError("unsupported snapshot")
        if companion is not None and canonical(catalog["conventions"]) == canonical(companion.reference):
            self.companion = companion
        else:
            self.companion = Companion(metadata_root, catalog["conventions"])
        ids: set[str] = set()
        for row in ([] if trusted else catalog["records"]):
            if set(row) != {"record_id", "record", "numeric_content", "content_sha256", "member_sha256"}:
                raise ValueError("catalog row fields")
            validate_header(row["record"], self.companion)
            if row["record_id"] != logical_id(row["record"]["identity"]) or row["record_id"] in ids:
                raise ValueError("catalog logical identity")
            ids.add(row["record_id"])
            _hash(row["member_sha256"])
            if (row["content_sha256"] != content_digest(row["numeric_content"])
                    or row["numeric_content"]["component"] != row["record"]["identity"]["component"]):
                raise ValueError("catalog numeric digest")
            sh = shapes(row["record"])
            if any(e["shape"] != list(sh[e["name"]]) for e in row["numeric_content"]["arrays"]):
                raise ValueError("catalog signature dimensions")
        self.sha256 = expected_sha256
        rows = {r["record_id"]: r for r in catalog["records"]}
        for row in ([] if trusted else rows.values()):
            header = row["record"]
            role = header["identity"]["component"]
            if role == "positions":
                continue
            pin = (header["dependencies"]["positions"] if role == "core"
                   else header["dependencies"]["core_axis"]["positions"])
            if pin["record_id"] in rows:
                positions = rows[pin["record_id"]]
                ph = positions["record"]
                declared = {"record_id": positions["record_id"], "content_sha256": positions["content_sha256"],
                            "conventions_sha256": ph["conventions"]["sha256"], **ph["dimensions"]}
                if ph["identity"]["component"] != "positions" or pin != declared:
                    raise ValueError("catalog positions dependency mismatch")
            if role == "detail":
                dep = header["dependencies"]["core_axis"]
                if dep["record_id"] in rows:
                    core = rows[dep["record_id"]]["record"]
                    declared_axis = {"record_id": dep["record_id"], "path_axis_sha256": core["path_axis_sha256"],
                                     "conventions_sha256": core["conventions"]["sha256"],
                                     "positions": core["dependencies"]["positions"]}
                    if dep != declared_axis or header["dimensions"]["paths"] != core["dimensions"]["paths"]:
                        raise ValueError("catalog core association mismatch")
        # Immutable index kept after validation.
        self._rows = {rid: canonical(row) for rid, row in rows.items()}
        self.identities = tuple(
            (rid, r["record"]["identity"]["component"], r["record"]["identity"]["scene_id"],
             r["record"]["identity"]["frequency_hz"], r["record"]["identity"]["bs_index"])
            for rid, r in rows.items()
        )

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._rows

    def row(self, record_id: str) -> dict[str, Any]:
        """A fresh copy of one validated row."""
        try:
            return strict_json(self._rows[record_id])
        except KeyError:
            raise KeyError(record_id) from None

    select = row  # alias of row()


# ----------------------------------------------------------------------------- decoding
def _readonly(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for a in arrays.values():
        a.flags.writeable = False
    return arrays


def decode_npz(payload: bytes, row: dict[str, Any], companion: Companion, verify: str) -> dict[str, np.ndarray]:
    """Decode one published NPZ member into read-only arrays at the requested verify level."""
    header = row["record"]
    component = header["identity"]["component"]
    if verify != "none":
        if sha256(payload) != row["member_sha256"]:
            raise IntegrityError(f"{_label(header)}: bytes differ from the published member")
        expected = shapes(header)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if (len(infos) != len({i.filename for i in infos})
                    or set(archive.namelist()) != {n + ".npy" for n in FIELDS[component]} | {"__meta__.npy"}
                    or sum(i.file_size for i in infos) > 256 * 1024**2):
                raise IntegrityError(f"{_label(header)}: unexpected NPZ entries")
            for info in infos:
                with archive.open(info) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise IntegrityError("unsupported NPY version")
                    name = info.filename[:-4]
                    if name == "__meta__":
                        if shape != () or dtype.kind != "U" or dtype.itemsize > 1024**2:
                            raise IntegrityError("NPY metadata header")
                    elif shape != expected[name] or dtype.str != FIELDS[component][name]:
                        raise IntegrityError(f"{_label(header)}: array {name} contradicts the catalog")
                    if stream.tell() + math.prod(shape) * dtype.itemsize != info.file_size:
                        raise IntegrityError("NPY body length mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if verify != "none" and strict_json(archive["__meta__"].item()) != header:
            raise IntegrityError(f"{_label(header)}: embedded header differs from the catalog")
        try:
            arrays = {name: archive[name] for name in FIELDS[component]}
        except KeyError as exc:
            raise IntegrityError(f"{_label(header)}: missing array {exc}") from None
    if verify == "full":
        check_full(row, arrays, companion)
    return _readonly(arrays)


def check_basic_arrays(row: dict[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    header = row["record"]
    component = header["identity"]["component"]
    expected = shapes(header)
    for name, dtype in FIELDS[component].items():
        a = arrays[name]
        if a.dtype.str != dtype or a.shape != expected[name]:
            raise IntegrityError(f"{_label(header)}: array {name} has unexpected dtype/shape")


def check_full(row: dict[str, Any], arrays: Mapping[str, np.ndarray], companion: Companion) -> None:
    """Every scientific and release check plus the published content proof."""
    header = row["record"]
    component = header["identity"]["component"]
    try:
        validate_header(header, companion)
        sh = shapes(header)
        if any(a.shape != sh[n] for n, a in arrays.items()):
            raise ValueError("numeric dimensions mismatch")
        if component == "core":
            if path_axis(arrays) != header["path_axis_sha256"]:
                raise ValueError("core path-axis mismatch")
            pin = header["dependencies"]["positions"]
            validate_numeric_arrays(component, arrays, ues=pin["ues"], bs_index=header["identity"]["bs_index"],
                                    bs_count=pin["bs"])
        else:
            validate_numeric_arrays(component, arrays)
        validate_release_policy(component, arrays, companion.release_policy)
    except ValueError as exc:
        raise IntegrityError(f"{_label(header)}: full check failed: {exc}") from None
    manifest = numeric_manifest(component, arrays)
    if manifest != row["numeric_content"] or content_digest(manifest) != row["content_sha256"]:
        raise IntegrityError(f"{_label(header)}: arrays differ from the published content proof")


def _label(header: dict[str, Any]) -> str:
    i = header["identity"]
    if i["component"] == "positions":
        return f"positions of {i['scene_id']}"
    return f"{i['component']} {i['scene_id']} {i['frequency_hz']} Hz BS {i['bs_index']}"
