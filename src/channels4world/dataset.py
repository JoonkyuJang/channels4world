"""Navigation, selection and reading.

Everything here is a lazy view over the root tables until a ``read*`` / ``records()`` /
``open_geometry`` / ``extract_geometry`` call performs payload I/O.
"""

from __future__ import annotations

import io
import os
import shlex
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import numpy as np

from . import _contract as C
from ._errors import IntegrityError, NotDownloadedError, NotPublishedError, UnknownIDError
from ._paths import safe_name, safe_relpath, write_file
from ._root import Root, SceneInfo

K = TypeVar("K")
V = TypeVar("V")
COMPONENTS = ("core", "detail", "positions", "geometry")
DEFAULT_COMPONENTS = ("core", "positions", "geometry")


def open(path: str | os.PathLike[str]) -> "Dataset":  # noqa: A001 - public name
    """Open a downloaded release, Tiny, an extracted subset or a cache. Reads only the small root."""
    return Dataset(Root(path))


# ============================================================================ collections
class Collection(Generic[K, V]):
    """Look up by ID; iterating yields the objects (not keys). ``.ids`` gives the keys."""

    def __init__(self, label: str, key_type: type, ids: list[K], make: Callable[[K], V]) -> None:
        self._label, self._type, self._ids, self._make = label, key_type, list(ids), make
        self._set = set(self._ids)

    def _check(self, key: Any) -> None:
        if self._type is int:
            if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
                hint = " (use integer Hz, e.g. 3_500_000_000)" if self._label == "frequency" else ""
                raise TypeError(f"{self._label} IDs are integers{hint}, not {type(key).__name__}")
        elif not isinstance(key, self._type):
            raise TypeError(f"{self._label} IDs are {self._type.__name__}, not {type(key).__name__}")

    def __getitem__(self, key: K) -> V:
        self._check(key)
        key = int(key) if self._type is int else key  # type: ignore[assignment]
        if key not in self._set:
            sample = ", ".join(map(str, self._ids[:5])) + (", ..." if len(self._ids) > 5 else "")
            raise UnknownIDError(f"no {self._label} {key!r} here (available: {sample or 'none'})")
        return self._make(key)

    def __contains__(self, key: object) -> bool:
        try:
            self._check(key)
        except TypeError:
            return False
        return key in self._set

    def __iter__(self) -> Iterator[V]:
        return (self._make(k) for k in self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[K]:
        return list(self._ids)

    def __repr__(self) -> str:
        return f"<{len(self._ids)} {self._label}{'s' if len(self._ids) != 1 else ''}>"


# ============================================================================ dataset
class Dataset:
    def __init__(self, root: Root) -> None:
        self._root = root

    def __reduce__(self) -> tuple[Any, ...]:  # picklable: reopens by path in another process
        return (open, (str(self._root.path),))

    @property
    def path(self) -> Path:
        return self._root.path

    @property
    def kind(self) -> str:
        """``release``, ``subset`` (Tiny and extractions) or ``cache``."""
        return self._root.kind

    @property
    def release(self) -> str:
        return self._root.release

    @property
    def tiers(self) -> list[str]:
        return list(self._root.tiers)

    @property
    def frequencies_hz(self) -> list[int]:
        return list(self._root.frequencies_hz)

    @property
    def splits(self) -> Collection[str, "Split"]:
        return Collection("split", str, self._root.splits, lambda s: Split(self, s))

    @property
    def scenes(self) -> Collection[str, "Scene"]:
        return Collection("scene", str, self._root.scene_order, self.scene)

    def scene(self, scene_id: str) -> "Scene":
        """Direct lookup, whatever the split."""
        if not isinstance(scene_id, str):
            raise TypeError(f"scene IDs are str, not {type(scene_id).__name__}")
        info = self._root.scenes.get(scene_id)
        if info is None:
            raise UnknownIDError(f"no scene {scene_id!r} in this {self.kind}")
        return Scene(self, info)

    def select(self, *, split: str | None = None, tier: str | None = None, profile: Any = None,
               scene_ids: Any = None, frequencies_hz: Any = None, bs_ids: Any = None,
               components: tuple[str, ...] | list[str] = DEFAULT_COMPONENTS) -> "Selection":
        """Choose scenes, carriers, base stations and components. Nothing is read.

        ``profile`` is one profile name or a list of names, spelled as in the catalog
        (``ds.profiles``; for example ``"metropolitan"``).
        """
        return Selection(self, split=split, tier=tier, profile=profile, scene_ids=scene_ids,
                         frequencies_hz=frequencies_hz, bs_ids=bs_ids, components=components)

    def select_recipe(self, recipe: "str | os.PathLike[str] | dict[str, Any]") -> "Selection":
        """The selection a recipe describes (a file path or a parsed recipe). Nothing is read.

        The recipe must name this root's release. Its filters are checked against this
        catalog exactly as ``select()`` checks them; nothing is silently widened or narrowed.
        """
        from . import recipe as R
        r = R.load(recipe)
        if r["release"] != self.release:
            raise R.RecipeError(f"the recipe is for release {r['release']}, but this {self.kind} is "
                                f"release {self.release}")
        return self.select(**R.select_kwargs(r))

    @property
    def profiles(self) -> list[str]:
        """Profile names present in this root, in first-appearance order."""
        r = self._root
        return list(dict.fromkeys(r.scenes[s].profile for s in r.scene_order))

    def __repr__(self) -> str:
        r = self._root
        lines = [f"Channels4World {r.release} ({r.kind}) at {r.path}"]
        counts = {s: 0 for s in r.splits}
        for s in r.scene_order:
            counts[r.scenes[s].split] += 1
        lines.append("  splits:      " + ", ".join(f"{k} ({v} scene{'s' if v != 1 else ''})" for k, v in counts.items()))
        rank = {t: i for i, t in enumerate(r.tiers)}
        tier_counts = [(t, sum(1 for s in r.scenes.values() if rank[s.tier] <= rank[t])) for t in r.tiers]
        lines.append("  tiers:       " + ", ".join(f"{t} ({n})" for t, n in tier_counts if n))
        lines.append("  frequencies: " + ", ".join(f"{f / 1e9:g} GHz" for f in r.frequencies_hz))
        comps = sorted({f["component"] for f in r.files if f["kind"] in ("payload", "cache_index") and f["component"]})
        lines.append("  components:  " + ", ".join(comps))
        return "\n".join(lines)


class Split:
    def __init__(self, dataset: Dataset, name: str) -> None:
        self.dataset, self.name = dataset, name

    @property
    def scenes(self) -> Collection[str, "Scene"]:
        r = self.dataset._root
        ids = [s for s in r.scene_order if r.scenes[s].split == self.name]
        return Collection("scene", str, ids, self.dataset.scene)

    def select(self, **kwargs: Any) -> "Selection":
        return self.dataset.select(split=self.name, **kwargs)

    def __repr__(self) -> str:
        return f"Split({self.name!r}, {len(self.scenes)} scenes)"


# ============================================================================ scene level
class Scene:
    def __init__(self, dataset: Dataset, info: SceneInfo) -> None:
        self.dataset, self._info = dataset, info

    id = property(lambda self: self._info.scene_id)
    split = property(lambda self: self._info.split)
    profile = property(lambda self: self._info.profile)
    tier = property(lambda self: self._info.tier, doc="Smallest tier containing this scene.")

    @property
    def _group(self) -> Any:
        return self.dataset._root.group(self._info.group)

    def _records(self) -> dict[tuple[str, int | None, int | None], str]:
        return self._group.records_of_scene(self.id)

    @property
    def frequencies(self) -> Collection[int, "Frequency"]:
        hz = sorted({k[1] for k in self._records() if k[0] == "core"})
        return Collection("frequency", int, hz, lambda f: Frequency(self, f))

    def read_positions(self, verify: str = "basic") -> "Positions":
        C.check_verify(verify)
        rid = self._records().get(("positions", None, None))
        if rid is None:
            raise NotPublishedError(f"positions of {self.id} are not part of this {self.dataset.kind}")
        return Positions(self.id, self._group.read_numeric(rid, verify))

    def _geometry_bytes(self, verify: str) -> bytes:
        C.check_verify(verify)
        r = self.dataset._root
        rid = C.geometry_record_id(r.collection_id, self.id)
        if not r.is_included(rid):
            raise NotPublishedError(f"geometry of {self.id} is not part of this {self.dataset.kind}")
        return self._group.read_member(rid, verify)

    def open_geometry(self, verify: str = "basic") -> zipfile.ZipFile:
        """The scene's geometry archive (Mitsuba XML + OBJ meshes + scene.json), in memory."""
        return zipfile.ZipFile(io.BytesIO(self._geometry_bytes(verify)))

    def extract_geometry(self, dest: str | os.PathLike[str], verify: str = "basic") -> Path:
        """Write the geometry to ``dest/<scene_id>/`` and return the path of the scene XML."""
        root = Path(dest).expanduser().resolve()
        rel_dir = safe_name(self.id)
        entry = None
        with self.open_geometry(verify) as z:
            names = [i.filename for i in z.infolist() if not i.is_dir()]
            for name in names:
                safe_relpath(name)
            for name in names:
                write_file(root, f"{rel_dir}/{name}", z.read(name))
            if "scene.json" in names:
                entry = C.strict_json(z.read("scene.json")).get("entrypoint")
        out = root / rel_dir
        if entry:
            return out / safe_relpath(entry)
        xml = sorted(n for n in names if n.endswith(".xml"))
        return out / xml[0] if xml else out

    def records(self, detail: bool = False, verify: str = "basic") -> Iterator["Record"]:
        """All channel records of this scene, ascending (frequency, BS)."""
        for f in self.frequencies:
            for bs in f.base_stations:
                yield bs.read(detail=detail, verify=verify)

    def __repr__(self) -> str:
        return f"Scene({self.id!r}, split={self.split!r}, profile={self.profile!r}, tier={self.tier!r})"


class Frequency:
    def __init__(self, scene: Scene, hz: int) -> None:
        self.scene, self.hz = scene, hz

    @property
    def base_stations(self) -> Collection[int, "BaseStation"]:
        ids = sorted(k[2] for k in self.scene._records() if k[0] == "core" and k[1] == self.hz)
        return Collection("base station", int, ids, lambda b: BaseStation(self.scene, self.hz, b))

    def __repr__(self) -> str:
        return f"Frequency({self.scene.id!r}, {self.hz})"


class BaseStation:
    def __init__(self, scene: Scene, frequency_hz: int, bs_id: int) -> None:
        self.scene, self.frequency_hz, self.id = scene, frequency_hz, bs_id

    def read(self, detail: bool = False, verify: str = "basic") -> "Record":
        """Read this channel record (core; plus interaction detail if ``detail=True``)."""
        C.check_verify(verify)
        recs = self.scene._records()
        group = self.scene._group
        core_rid = recs[("core", self.frequency_hz, self.id)]
        core = group.read_numeric(core_rid, verify)
        det = None
        if detail:
            det_rid = recs.get(("detail", self.frequency_hz, self.id))
            if det_rid is None:
                ident = {"bs_index": self.id, "collection_id": self.scene.dataset._root.collection_id,
                         "component": "detail", "frequency_hz": self.frequency_hz, "scene_id": self.scene.id}
                where = ("not included in this subset" if C.logical_id(ident) in group.snapshot
                         else "not published in this release")
                raise NotPublishedError(f"detail for {self.scene.id} {self.frequency_hz} Hz BS {self.id} is {where}")
            det = group.read_numeric(det_rid, verify)
            if verify == "full":
                try:
                    C.validate_detail(det, core)
                except ValueError as exc:
                    raise IntegrityError(f"detail/core association: {exc}") from None
        return Record(group.snapshot.row(core_rid)["record"], core_rid, core, det)

    def __repr__(self) -> str:
        return f"BaseStation({self.scene.id!r}, {self.frequency_hz}, {self.id})"


# ============================================================================ loaded data
class Record:
    """One loaded scene/carrier/BS channel record. Arrays are read-only NumPy."""

    def __init__(self, header: dict[str, Any], record_id: str, core: dict[str, np.ndarray],
                 detail: dict[str, np.ndarray] | None) -> None:
        self.header = header
        self.id = record_id
        i = header["identity"]
        self.scene_id: str = i["scene_id"]
        self.frequency_hz: int = i["frequency_hz"]
        self.bs_id: int = i["bs_index"]
        self.n_ues: int = header["dependencies"]["positions"]["ues"]
        for arrays in (core, detail or {}):
            for a in arrays.values():
                a.flags.writeable = False
        self.core = MappingProxyType(core)
        self.detail = None if detail is None else MappingProxyType(detail)

    def __reduce__(self) -> tuple[Any, ...]:
        # Picklable (e.g. returned from DataLoader workers); arrays travel as plain read-only ndarrays.
        plain = lambda m: None if m is None else {k: np.asarray(v) for k, v in m.items()}  # noqa: E731
        return (Record, (self.header, self.id, plain(self.core), plain(self.detail)))

    @property
    def ue_ids(self) -> np.ndarray:
        return self.core["link_ue_index"]

    @property
    def n_links(self) -> int:
        return int(self.core["link_ue_index"].shape[0])

    @property
    def n_paths(self) -> int:
        return int(self.core["path_rank"].shape[0])

    def get_link(self, ue_id: int) -> "Link | None":
        """The retained link to ``ue_id``, or ``None`` if that valid UE has no retained link."""
        if isinstance(ue_id, bool) or not isinstance(ue_id, (int, np.integer)):
            raise TypeError(f"UE IDs are integers, not {type(ue_id).__name__}")
        if not 0 <= ue_id < self.n_ues:
            raise UnknownIDError(f"UE {ue_id} is outside this scene's UE range 0..{self.n_ues - 1}")
        ids = self.ue_ids
        i = int(np.searchsorted(ids, ue_id))
        return Link(self, i) if i < len(ids) and int(ids[i]) == ue_id else None

    def links(self) -> Iterator["Link"]:
        return (Link(self, i) for i in range(self.n_links))

    def copy(self) -> dict[str, dict[str, np.ndarray]]:
        """Writable copies of all arrays."""
        out = {"core": {k: np.array(v) for k, v in self.core.items()}}
        if self.detail is not None:
            out["detail"] = {k: np.array(v) for k, v in self.detail.items()}
        return out

    def __repr__(self) -> str:
        d = ", detail" if self.detail is not None else ""
        return (f"Record({self.scene_id!r}, {self.frequency_hz} Hz, BS {self.bs_id}: "
                f"{self.n_links} links, {self.n_paths} paths{d})")


_PATH_FIELDS = ("path_rank", "a_real", "a_imag", "tau", "theta_t", "phi_t", "theta_r", "phi_r",
                "interaction_count", "interaction_flags")


class Link:
    """Paths of one retained link: views into the record's arrays, no I/O."""

    def __init__(self, record: Record, index: int) -> None:
        self.record, self.index = record, index
        off = record.core["link_path_offsets"]
        self.paths = slice(int(off[index]), int(off[index + 1]))

    @property
    def ue_id(self) -> int:
        return int(self.record.ue_ids[self.index])

    @property
    def n_paths(self) -> int:
        return self.paths.stop - self.paths.start

    @property
    def core(self) -> dict[str, np.ndarray]:
        return {k: self.record.core[k][self.paths] for k in _PATH_FIELDS}

    @property
    def detail(self) -> dict[str, np.ndarray] | None:
        """Interaction arrays of this link's paths; offsets rebased to start at 0."""
        d = self.record.detail
        if d is None:
            return None
        off = d["path_interaction_offsets"][self.paths.start:self.paths.stop + 1]
        span = slice(int(off[0]), int(off[-1]))
        rebased = off - off[0]
        rebased.flags.writeable = False
        return {"path_interaction_offsets": rebased, "interaction_rank": d["interaction_rank"][span],
                "interaction_type": d["interaction_type"][span], "interaction_vertex": d["interaction_vertex"][span]}

    def __repr__(self) -> str:
        return f"Link(UE {self.ue_id}, {self.n_paths} paths)"


class Positions:
    def __init__(self, scene_id: str, arrays: dict[str, np.ndarray]) -> None:
        self.scene_id = scene_id
        self.bs: np.ndarray = arrays["bs_positions"]
        self.ue: np.ndarray = arrays["ue_positions"]

    def __repr__(self) -> str:
        return f"Positions({self.scene_id!r}: {len(self.bs)} BS, {len(self.ue)} UEs, metres)"


# ============================================================================ selection
class _Key:
    __slots__ = ("scene", "hz", "bs", "core", "detail")

    def __init__(self, scene: Scene, hz: int, bs: int, core: str, detail: str | None) -> None:
        self.scene, self.hz, self.bs, self.core, self.detail = scene, hz, bs, core, detail


class _Item:
    __slots__ = ("scene", "component", "hz", "bs", "record_id", "paths")

    def __init__(self, scene: Scene, component: str, hz: int | None, bs: int | None, record_id: str | None,
                 paths: list[str] | None) -> None:
        self.scene, self.component, self.hz, self.bs = scene, component, hz, bs
        self.record_id, self.paths = record_id, paths

    @property
    def label(self) -> str:
        if self.hz is None:
            return f"{self.component} of {self.scene.id}"
        return f"{self.component} {self.scene.id} {self.hz} Hz BS {self.bs}"


class Selection:
    """A chosen set of scenes, carriers, BSs and components. Nothing is read until iteration."""

    def __init__(self, dataset: Dataset, *, split: str | None, tier: str | None, scene_ids: Any,
                 frequencies_hz: Any, bs_ids: Any, components: Any, profile: Any = None,
                 _keep: set[str] | None = None, _scenes: list[str] | None = None) -> None:
        r = dataset._root
        self.dataset = dataset
        comps = (components,) if isinstance(components, str) else tuple(components)
        bad = [c for c in comps if c not in COMPONENTS]
        if bad:
            raise ValueError(f"unknown component(s) {bad}; choose from {COMPONENTS}")
        if "detail" in comps and "core" not in comps:
            comps = ("core",) + comps  # detail is always read together with its core record
        self.components = tuple(c for c in COMPONENTS if c in comps)
        if split is not None and split not in r.splits:
            raise UnknownIDError(f"no split {split!r} (available: {', '.join(r.splits)})")
        if tier is not None and tier not in r.tiers:
            raise UnknownIDError(f"no tier {tier!r} (tiers: {', '.join(r.tiers)})")
        self.split, self.tier = split, tier
        profiles = None
        if profile is not None:
            profiles = [profile] if isinstance(profile, str) else list(profile)
            known = dataset.profiles
            for p in profiles:
                if not isinstance(p, str):
                    raise TypeError(f"profile names are str, not {type(p).__name__}")
            unknown = [p for p in profiles if p not in known]
            if unknown:
                raise UnknownIDError(f"no profile {unknown[0]!r} (available: {', '.join(known)})")
        self.profiles = None if profiles is None else tuple(profiles)
        self.frequencies_hz = None if frequencies_hz is None else _int_list(frequencies_hz, "frequency")
        for f in self.frequencies_hz or ():
            if f not in r.frequencies_hz:
                raise UnknownIDError(f"no frequency {f} Hz (available: {r.frequencies_hz})")
        self.bs_ids = None if bs_ids is None else set(_int_list(bs_ids, "base station"))
        wanted = None
        if scene_ids is not None:
            wanted = [scene_ids] if isinstance(scene_ids, str) else list(scene_ids)
            unknown = [s for s in wanted if s not in r.scenes]
            if unknown:
                raise UnknownIDError(f"unknown scene(s): {unknown[:5]}")
            wanted = set(wanted)
        if _scenes is not None:
            ids = _scenes
        else:
            rank = {t: i for i, t in enumerate(r.tiers)}
            ids = [s for s in r.scene_order
                   if (split is None or r.scenes[s].split == split)
                   and (tier is None or rank[r.scenes[s].tier] <= rank[tier])
                   and (profiles is None or r.scenes[s].profile in profiles)
                   and (wanted is None or s in wanted)]
        self._scene_ids = ids
        self._keep = _keep
        self._args = dict(split=split, tier=tier, profile=profile, scene_ids=scene_ids,
                          frequencies_hz=frequencies_hz, bs_ids=bs_ids, components=self.components)

    @property
    def scene_ids(self) -> list[str]:
        return list(self._scene_ids)

    def scenes(self) -> Iterator["SelectedScene"]:
        """Selected scenes in pinned order."""
        for s in self._scene_ids:
            yield SelectedScene(self, self.dataset.scene(s))

    def _keys(self, scene: Scene, apply_keep: bool = True) -> Iterator[_Key]:
        if "core" not in self.components:
            return
        recs = scene._records()
        for (comp, hz, bs), rid in sorted(((k, v) for k, v in recs.items() if k[0] == "core"),
                                          key=lambda kv: (kv[0][1], kv[0][2])):
            if self.frequencies_hz is not None and hz not in self.frequencies_hz:
                continue
            if self.bs_ids is not None and bs not in self.bs_ids:
                continue
            if apply_keep and self._keep is not None and rid not in self._keep:
                continue
            det = recs.get(("detail", hz, bs)) if "detail" in self.components else None
            yield _Key(scene, hz, bs, rid, det)

    def records(self, verify: str = "basic") -> Iterator[Record]:
        """All selected channel records: scenes in pinned order, then ascending (frequency, BS)."""
        for scene in self.scenes():
            yield from scene.records(verify=verify)

    # ---------------------------------------------------------------- availability
    def _paths_of(self, scene: Scene, rid: str, component: str = "numeric") -> list[str] | None:
        """Files holding ``rid``, or None when this root does not publish it.

        A record can be catalogued in its group (members, snapshot) while its archive is
        left out of the root's file table, as the release does for Large and Full detail.
        Planning reports such items as unpublished; direct reads still raise.
        """
        r = self.dataset._root
        g = scene._group
        if r.kind == "cache" and component != "geometry" and rid in g.snapshot:
            paths = g.part_paths(rid)
            if paths is not None:
                return paths
        try:
            return [r.locate(g, rid).path]
        except NotPublishedError:
            return None

    def _scene_items(self, scene: Scene) -> Iterator["_Item"]:
        """Every requested item of one scene; ``paths`` is None when it is not published."""
        r = self.dataset._root
        g = scene._group
        recs = scene._records() if ("positions" in self.components or "core" in self.components) else {}
        if "positions" in self.components:
            rid = recs.get(("positions", None, None))
            yield _Item(scene, "positions", None, None, rid, self._paths_of(scene, rid) if rid else None)
        if "geometry" in self.components:
            rid = C.geometry_record_id(r.collection_id, scene.id)
            ok = r.is_included(rid) and rid in g.members
            yield _Item(scene, "geometry", None, None, rid, self._paths_of(scene, rid, "geometry") if ok else None)
        for k in self._keys(scene, apply_keep=False):
            yield _Item(scene, "core", k.hz, k.bs, k.core, self._paths_of(scene, k.core))
            if "detail" in self.components:
                yield _Item(scene, "detail", k.hz, k.bs, k.detail,
                            self._paths_of(scene, k.detail) if k.detail else None)

    def _absent_requests(self, scene: Scene) -> list[str]:
        """Requested channel records that this root does not hold for ``scene``.

        A scene with no channel records at all is reported whenever core is requested.
        Otherwise only explicit ``frequencies_hz`` / ``bs_ids`` filters are checked; without
        them a selection simply takes what each scene has. Needs the group metadata. Every
        consumer (``missing``, ``summary``, preparation plans, CLI ``verify``) uses this one
        calculation; ``local()`` selections skip it because they narrow on purpose.
        """
        if "core" not in self.components or self._keep is not None:
            return []
        have = {(k[1], k[2]) for k in scene._records() if k[0] == "core"}
        if not have:
            return [f"core {scene.id}: no channel records in this {self.dataset.kind}"]
        if self.frequencies_hz is None and self.bs_ids is None:
            return []
        hz_all = sorted({hz for hz, _ in have})
        bs_all = sorted({bs for _, bs in have})
        hzs = self.frequencies_hz if self.frequencies_hz is not None else hz_all
        bss = sorted(self.bs_ids) if self.bs_ids is not None else bs_all
        return [f"core {scene.id} {hz} Hz BS {bs}" for hz in hzs for bs in bss if (hz, bs) not in have]

    def _absent(self) -> list[str]:
        """``_absent_requests`` over every selected scene."""
        return [label for s in self._scene_ids for label in self._absent_requests(self.dataset.scene(s))]

    def _plan(self) -> Iterator["_Item"]:
        """Requested items of all selected scenes (only the locally usable ones after ``local()``)."""
        for scene in (self.dataset.scene(s) for s in self._scene_ids):
            for item in self._scene_items(scene):
                if self._keep is None or item.record_id in self._keep:
                    yield item

    def missing(self) -> "MissingReport":
        """Files this selection needs that are not downloaded, plus requested items not published."""
        r = self.dataset._root
        exists: dict[str, bool] = {}

        def present(p: str) -> bool:
            if p not in exists:
                exists[p] = (r.path / p).exists()
            return exists[p]

        files: dict[str, int] = {}
        unpublished: list[str] = []
        schema = [f for f in r.files if f["kind"] == "schema" and not present(f["path"])]
        for f in schema:
            files[f["path"]] = f["bytes"]
        coarse: set[str] = set()  # groups whose records cannot be enumerated yet
        for s in self._scene_ids:
            g = r.scenes[s].group
            if g in coarse:
                continue
            absent = [m for m in (r.group_meta_file(g, k) for k in ("members", "snapshot")) if not present(m["path"])]
            if absent or schema:
                coarse.add(g)
                for m in absent:
                    files[m["path"]] = m["bytes"]
                for f in r.group_files.get(g, ()):
                    if (f["kind"] in ("payload", "cache_index", "cache_array") and f["component"] in self.components
                            and (f["frequency_hz"] is None or self.frequencies_hz is None
                                 or f["frequency_hz"] in self.frequencies_hz)
                            and not present(f["path"])):
                        files[f["path"]] = f["bytes"]
        for s in self._scene_ids:
            if r.scenes[s].group in coarse:
                continue
            unpublished += self._absent_requests(self.dataset.scene(s))
            for item in self._scene_items(self.dataset.scene(s)):
                if self._keep is not None and item.record_id not in self._keep:
                    continue
                if item.paths is None:
                    unpublished.append(item.label)
                    continue
                for p in item.paths:
                    if not present(p):
                        files[p] = (r.by_path[p]["bytes"] if p in r.by_path else 0) or 0
        return MissingReport(sorted(files), sum(files.values()), unpublished, r.hf_repo, r.hf_path, r.path)

    def local(self) -> "Selection":
        """The same selection narrowed, per component, to what is present locally and readable.

        Channel records are kept only if every requested channel component (core, and detail if
        selected) is present. Positions and geometry are kept independently. Scenes with nothing
        usable, or whose group metadata (members, snapshot, shared schemas) is not downloaded,
        are dropped: every selection-level operation requires that metadata. Direct reads such
        as ``scene.open_geometry()`` need only the members table.
        """
        r = self.dataset._root
        exists: dict[str, bool] = {}

        def present(paths: list[str] | None) -> bool:
            if paths is None:
                return False
            for p in paths:
                if p not in exists:
                    exists[p] = (r.path / p).exists()
                if not exists[p]:
                    return False
            return True

        keep: set[str] = set()
        scenes: list[str] = []
        for s in self._scene_ids:
            if not present(r.metadata_paths(r.scenes[s].group)):
                continue  # selections always require the group's metadata (see missing())
            try:
                items = list(self._scene_items(self.dataset.scene(s)))
            except NotDownloadedError:
                continue
            if self._keep is not None:
                items = [i for i in items if i.record_id in self._keep]
            ok = {(i.component, i.hz, i.bs): i for i in items if present(i.paths)}
            usable = [i for i in ok.values() if i.component in ("positions", "geometry")]
            for i in ok.values():
                if i.component == "core" and ("detail" not in self.components or ("detail", i.hz, i.bs) in ok):
                    usable.append(i)
                    if "detail" in self.components:
                        usable.append(ok[("detail", i.hz, i.bs)])
            keep.update(i.record_id for i in usable)
            if usable:
                scenes.append(s)
        return Selection(self.dataset, split=None, tier=None, scene_ids=None, frequencies_hz=self.frequencies_hz,
                         bs_ids=self.bs_ids, components=self.components, profile=None, _keep=keep, _scenes=scenes)

    def summary(self) -> dict[str, Any]:
        """Counts and published bytes of what this selection would read or extract (metadata only)."""
        sizes: dict[str, int] = {}
        records = scenes = unpublished = 0
        seen_scene: set[str] = set()
        for item in self._plan():
            if item.scene.id not in seen_scene:
                seen_scene.add(item.scene.id)
                scenes += 1
            if item.paths is None:
                unpublished += 1
                continue
            if item.component == "core":
                records += 1
            g = item.scene._group
            if item.record_id in g.members:
                sizes[item.record_id] = g.members[item.record_id]["member_bytes"]
        unpublished += len(self._absent())
        return {"scenes": scenes, "records": records, "files": len(sizes), "bytes": sum(sizes.values()),
                "unpublished": unpublished, "components": self.components}

    def to_recipe(self, revision: str | None = None) -> dict[str, Any]:
        """This selection as a recipe (see :mod:`channels4world.recipe`).

        ``revision`` records the Hugging Face revision the catalog came from. A ``local()``
        selection depends on what happens to be downloaded, so export the selection it came from.
        """
        from . import recipe as R
        if self._keep is not None:
            raise ValueError("a local() selection depends on the files on this machine; "
                             "export the selection it was made from instead")
        a = self._args
        sel: dict[str, Any] = {"components": list(self.components)}
        if a["tier"] is not None:
            sel["tier"] = a["tier"]
        if a["split"] is not None:
            sel["split"] = a["split"]
        if self.profiles is not None:
            sel["profiles"] = list(self.profiles)
        if a["scene_ids"] is not None:
            ids = [a["scene_ids"]] if isinstance(a["scene_ids"], str) else list(a["scene_ids"])
            sel["scene_ids"] = list(dict.fromkeys(ids))
        if self.frequencies_hz is not None:
            sel["frequencies_hz"] = sorted(set(self.frequencies_hz))
        if self.bs_ids is not None:
            sel["bs_ids"] = sorted(self.bs_ids)
        recipe: dict[str, Any] = {"format": R.FORMAT, "format_version": R.FORMAT_VERSION,
                                  "release": self.dataset.release, "selection": sel}
        if revision is not None:
            recipe["revision"] = revision
        return R.validate(recipe)

    def __repr__(self) -> str:
        filt = ", ".join(f"{k}={v!r}" for k, v in self._args.items() if v is not None and k != "components")
        local = ", local only" if self._keep is not None else ""
        return f"Selection({len(self._scene_ids)} scenes, components={self.components}{', ' if filt else ''}{filt}{local})"


class SelectedScene(Scene):
    """A scene seen through a selection: ``records()`` applies its filters and components."""

    def __init__(self, selection: Selection, scene: Scene) -> None:
        super().__init__(scene.dataset, scene._info)
        self.selection = selection

    def records(self, detail: bool | None = None, verify: str = "basic") -> Iterator[Record]:  # type: ignore[override]
        want_detail = ("detail" in self.selection.components) if detail is None else detail
        for k in self.selection._keys(self):
            yield BaseStation(self, k.hz, k.bs).read(detail=want_detail, verify=verify)


class MissingReport:
    """What a selection still needs.

    ``files`` are paths relative to the opened root. ``repo_files`` and ``hf_command()`` map
    them to repository paths, and are available only for a published root (a release or
    Tiny root opened from a download that mirrors the repository layout).
    """

    def __init__(self, files: list[str], nbytes: int, unpublished: list[str], hf_repo: str | None,
                 hf_path: str | None, root_dir: Path) -> None:
        self.files, self.bytes, self.unpublished = files, nbytes, unpublished
        self._repo, self._path, self._root = hf_repo, hf_path, root_dir

    def __bool__(self) -> bool:
        """True when files need downloading (unpublished items cannot be downloaded)."""
        return bool(self.files)

    @property
    def complete(self) -> bool:
        """Everything requested is published and downloaded."""
        return not self.files and not self.unpublished

    def _published(self) -> tuple[str, str]:
        if not self._repo or self._path is None:
            raise ValueError("this root is not a published Hugging Face root (for example, a subset or cache "
                             "you created), so its missing files cannot be downloaded from the repository")
        return self._repo, self._path

    @property
    def repo_files(self) -> list[str]:
        """Missing files as repository paths, e.g. for ``snapshot_download(allow_patterns=...)``."""
        _, prefix = self._published()
        return [f"{prefix}/{p}" if prefix else p for p in self.files]

    @property
    def local_dir(self) -> Path:
        """The download directory that corresponds to the repository's top level."""
        _, prefix = self._published()
        d = self._root
        for part in reversed(prefix.split("/") if prefix else []):
            if d.name != part:
                raise ValueError(f"{self._root} is not at '<download dir>/{prefix}' as published; "
                                 "download with your own mapping of report.repo_files")
            d = d.parent
        return d

    def hf_command(self, revision: str | None = None) -> str:
        """An ``hf download`` command for exactly the missing files.

        Pass the release ``revision`` you downloaded the catalog from, so every file comes from
        the same release.
        """
        if not self.files:
            raise ValueError("nothing to download: report.files is empty")
        repo, _ = self._published()
        args = ["hf", "download", repo, "--repo-type", "dataset"]
        if revision:
            args += ["--revision", revision]
        args += ["--local-dir", str(self.local_dir)]
        for path in self.repo_files:
            args += ["--include", path]
        return shlex.join(args)  # every argument quoted for POSIX shells

    def __repr__(self) -> str:
        lines = [f"{len(self.files)} file(s) missing ({_fmt_bytes(self.bytes)})"]
        lines += [f"  {p}" for p in self.files[:20]]
        if len(self.files) > 20:
            lines.append(f"  ... and {len(self.files) - 20} more")
        if self.unpublished:
            lines.append(f"{len(self.unpublished)} requested item(s) are not published, e.g. {self.unpublished[0]}")
        return "\n".join(lines)


def _int_list(values: Any, label: str) -> list[int]:
    vals = [values] if isinstance(values, (int, np.integer)) else list(values)
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"{label} IDs are integers, not {type(v).__name__}")
    return [int(v) for v in vals]


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n} B"
