#!/usr/bin/env python3
"""Build the library tutorials and the dataset's NumPy example from one source.

Library pages and notebooks are maintained here. NumPy-only outputs are published
on Hugging Face with --hf-card. Existing HF library notebooks are kept as frozen,
clearly labelled compatibility copies; their code is never regenerated here.

    python tutorials-src/build_tutorials.py [--check] [--hf-card PATH]
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parent.parent
GH = "https://github.com/JoonkyuJang/channels4world/blob/main/"
HF = "https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World/blob/main/"
NOTEBOOK_PATHS = {
    "01-read-one-scene-library.ipynb": "examples/01-read-one-scene-library.ipynb",
    "03-dataloader.ipynb": "examples/02-dataloader.ipynb",
    "02-read-one-scene-numpy.ipynb": "docs/notebooks/02-read-one-scene-numpy.ipynb",
}

COMMON_IDS = '''REPO, REVISION = "Intelligence-Networking-Lab/Channels4World", "v1.0.0"
SCENE_ID, GROUP = "urban_scene_019234", "test-urban-g00000"  # GROUP: from the tiny_scenes catalog
CARRIER_HZ, BS_ID, UE_ID = 3_500_000_000, 0, 14
OTHER_HZ = 7_000_000_000
DATA = Path("c4w")'''

READ_OBJ = '''def top_view(read, names):
    """Building triangles projected onto the ground (x, y), from OBJ meshes."""
    triangles = []
    for name in names:
        verts, faces = [], []
        for line in read(name).decode().splitlines():
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
        v = np.array(verts)
        triangles += [v[[i - 1 for i in f], :2] for f in faces]   # OBJ indices start at 1
    return triangles'''

PLOT_SCENE = '''fig, ax = plt.subplots(figsize=(6, 6))
ax.add_collection(PolyCollection(triangles, facecolor="#c3cbc2", edgecolor="none"))
ax.scatter(ue_positions[:, 0], ue_positions[:, 1], s=0.2, color="#4d9688", label=f"{len(ue_positions):,} UEs")
ax.scatter(bs_positions[:, 0], bs_positions[:, 1], marker="^", s=30, color="k", label=f"{len(bs_positions)} BSs")
ax.plot(*zip(bs_xyz[:2], ue_xyz[:2]), color="#e2733d", lw=2, label=f"BS {BS_ID} → UE {UE_ID}")
ax.set(xlim=(-250, 250), ylim=(-250, 250), aspect="equal", xlabel="x (m)", ylabel="y (m)", title=SCENE_ID)
ax.legend(loc="upper right")
plt.show()'''

PATH_TABLE = '''NAMES = {1: "specular reflection", 2: "diffuse reflection", 4: "refraction", 8: "diffraction"}

def describe(core, detail):
    """One line per retained path, strongest first."""
    re, im = core["a_real"].astype(np.float64), core["a_imag"].astype(np.float64)
    power_db = 10 * np.log10(re**2 + im**2)
    off = detail["path_interaction_offsets"]
    for p in range(len(core["tau"])):
        kinds = [NAMES.get(int(t), str(t)) for t in detail["interaction_type"][off[p]:off[p + 1]]]
        print(f"rank {int(core['path_rank'][p]):2d}  {core['tau'][p] * 1e9:8.2f} ns  {power_db[p]:7.2f} dB  "
              f"AoD az {np.degrees(core['phi_t'][p]):6.1f}°  AoA az {np.degrees(core['phi_r'][p]):6.1f}°  "
              f"{' → '.join(kinds) or 'line of sight'}")'''

PDP = '''def stems(ax, tau, a_real, a_imag, label, marker, color):
    power_db = 10 * np.log10(a_real.astype(np.float64)**2 + a_imag.astype(np.float64)**2)
    ax.vlines(tau * 1e9, -150, power_db, color=color, lw=1.2)
    ax.scatter(tau * 1e9, power_db, marker=marker, color=color, zorder=3, label=label)'''

LIB_TUTORIAL = [
    ("md", """# Read one scene with the channels4world library

This tutorial follows one link of Channels4World from the catalog down to its paths:
**BS 0 → UE 14 of `urban_scene_019234` at 3.5 GHz**, the link the project website shows.
It downloads about 26 MB of one Tiny scene, walks the hierarchy, reads the geometry, the
positions and the channel record, explains the stored arrays, and ends with the same link
at 7 GHz. The [notebook version](notebooks/01-read-one-scene-library.ipynb) runs top to
bottom; [the same tutorial without the library](tutorial-numpy.md) reads the same link with
NumPy alone.

Install the library with `pip install channels4world` (Python 3.11+), then
`pip install huggingface_hub matplotlib`."""),
    ("code", """from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
from matplotlib.collections import PolyCollection

import channels4world as c4w"""),
    ("md", """## 1. Choose the link

Change these to read any other link. `GROUP` is the scene's `group` in the `tiny_scenes`
catalog. BS IDs run from 0 to 31 and UE IDs from 0 to 16,383; both are rows
of the scene's positions. Carriers are integers in Hz."""),
    ("code", COMMON_IDS),
    ("md", """## 2. Download what this tutorial reads

Tiny is published as loose files under `tiny/`. The library needs Tiny's catalog, the shared
schemas and the scene's group metadata (about 11 MB); the records are the scene's positions
and geometry and BS 0's core and detail at both carriers (about 15 MB)."""),
    ("code", """scene_dir = f"tiny/records/scene={SCENE_ID}"
patterns = [
    "tiny/c4w-root.json", "tiny/c4w-root/*", "tiny/schema/*",
    f"tiny/groups/{GROUP}/catalog/members.parquet", f"tiny/groups/{GROUP}/snapshot.json",
    f"{scene_dir}/positions.npz", f"{scene_dir}/geometry.zip",
    f"{scene_dir}/band_hz=*/component=*/bs{BS_ID:05d}.npz",
]
snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=DATA, allow_patterns=patterns)"""),
    ("md", """## 3. Walk the hierarchy

The dataset is organized as **dataset → scene → carrier → base station → UE link → paths**.
Opening reads only the small catalog; group metadata and records load when first needed."""),
    ("code", """ds = c4w.open(DATA / "tiny")
print(ds)
scene = ds.scene(SCENE_ID)
print(scene)                                        # split, profile and smallest tier
print("carriers (Hz):", scene.frequencies.ids)
print("base stations:", scene.frequencies[CARRIER_HZ].base_stations.ids)"""),
    ("md", """`urban_scene_019234` is a **test** scene: every link and both carriers of a scene stay in
one split, so a model trained on the training split has never seen this place.

## 4. Geometry and positions

Geometry, positions and interaction points share one frame: scene metres, z up. Positions
hold all 32 BS and 16,384 UE coordinates of the scene; the geometry is a Mitsuba XML file
with OBJ meshes in a ZIP."""),
    ("code", """positions = scene.read_positions()
bs_positions, ue_positions = positions.bs, positions.ue
bs_xyz, ue_xyz = bs_positions[BS_ID], ue_positions[UE_ID]
print(positions)
print("BS", bs_xyz, "UE", ue_xyz, f"distance {np.linalg.norm(ue_xyz - bs_xyz):.1f} m")

""" + READ_OBJ + """

with scene.open_geometry() as geometry:
    print("entry point:", json.loads(geometry.read("scene.json"))["entrypoint"])
    meshes = [n for n in geometry.namelist() if n.endswith(".obj") and "ground" not in n]
    triangles = top_view(geometry.read, meshes)
print(len(meshes), "building meshes")"""),
    ("code", PLOT_SCENE),
    ("md", """## 5. One channel record

A record holds every retained link from one BS at one carrier. `get_link(UE_ID)` returns
that UE's link, or `None` when the UE has no retained link to this BS (which is not a
zero-valued channel). A UE ID is a row of the positions; the link's row inside the record
is a different index."""),
    ("code", """record = scene.frequencies[CARRIER_HZ].base_stations[BS_ID].read(detail=True)
print(record)
link = record.get_link(UE_ID)
print(link, f"- row {link.index} of the record's {record.n_links} links")"""),
    ("md", """## 6. What each path stores

For every retained path the core record stores the complex coefficient `a_real + 1j*a_imag`
(passband, without the carrier-delay phase, antennas included), the delay `tau` in seconds,
departure and arrival angles in radians, and its interaction count and type flags; the detail
record adds each interaction's type and location in order. Paths are stored strongest first,
and each link keeps at most its 16 strongest paths above −150 dB."""),
    ("code", PATH_TABLE + """

describe(link.core, link.detail)"""),
    ("md", """## 7. A power–delay profile

Placing each path by delay and power gives the link's power–delay profile: a path-based view
of the stored channel. It is not a sampled impulse response; for a channel at a given
bandwidth, `channels4world.transforms` provides baseband coefficients and frequency
responses."""),
    ("code", PDP + """

fig, ax = plt.subplots(figsize=(7, 3))
stems(ax, link.core["tau"], link.core["a_real"], link.core["a_imag"], "3.5 GHz", "o", "#e2733d")
ax.set(xlabel="delay (ns)", ylabel="path power (dB)", ylim=(-150, None),
       title=f"{SCENE_ID} · BS {BS_ID} → UE {UE_ID}")
ax.legend()
plt.show()"""),
    ("md", """## 8. The same link at 7 GHz

Every scene is simulated at both carriers with the same BS and UE positions. Read the same
BS and UE at 7 GHz and compare on shared axes:"""),
    ("code", """link7 = scene.frequencies[OTHER_HZ].base_stations[BS_ID].read(detail=True).get_link(UE_ID)
fig, ax = plt.subplots(figsize=(7, 3))
stems(ax, link.core["tau"], link.core["a_real"], link.core["a_imag"], "3.5 GHz", "o", "#e2733d")
if link7 is None:
    print("UE", UE_ID, "has no retained link at 7 GHz")
else:
    stems(ax, link7.core["tau"], link7.core["a_real"], link7.core["a_imag"], "7 GHz", "s", "#4d9688")
    for l, label in ((link, "3.5 GHz"), (link7, "7 GHz")):
        a = l.core["a_real"].astype(np.float64) + 1j * l.core["a_imag"].astype(np.float64)
        print(f"{label}: {l.n_paths} paths, link power {10 * np.log10(np.sum(np.abs(a) ** 2)):.2f} dB")
ax.set(xlabel="delay (ns)", ylabel="path power (dB)", ylim=(-150, None))
ax.legend()
plt.show()"""),
    ("md", """Ranks order the paths within one record: rank 0 at 7 GHz need not be the route of rank 0
at 3.5 GHz. To match routes across carriers, compare delays, interaction sequences or
interaction points (`link7.detail`), not ranks.

## Next

- [From channel records to a DataLoader](ml-guide.md): link samples, splits, padding and batches.
- [Library guide](library-guide.md): download plans for larger tiers, recipes, extraction and caches.
- [Schema reference](schema-reference.md): every array and its meaning."""),
]

NUMPY_TUTORIAL = [
    ("md", """# Read one scene without the library

The same link as the [library tutorial](tutorial-library.md), **BS 0 → UE 14 of
`urban_scene_019234` at 3.5 GHz**, read with NumPy and the standard library from Tiny's loose
files: about 15 MB. The [notebook version](notebooks/02-read-one-scene-numpy.ipynb) runs top
to bottom. Archives from larger tiers hold the same files; see
[without the library](without-library.md) for TAR access.

Install with `pip install numpy huggingface_hub matplotlib`."""),
    ("code", """from pathlib import Path
import json
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
from matplotlib.collections import PolyCollection"""),
    ("md", """## 1. Choose the link

The same identifiers as the library tutorial. A UE ID is a row of `ue_positions`, and the BS
ID is both a row of `bs_positions` and the number in the record's file name."""),
    ("code", COMMON_IDS),
    ("md", """## 2. Download the scene's files

NumPy needs only the record files: positions, geometry, and BS 0's core and detail at both
carriers."""),
    ("code", """scene_dir = DATA / "tiny" / "records" / f"scene={SCENE_ID}"
remote = f"tiny/records/scene={SCENE_ID}"
snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=DATA, allow_patterns=[
    f"{remote}/positions.npz", f"{remote}/geometry.zip", f"{remote}/band_hz=*/component=*/bs{BS_ID:05d}.npz"])

def record_path(carrier_hz, component):
    return scene_dir / f"band_hz={carrier_hz}" / f"component={component}" / f"bs{BS_ID:05d}.npz"

print(sorted(p.relative_to(scene_dir).as_posix() for p in scene_dir.rglob("*") if p.is_file()))"""),
    ("md", """## 3. The hierarchy on disk

Each scene folder holds `positions.npz` and `geometry.zip`, shared by both carriers, and one
file per carrier, component and BS: `band_hz=<Hz>/component=<core|detail>/bs<NNNNN>.npz`.
Every NPZ carries its identity in a JSON `__meta__` entry:"""),
    ("code", """with np.load(record_path(CARRIER_HZ, "core"), allow_pickle=False) as npz:
    meta = json.loads(npz["__meta__"].item())
print(json.dumps(meta["identity"], indent=1))
print(meta["dimensions"])                      # retained links and paths in this record"""),
    ("md", """## 4. Geometry and positions

Geometry, positions and interaction points share one frame: scene metres, z up."""),
    ("code", """with np.load(scene_dir / "positions.npz", allow_pickle=False) as positions:
    bs_positions, ue_positions = positions["bs_positions"], positions["ue_positions"]
bs_xyz, ue_xyz = bs_positions[BS_ID], ue_positions[UE_ID]
print(bs_positions.shape, ue_positions.shape)
print("BS", bs_xyz, "UE", ue_xyz, f"distance {np.linalg.norm(ue_xyz - bs_xyz):.1f} m")

""" + READ_OBJ + """

with zipfile.ZipFile(scene_dir / "geometry.zip") as geometry:
    print("entry point:", json.loads(geometry.read("scene.json"))["entrypoint"])
    meshes = [n for n in geometry.namelist() if n.endswith(".obj") and "ground" not in n]
    triangles = top_view(geometry.read, meshes)
print(len(meshes), "building meshes")"""),
    ("code", PLOT_SCENE),
    ("md", """## 5. Find the link

`link_ue_index` lists the UEs this BS reaches, in ascending order; the link's row is where
`UE_ID` sits in that list. Link `row` owns paths `link_path_offsets[row]` to
`link_path_offsets[row + 1]`. A UE that is not listed has no retained link to this BS (not a
zero-valued channel)."""),
    ("code", """def load(carrier_hz, component):
    with np.load(record_path(carrier_hz, component), allow_pickle=False) as npz:
        return {name: npz[name] for name in npz.files if name != "__meta__"}

def link_of(core, detail, ue_id):
    ue_ids = core["link_ue_index"]
    row = int(np.searchsorted(ue_ids, ue_id))
    if row == ue_ids.size or ue_ids[row] != ue_id:
        return None                                        # no retained link to this UE
    p0, p1 = map(int, core["link_path_offsets"][row:row + 2])
    link_core = {k: v[p0:p1] for k, v in core.items() if k not in ("link_ue_index", "link_path_offsets")}
    i0, i1 = map(int, detail["path_interaction_offsets"][[p0, p1]])
    link_detail = {k: detail[k][i0:i1] for k in ("interaction_rank", "interaction_type", "interaction_vertex")}
    link_detail["path_interaction_offsets"] = detail["path_interaction_offsets"][p0:p1 + 1] - i0
    return row, link_core, link_detail

core, detail = load(CARRIER_HZ, "core"), load(CARRIER_HZ, "detail")
row, link_core, link_detail = link_of(core, detail, UE_ID)
print(f"UE {UE_ID}: link row {row} of {core['link_ue_index'].size}, {link_core['tau'].size} paths")"""),
    ("md", """## 6. What each path stores

The same arrays as in the library tutorial: the complex coefficient `a_real + 1j*a_imag`
(passband, without the carrier-delay phase, antennas included), the delay `tau` in seconds,
departure and arrival angles in radians and, in detail, each interaction's type and location
in order. Paths are stored strongest first, at most 16 per link above −150 dB."""),
    ("code", PATH_TABLE + """

describe(link_core, link_detail)"""),
    ("md", """## 7. A power–delay profile

Placing each path by delay and power gives the link's power–delay profile: a path-based view
of the stored channel, not a sampled impulse response."""),
    ("code", PDP + """

fig, ax = plt.subplots(figsize=(7, 3))
stems(ax, link_core["tau"], link_core["a_real"], link_core["a_imag"], "3.5 GHz", "o", "#e2733d")
ax.set(xlabel="delay (ns)", ylabel="path power (dB)", ylim=(-150, None),
       title=f"{SCENE_ID} · BS {BS_ID} → UE {UE_ID}")
ax.legend()
plt.show()"""),
    ("md", """## 8. The same link at 7 GHz"""),
    ("code", """other = link_of(load(OTHER_HZ, "core"), load(OTHER_HZ, "detail"), UE_ID)
fig, ax = plt.subplots(figsize=(7, 3))
stems(ax, link_core["tau"], link_core["a_real"], link_core["a_imag"], "3.5 GHz", "o", "#e2733d")
if other is None:
    print("UE", UE_ID, "has no retained link at 7 GHz")
else:
    _, core7, _ = other
    stems(ax, core7["tau"], core7["a_real"], core7["a_imag"], "7 GHz", "s", "#4d9688")
    for c, label in ((link_core, "3.5 GHz"), (core7, "7 GHz")):
        power = np.sum(c["a_real"].astype(np.float64) ** 2 + c["a_imag"].astype(np.float64) ** 2)
        print(f"{label}: {c['tau'].size} paths, link power {10 * np.log10(power):.2f} dB")
ax.set(xlabel="delay (ns)", ylabel="path power (dB)", ylim=(-150, None))
ax.legend()
plt.show()"""),
    ("md", """Ranks order the paths within one record, so the same rank at 3.5 and 7 GHz need not be
the same route. Compare delays, interaction sequences or interaction points instead.

## Next

- [Without the library](without-library.md): TAR archives, catalogs and byte-range reads.
- [Schema reference](schema-reference.md): every array, its dtype and its meaning."""),
]

ML_GUIDE = [
    ("md", """# From channel records to a DataLoader

This guide turns Channels4World records into fixed-size arrays and a PyTorch `DataLoader`,
with train, validation and test sets split by scene. It ends at one batch whose contents
are explained, so you can adapt the inputs and targets to your model, or swap these
batches into an existing training loop. The [notebook version](notebooks/03-dataloader.ipynb)
runs top to bottom.

The example uses the 100 urban scenes of Tiny (90 train, 10 test), BS 0 at 3.5 GHz. Change the
selection to scale up; the steps stay the same. Install the library with `pip install channels4world`
(Python 3.11+), then `pip install torch huggingface_hub`."""),
    ("code", """from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset

import channels4world as c4w
from channels4world import transforms as T

REPO, REVISION = "Intelligence-Networking-Lab/Channels4World", "v1.0.0"
DATA = Path("c4w")"""),
    ("md", """## 1. Choose the records

One **sample** here is one retained link: a (scene, carrier, BS, UE) channel. A record holds
every retained link of one BS at one carrier, so records are read whole and split into
links. First download Tiny's catalog and the metadata of the five groups that hold its urban
scenes (about 42 MB), so the library can plan exactly which record files the selection needs:
here about 0.4 GB."""),
    ("code", """snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=DATA, allow_patterns=[
    "tiny/c4w-root.json", "tiny/c4w-root/*", "tiny/schema/*",
    "tiny/groups/*-urban-g*/catalog/members.parquet", "tiny/groups/*-urban-g*/snapshot.json"])

ds = c4w.open(DATA / "tiny")
selection = ds.select(profile="urban", frequencies_hz=[3_500_000_000], bs_ids=[0],
                      components=("core", "positions"))
report = selection.missing()
print(report)
if report.files:
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=report.local_dir,
                      allow_patterns=report.repo_files)
assert selection.missing().complete"""),
    ("md", """## 2. Split by scene

Keep the release's train and test split: test scenes are places no training scene shows.
Take validation scenes from the training split, by scene and never by link, so links of one
place never sit on both sides. Hashing the scene ID makes the choice reproducible, and
saving the lists records it."""),
    ("code", """def is_validation(scene_id, fraction=0.1):
    return int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 1000 < fraction * 1000

splits = {"train": [], "validation": [], "test": []}
for scene in selection.scenes():
    part = "test" if scene.split == "test" else ("validation" if is_validation(scene.id) else "train")
    splits[part].append(scene.id)
print({k: len(v) for k, v in splits.items()})
Path("splits.json").write_text(json.dumps(splits, indent=1))"""),
    ("md", """## 3. From a record to arrays

Each record becomes arrays with one row per link. This is where your own preprocessing goes:
`inputs` here are the UE position relative to the BS and the BS height; `targets` are the
strongest 16 paths' delays (ns), powers (dB) and arrival angles, padded to fixed size with a
mask that marks stored paths. Replace either function to fit your task. `LINKS_PER_RECORD`
bounds memory by sampling links reproducibly."""),
    ("code", """LINKS_PER_RECORD = 512
PREPROCESSING = "v1: dxyz+bs_z -> 16 x (delay ns, power dB, cos/sin AoA az, AoA zenith)"

def inputs(record, positions):
    ue = positions.ue[record.ue_ids].astype(np.float32)
    bs = positions.bs[record.bs_id].astype(np.float32)
    return np.concatenate([ue - bs, np.full((len(ue), 1), bs[2], np.float32)], axis=1)   # (links, 4)

def targets(record):
    core = record.core
    power = 10 * np.log10(core["a_real"].astype(np.float64) ** 2 + core["a_imag"].astype(np.float64) ** 2)
    per_path = np.stack([core["tau"] * 1e9, power, np.cos(core["phi_r"]), np.sin(core["phi_r"]),
                         core["theta_r"]], axis=1).astype(np.float32)           # (paths, 5)
    padded, mask = T.pad_paths(core, per_path)                                   # (links, 16, 5), (links, 16)
    return padded, mask

def arrays(scene_ids):
    rng = np.random.default_rng(0)
    x, y, m, ids = [], [], [], []
    for scene in selection.scenes():
        if scene.id not in scene_ids:
            continue
        positions = scene.read_positions()
        for record in scene.records():
            keep = np.sort(rng.permutation(record.n_links)[:LINKS_PER_RECORD])
            padded, mask = targets(record)
            x.append(inputs(record, positions)[keep]); y.append(padded[keep]); m.append(mask[keep])
            ids += [(scene.id, record.frequency_hz, record.bs_id, int(u)) for u in record.ue_ids[keep]]
    return {"x": np.concatenate(x), "y": np.concatenate(y), "mask": np.concatenate(m),
            "ids": np.array(ids, dtype=object)}"""),
    ("md", """## 4. Cache, then normalize with training statistics only

Converting records takes a while, so save the arrays once. The cache key covers the release,
the selection, the scene lists of the split and the preprocessing version: change any of them
(including the validation fraction) and a new cache is built. Loading checks that every cached
row comes from a scene of its part.
Normalization is fitted on the training split and then applied unchanged to validation and
test, so no information leaks from held-out scenes. Target statistics use stored paths only,
and padded entries stay zero. Keep the statistics to turn predictions back into units."""),
    ("code", """recipe = c4w.recipe.dumps(selection.to_recipe(revision=REVISION))
split_lists = json.dumps(splits, sort_keys=True)
key = hashlib.sha256(
    f"{ds.release}|{recipe}|{split_lists}|{PREPROCESSING}|{LINKS_PER_RECORD}".encode()).hexdigest()[:16]
cache = Path("cache") / key
data = {}
for part, scene_ids in splits.items():
    path = cache / f"{part}.npz"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        a = arrays(set(scene_ids))
        np.savez(path, x=a["x"], y=a["y"], mask=a["mask"], ids=a["ids"].astype(str))
    with np.load(path) as z:
        data[part] = {k: z[k] for k in z.files}
    assert set(data[part]["ids"][:, 0]) <= set(scene_ids), f"cached {part} rows from other scenes"
    print(part, data[part]["x"].shape, data[part]["y"].shape)

train = data["train"]
stored = train["y"][train["mask"]]                                  # (stored paths, 5)
stats = {"x_mean": train["x"].mean(axis=0), "x_std": train["x"].std(axis=0) + 1e-6,
         "y_mean": stored.mean(axis=0), "y_std": stored.std(axis=0) + 1e-6}
np.savez(cache / "normalization.npz", **stats)
for part in data.values():
    part["x"] = ((part["x"] - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    y = (part["y"] - stats["y_mean"]) / stats["y_std"]
    part["y"] = np.where(part["mask"][..., None], y, 0).astype(np.float32)"""),
    ("md", """## 5. A Dataset and a DataLoader

Every sample has the same shape, so PyTorch's default collate function batches them. Shuffle
the training set; keep validation and test in order."""),
    ("code", """class LinkDataset(Dataset):
    def __init__(self, part):
        self.x = torch.from_numpy(part["x"])
        self.y = torch.from_numpy(part["y"])
        self.mask = torch.from_numpy(part["mask"])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i], "mask": self.mask[i]}

loaders = {part: DataLoader(LinkDataset(data[part]), batch_size=256, shuffle=(part == "train"))
           for part in data}
batch = next(iter(loaders["train"]))
for name, value in batch.items():
    print(f"{name:5s} {tuple(value.shape)} {value.dtype}")"""),
    ("md", """## 6. What a batch holds

- `x` (batch, 4): the UE position relative to the BS, in metres, and the BS height,
  normalized with training statistics.
- `y` (batch, 16, 5): up to 16 paths per link, strongest first; per path the delay in ns, the
  power in dB relative to unit power, the cosine and sine of the arrival azimuth and the
  arrival zenith in radians, each normalized with the training split's stored paths.
- `mask` (batch, 16): true where a path is stored. Padded paths are zeros: use the mask in
  every loss and metric so they never count as weak paths.

The saved `ids` (scene, carrier, BS, UE) tie each row back to its record."""),
    ("code", """paths = data["train"]["mask"].sum(axis=1)
print("stored paths per link: min", paths.min(), "mean", round(float(paths.mean()), 1), "max", paths.max())

def masked_mse(prediction, target, mask):
    error = ((prediction - target) ** 2).mean(dim=-1)          # (batch, 16)
    return (error * mask).sum() / mask.sum().clamp(min=1)

# Predicting the training mean (zero after normalization) gives a masked MSE near 1.
print("masked MSE of predicting the mean:",
      round(float(masked_mse(torch.zeros_like(batch["y"]), batch["y"], batch["mask"])), 3))"""),
    ("md", """## Adapt it

- **Other inputs:** add the carrier, the scene geometry (`scene.open_geometry()`), or the
  channel at the other carrier for cross-band tasks; keep them per link.
- **Other targets:** use the complex coefficients, departure angles, the frequency response
  (`T.frequency_response`) or the interaction detail (`components=("core", "detail")`,
  `T.pad_interactions`).
- **More data:** widen the selection (`tier="small"` on the release catalog, other profiles or
  BSs) and raise `LINKS_PER_RECORD`; the [library guide](library-guide.md) plans TAR
  downloads, and `c4w.to_cache` keeps records memory-mapped for repeated reads.
- **Shuffling at scale:** records are stored in groups of 25 scenes; shuffle groups or scenes
  and read each one's records together, as above.
- **Records instead of links:** `channels4world.torch.RecordDataset(selection, transform=...)`
  yields one record per item and reopens the dataset in each worker process."""),
]


def link_target(target: str, surface: str, *, absolute: bool = False) -> str:
    """Keep page links local where possible; notebook links must survive HF's CDN."""
    if target.startswith(("#", "/")) or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
        return target
    if surface == "library":
        if target in ("tutorial-numpy.md", "schema-reference.md", "without-library.md"):
            return HF + "docs/" + target
        if target == "library-guide.md":
            target = "guide.md"
        if target.startswith("notebooks/"):
            target = "../" + NOTEBOOK_PATHS[target.split("/")[-1]]
        return urljoin(GH + "docs/", target) if absolute else target
    if target in ("tutorial-library.md", "ml-guide.md"):
        return GH + "docs/" + target
    return urljoin(HF + "docs/", target) if absolute else target


def markdown(text: str, surface: str, *, absolute: bool = False) -> str:
    return re.sub(r"(\]\()([^\s)]+)(\))",
                  lambda m: m[1] + link_target(m[2], surface, absolute=absolute) + m[3], text)


def page(cells, name, surface):
    out = [markdown(t, surface) if k == "md" else f"```python\n{t}\n```" for k, t in cells]
    target = link_target("notebooks/" + name, surface)
    back = "[Library documentation](../README.md)" if surface == "library" else "[Return to card](../README.md)"
    out.append(f"*Notebook: [{Path(NOTEBOOK_PATHS[name]).name}]({target}). The page and the notebook hold the "
               f"same code.*\n\n{back}")
    return "\n\n".join(out) + "\n"


# A page links to its notebook; a notebook does not link to itself. These are the exact
# notebook-only rewordings of each page's sentence about its notebook version.
NOTEBOOK_ONLY = {
    "01-read-one-scene-library.ipynb": (
        "at 7 GHz. The [notebook version](notebooks/01-read-one-scene-library.ipynb) runs top to\n"
        "bottom; [the same tutorial without the library](tutorial-numpy.md) reads the same link with\n"
        "NumPy alone.",
        "at 7 GHz. [The same tutorial without the library](tutorial-numpy.md) reads the same link\n"
        "with NumPy alone."),
    "02-read-one-scene-numpy.ipynb": (
        "files: about 15 MB. The [notebook version](notebooks/02-read-one-scene-numpy.ipynb) runs top\n"
        "to bottom. Archives from larger tiers hold the same files; see\n",
        "files: about 15 MB. Archives from larger tiers hold the same files; see\n"),
    "03-dataloader.ipynb": (
        "batches into an existing training loop. The [notebook version](notebooks/03-dataloader.ipynb)\n"
        "runs top to bottom.\n",
        "batches into an existing training loop.\n"),
}


def notebook(cells, name, surface):
    old, new = NOTEBOOK_ONLY[name]
    if sum(t.count(old) for k, t in cells if k == "md") != 1:
        sys.exit(f"{name}: its page's notebook sentence changed; update NOTEBOOK_ONLY")
    cells = [(k, t.replace(old, new) if k == "md" else t) for k, t in cells]

    def src(text):
        lines = text.split("\n")
        return [line + "\n" for line in lines[:-1]] + [lines[-1]]
    nb = {  # stable cell IDs keep rebuilt notebooks byte-identical
        "cells": [{"cell_type": "markdown", "id": f"cell-{i:02d}", "metadata": {}, "source": src(markdown(t, surface, absolute=True))} if k == "md"
                  else {"cell_type": "code", "execution_count": None, "id": f"cell-{i:02d}", "metadata": {},
                        "outputs": [], "source": src(t)}
                  for i, (k, t) in enumerate(cells)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    return nb


TUTORIALS = [
    ("tutorial-library.md", "01-read-one-scene-library.ipynb", LIB_TUTORIAL, "library"),
    ("tutorial-numpy.md", "02-read-one-scene-numpy.ipynb", NUMPY_TUTORIAL, "huggingface"),
    ("ml-guide.md", "03-dataloader.ipynb", ML_GUIDE, "library"),
]


def outputs(root: Path, card: Path | None = None) -> dict[Path, str]:
    result = {}
    for md, name, cells, surface in TUTORIALS:
        dest = root if surface == "library" else card
        if dest is None:
            continue
        result[dest / "docs" / md] = page(cells, name, surface)
        result[dest / NOTEBOOK_PATHS[name]] = json.dumps(notebook(cells, name, surface), indent=1, ensure_ascii=False) + "\n"
        if surface == "library" and card is not None:
            result[card / "docs" / md] = (
                f"# {cells[0][1].splitlines()[0].lstrip('# ')}\n\n"
                "This library tutorial is now maintained with the channels4world source.\n\n"
                f"- [Read the current tutorial]({GH}docs/{md})\n"
                f"- [Open the current notebook]({GH}{NOTEBOOK_PATHS[name]})\n\n"
                "The previous Hugging Face notebook is retained as a frozen compatibility copy. "
                "Use the GitHub version for updates and fixes.\n\n"
                "[Pure NumPy tutorial](tutorial-numpy.md) · [Dataset card](../README.md)\n"
            )
            # Keep old URLs working without maintaining a second editable tutorial.
            legacy = card / "docs" / "notebooks" / name
            if legacy.exists():
                nb = json.loads(legacy.read_text())
                notice = {"cell_type": "markdown", "id": "library-tutorial-moved", "metadata": {}, "source": [
                    "**Archived compatibility copy.** This notebook is frozen. The maintained version, "
                    f"including future fixes, is [in the channels4world library]({GH}{NOTEBOOK_PATHS[name]}).\n",
                    f"[Read the current tutorial]({GH}docs/{md}). The dataset's "
                    f"[pure NumPy example]({HF}docs/tutorial-numpy.md) remains on Hugging Face."
                ]}
                if nb['cells'] and nb['cells'][0].get('id') == notice['id']:
                    nb['cells'][0] = notice
                else:
                    nb['cells'].insert(0, notice)
                result[legacy] = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-card", type=Path, help="also update the NumPy tutorial and compatibility links in a local dataset card tree")
    ap.add_argument("--check", action="store_true", help="fail if generated files differ, without writing")
    args = ap.parse_args()
    failures = []
    for path, expected in outputs(ROOT, args.hf_card).items():
        if args.check:
            if not path.exists() or path.read_text() != expected:
                failures.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
            print("wrote", path)
    for path in failures:
        print("out of date:", path)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
