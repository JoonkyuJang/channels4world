# Read one scene with the channels4world library

This tutorial follows one link of Channels4World from the catalog down to its paths:
**BS 0 → UE 14 of `urban_scene_019234` at 3.5 GHz**, the link the project website shows.
It downloads about 26 MB of one Tiny scene, walks the hierarchy, reads the geometry, the
positions and the channel record, explains the stored arrays, and ends with the same link
at 7 GHz. The [notebook version](../examples/01-read-one-scene-library.ipynb) runs top to
bottom; [the same tutorial without the library](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World/blob/main/docs/tutorial-numpy.md) reads the same link with
NumPy alone.

Install the library with `pip install channels4world` (Python 3.11+), then
`pip install huggingface_hub matplotlib`.

```python
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
from matplotlib.collections import PolyCollection

import channels4world as c4w
```

## 1. Choose the link

Change these to read any other link. `GROUP` is the scene's `group` in the `tiny_scenes`
catalog. BS IDs run from 0 to 31 and UE IDs from 0 to 16,383; both are rows
of the scene's positions. Carriers are integers in Hz.

```python
REPO, REVISION = "Intelligence-Networking-Lab/Channels4World", "v1.0.0"
SCENE_ID, GROUP = "urban_scene_019234", "test-urban-g00000"  # GROUP: from the tiny_scenes catalog
CARRIER_HZ, BS_ID, UE_ID = 3_500_000_000, 0, 14
OTHER_HZ = 7_000_000_000
DATA = Path("c4w")
```

## 2. Download what this tutorial reads

Tiny is published as loose files under `tiny/`. The library needs Tiny's catalog, the shared
schemas and the scene's group metadata (about 11 MB); the records are the scene's positions
and geometry and BS 0's core and detail at both carriers (about 15 MB).

```python
scene_dir = f"tiny/records/scene={SCENE_ID}"
patterns = [
    "tiny/c4w-root.json", "tiny/c4w-root/*", "tiny/schema/*",
    f"tiny/groups/{GROUP}/catalog/members.parquet", f"tiny/groups/{GROUP}/snapshot.json",
    f"{scene_dir}/positions.npz", f"{scene_dir}/geometry.zip",
    f"{scene_dir}/band_hz=*/component=*/bs{BS_ID:05d}.npz",
]
snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=DATA, allow_patterns=patterns)
```

## 3. Walk the hierarchy

The dataset is organized as **dataset → scene → carrier → base station → UE link → paths**.
Opening reads only the small catalog; group metadata and records load when first needed.

```python
ds = c4w.open(DATA / "tiny")
print(ds)
scene = ds.scene(SCENE_ID)
print(scene)                                        # split, profile and smallest tier
print("carriers (Hz):", scene.frequencies.ids)
print("base stations:", scene.frequencies[CARRIER_HZ].base_stations.ids)
```

`urban_scene_019234` is a **test** scene: every link and both carriers of a scene stay in
one split, so a model trained on the training split has never seen this place.

## 4. Geometry and positions

Geometry, positions and interaction points share one frame: scene metres, z up. Positions
hold all 32 BS and 16,384 UE coordinates of the scene; the geometry is a Mitsuba XML file
with OBJ meshes in a ZIP.

```python
positions = scene.read_positions()
bs_positions, ue_positions = positions.bs, positions.ue
bs_xyz, ue_xyz = bs_positions[BS_ID], ue_positions[UE_ID]
print(positions)
print("BS", bs_xyz, "UE", ue_xyz, f"distance {np.linalg.norm(ue_xyz - bs_xyz):.1f} m")

def top_view(read, names):
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
    return triangles

with scene.open_geometry() as geometry:
    print("entry point:", json.loads(geometry.read("scene.json"))["entrypoint"])
    meshes = [n for n in geometry.namelist() if n.endswith(".obj") and "ground" not in n]
    triangles = top_view(geometry.read, meshes)
print(len(meshes), "building meshes")
```

```python
fig, ax = plt.subplots(figsize=(6, 6))
ax.add_collection(PolyCollection(triangles, facecolor="#c3cbc2", edgecolor="none"))
ax.scatter(ue_positions[:, 0], ue_positions[:, 1], s=0.2, color="#4d9688", label=f"{len(ue_positions):,} UEs")
ax.scatter(bs_positions[:, 0], bs_positions[:, 1], marker="^", s=30, color="k", label=f"{len(bs_positions)} BSs")
ax.plot(*zip(bs_xyz[:2], ue_xyz[:2]), color="#e2733d", lw=2, label=f"BS {BS_ID} → UE {UE_ID}")
ax.set(xlim=(-250, 250), ylim=(-250, 250), aspect="equal", xlabel="x (m)", ylabel="y (m)", title=SCENE_ID)
ax.legend(loc="upper right")
plt.show()
```

## 5. One channel record

A record holds every retained link from one BS at one carrier. `get_link(UE_ID)` returns
that UE's link, or `None` when the UE has no retained link to this BS (which is not a
zero-valued channel). A UE ID is a row of the positions; the link's row inside the record
is a different index.

```python
record = scene.frequencies[CARRIER_HZ].base_stations[BS_ID].read(detail=True)
print(record)
link = record.get_link(UE_ID)
print(link, f"- row {link.index} of the record's {record.n_links} links")
```

## 6. What each path stores

For every retained path the core record stores the complex coefficient `a_real + 1j*a_imag`
(passband, without the carrier-delay phase, antennas included), the delay `tau` in seconds,
departure and arrival angles in radians, and its interaction count and type flags; the detail
record adds each interaction's type and location in order. Paths are stored strongest first,
and each link keeps at most its 16 strongest paths above −150 dB.

```python
NAMES = {1: "specular reflection", 2: "diffuse reflection", 4: "refraction", 8: "diffraction"}

def describe(core, detail):
    """One line per retained path, strongest first."""
    re, im = core["a_real"].astype(np.float64), core["a_imag"].astype(np.float64)
    power_db = 10 * np.log10(re**2 + im**2)
    off = detail["path_interaction_offsets"]
    for p in range(len(core["tau"])):
        kinds = [NAMES.get(int(t), str(t)) for t in detail["interaction_type"][off[p]:off[p + 1]]]
        print(f"rank {int(core['path_rank'][p]):2d}  {core['tau'][p] * 1e9:8.2f} ns  {power_db[p]:7.2f} dB  "
              f"AoD az {np.degrees(core['phi_t'][p]):6.1f}°  AoA az {np.degrees(core['phi_r'][p]):6.1f}°  "
              f"{' → '.join(kinds) or 'line of sight'}")

describe(link.core, link.detail)
```

## 7. A power–delay profile

Placing each path by delay and power gives the link's power–delay profile: a path-based view
of the stored channel. It is not a sampled impulse response; for a channel at a given
bandwidth, `channels4world.transforms` provides baseband coefficients and frequency
responses.

```python
def stems(ax, tau, a_real, a_imag, label, marker, color):
    power_db = 10 * np.log10(a_real.astype(np.float64)**2 + a_imag.astype(np.float64)**2)
    ax.vlines(tau * 1e9, -150, power_db, color=color, lw=1.2)
    ax.scatter(tau * 1e9, power_db, marker=marker, color=color, zorder=3, label=label)

fig, ax = plt.subplots(figsize=(7, 3))
stems(ax, link.core["tau"], link.core["a_real"], link.core["a_imag"], "3.5 GHz", "o", "#e2733d")
ax.set(xlabel="delay (ns)", ylabel="path power (dB)", ylim=(-150, None),
       title=f"{SCENE_ID} · BS {BS_ID} → UE {UE_ID}")
ax.legend()
plt.show()
```

## 8. The same link at 7 GHz

Every scene is simulated at both carriers with the same BS and UE positions. Read the same
BS and UE at 7 GHz and compare on shared axes:

```python
link7 = scene.frequencies[OTHER_HZ].base_stations[BS_ID].read(detail=True).get_link(UE_ID)
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
plt.show()
```

Ranks order the paths within one record: rank 0 at 7 GHz need not be the route of rank 0
at 3.5 GHz. To match routes across carriers, compare delays, interaction sequences or
interaction points (`link7.detail`), not ranks.

## Next

- [From channel records to a DataLoader](ml-guide.md): link samples, splits, padding and batches.
- [Library guide](guide.md): download plans for larger tiers, recipes, extraction and caches.
- [Schema reference](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World/blob/main/docs/schema-reference.md): every array and its meaning.

*Notebook: [01-read-one-scene-library.ipynb](../examples/01-read-one-scene-library.ipynb). The page and the notebook hold the same code.*

[Library documentation](../README.md)
