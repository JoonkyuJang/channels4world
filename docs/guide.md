# Guide

The library reads local folders: a downloaded release, Tiny, a subset you extracted or
an NPY cache. Hugging Face handles network transfers. The examples follow one link:
base station 0 and UE 14 of `urban_scene_019234` at 3.5 GHz.

## Download and availability

The [dataset card](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World)
has the download commands, the tier and component sizes, the full schema and access
without the library.

### One Tiny scene

Download BS 0 of one scene with the [README's command](../README.md#quick-start-download-and-read-one-scene)
(about 26 MB); it includes the catalog files the library needs. Then open that scene by
its ID:

```python
import channels4world as c4w

ds = c4w.open("c4w/tiny")
scene = ds.scene("urban_scene_019234")
record = scene.frequencies[3_500_000_000].base_stations[0].read()
positions = scene.read_positions()
```

Opening reads `c4w-root.json` and the tables it references; group metadata and payloads
load when first needed. A partial download holds only what you fetched, so open the
scenes and BSs you downloaded.

### Plan a TAR download from the catalog

Download the release catalog (`c4w-root.json` and `c4w-root/`) first. No group metadata
or payloads are needed for the first plan:

```python
import channels4world as c4w

ds = c4w.open("c4w")
selection = ds.select(tier="small", frequencies_hz=[3_500_000_000],
                      components=("core", "positions"))
report = selection.missing()
print(report)
```

Before group metadata is local, the plan lists the matching archives and the metadata
they need. Archives are the download unit for TAR releases. Once the metadata arrives,
`missing()` resolves individual records, including requests the release does not host.

For a published root, the report gives a download command. Pass the same revision you
downloaded the catalog from, so every file comes from one release:

```python
if report.files:
    print(report.hf_command(revision="v1.0.0"))
```

For long file lists, use the Hugging Face Python client:

```python
from huggingface_hub import snapshot_download

if report.files:
    snapshot_download(
        repo_id="Intelligence-Networking-Lab/Channels4World", repo_type="dataset",
        revision="v1.0.0", local_dir=report.local_dir, allow_patterns=report.repo_files,
    )
report = selection.missing()
if not report.complete:
    raise RuntimeError(f"Selection is not ready: {report}")
```

`report.files` are relative to the opened root; `report.repo_files` are repository
paths, and `report.local_dir` is the folder that corresponds to the repository's top
level. These helpers refuse subsets and caches you made yourself, which are not
published roots; their root-relative `files` still work in your own workflow.

`bool(report)` is true when files need downloading. `report.complete` is true when
nothing needs downloading and everything requested is hosted. An item the release does
not host, such as detail for a scene outside the Medium tier, is listed in
`report.unpublished`; downloading more of the same release cannot provide it. Reports
check that files are present; contents are verified when they are read.

## Navigate and select

```python
scene = ds.scene("urban_scene_019234")   # lookup across all splits
split = ds.splits[scene.split]
print(scene.profile, scene.tier)
```

Scene and split IDs are strings, carriers are integer Hz and BS IDs are integers.
Collections iterate over objects and expose their keys as `.ids`. `scene.tier` is the
smallest tier that contains the scene; larger tiers contain it too.

Selections accept `split`, `tier`, `profile`, `scene_ids`, `frequencies_hz`, `bs_ids`
and `components`. `profile` uses the catalog spelling (`ds.profiles`: `urban`,
`metropolitan`, `suburban`; scene IDs use `metro_`). The default components are core,
positions and geometry. Adding detail requests it for the same channel records and
includes core.

```python
selection = ds.select(
    scene_ids=[scene.id], frequencies_hz=[3_500_000_000], bs_ids=[0],
    components=("core", "detail", "positions"),
)
report = selection.missing()
if report.complete:
    for selected_scene in selection.scenes():
        positions = selected_scene.read_positions()
        for record in selected_scene.records():
            print(record.scene_id, record.bs_id, record.n_links)
```

Scenes come in catalog order and channel records in ascending carrier and BS order. A
requested component that is missing or not hosted is reported, or raises when read;
iteration never silently drops it.

### Use only what is downloaded

`selection.local()` narrows a selection on purpose. A channel record stays only when
every requested channel component is usable locally; positions and geometry are kept
independently. Scenes with nothing usable, or without their group metadata, are left
out. Use it for deliberate partial-data work, not as a completeness check.

### Share a selection as a recipe

A recipe is a small JSON file that records a selection and the release it refers to. It
holds no paths or commands, so the recipient plans the download from their own catalog;
the same recipe works with the release root and with Tiny.

```python
recipe = selection.to_recipe(revision="v1.0.0")
with open("selection.json", "w") as fh:
    fh.write(c4w.recipe.dumps(recipe))

same = ds.select_recipe("selection.json")
assert same.scene_ids == selection.scene_ids
```

The project website's download helper writes the same format. `select_recipe` refuses a
recipe for another release and checks every filter against the catalog, exactly as
`select()` does. A `local()` selection cannot be exported, because it depends on what one
machine has downloaded. On the command line:

```bash
c4w recipe c4w --scenes urban_scene_019234 --frequency 3500000000 --bs 0 \
  --revision v1.0.0 -o one-bs.json
c4w missing c4w --recipe one-bs.json     # the recipe's revision pins the printed command
```

## Records, links and UE IDs

A record holds read-only `core` arrays and, when requested, `detail` arrays. UE IDs are
rows of the scene's positions; a link's row inside the record is a different index:

```python
link = record.get_link(14)
if link is not None:
    delays_s = link.core["tau"]
    gains = link.core["a_real"] + 1j * link.core["a_imag"]
    bs_xyz = positions.bs[record.bs_id]
    ue_xyz = positions.ue[link.ue_id]
```

`get_link(ue_id)` returns `None` when that UE has no retained link to this BS; that is
not a zero-valued channel. An ID outside the scene's UEs raises `UnknownIDError`.
`record.links()` iterates every retained link.

Arithmetic returns new arrays. To change stored values, copy them first:
`record.copy()` gives dictionaries of writable arrays, or copy one array with
`record.core["tau"].copy()`.

## Positions and geometry

Positions and geometry can be read without channel payloads:

```python
positions = scene.read_positions()
with scene.open_geometry() as geometry:
    print(geometry.namelist())
```

`open_geometry()` returns an in-memory `ZipFile`. To give a renderer the scene XML and
meshes on disk, extract them:

```python
xml_path = scene.extract_geometry("geometry-output")
```

`read_*` loads data into memory, `open_*` returns a handle and `extract_*` writes files.

## Extract a subset or make a cache

TAR and loose payloads can both be read directly; preparation is optional. Start from
a complete selection and choose new destinations:

```python
if not selection.missing().complete:
    raise RuntimeError("Download the selection first.")

c4w.extract(selection, "selected-loose", verify="basic", check_space=True)
subset = c4w.open("selected-loose")
c4w.to_cache(selection, "selected-npy", verify="basic", check_space=True)
cache = c4w.open("selected-npy")
```

Extraction copies the published NPZ and ZIP files to their member paths and writes a
subset root. NPY caches keep values, dtypes, IDs and the offset layout; numeric arrays
are memory-mapped when read and geometry stays ZIP.

Extract from a release or a loose subset, not from a cache. `skip_existing=True` resumes
or extends an earlier extraction: select everything the destination already holds plus
what to add (for example both carriers). A narrower selection is refused rather than
dropping earlier contents. Caches need a new destination. Python calls accept a
`confirm` callback that receives the `Plan`; the CLI prints the estimate and asks
before writing.

## Verification

The default is `verify="basic"`: published bytes, declared arrays, dtypes and shapes.
`none` skips payload checks; `full` adds scientific checks of the components being read.
Group metadata is always checked against the root when first loaded. A full read of core
and detail together also checks how they correspond. See the
[verification table](api.md#verification).

## Transforms

```python
import numpy as np
from channels4world import transforms as T

rms_delay_s = T.delay_spread(record)
response = T.frequency_response(record, np.array([-1e6, 0.0, 1e6]))   # offsets from the carrier
```

Transforms describe the retained paths and return new writable arrays. Stored
coefficients exclude the carrier-delay phase; read the
[scientific conventions](conventions.md) before comparing responses or using another
simulator.

### Fixed-size arrays

Each link stores up to 16 paths and each path up to 4 interactions, in flat arrays with
offsets. `pad_paths` and `pad_interactions` give one row per link or per path, padded
with zeros, with a mask that marks stored entries:

```python
tau, mask = T.pad_paths(record.core, "tau")              # (links, 16), (links, 16)
ue_ids = record.ue_ids                                   # row i is UE ue_ids[i]

full = scene.frequencies[3_500_000_000].base_stations[0].read(detail=True)
vertices, vmask = T.pad_interactions(full.detail, "interaction_vertex")   # (paths, 4, 3)
per_link, per_link_mask = T.pad_paths(full.core, vertices)                # (links, 16, 4, 3)
```

They work on any mapping of arrays: library records, opened NPZ files and HF Datasets
rows with restored shapes, and they match the dataset card's NumPy snippet. Padding
never truncates; a row longer than the requested width raises `ValueError`. Use the
mask, not the zero values, to tell padding apart.

## PyTorch

PyTorch is optional. `RecordDataset` turns a selection into a map-style dataset that
reads one channel record per item:

```python
from torch.utils.data import DataLoader
from channels4world.torch import RecordDataset

data = RecordDataset(selection)
loader = DataLoader(data, batch_size=None)                  # one Record per step
# Ragged records as lists: DataLoader(data, batch_size=8, collate_fn=list)
```

The record list is built from metadata, and records are read when indexed. Worker
processes reopen the root on their own; with spawned workers, use the usual
`if __name__ == "__main__":` guard. Copy a stored array before `torch.from_numpy`
(transform outputs are already writable). The default collate function cannot batch
`Record` objects, so pass `batch_size=None`, `collate_fn=list` or a `transform` that
returns tensors.

A record holds every retained link of one BS, so most models train on links instead.
The dataset card's [ML guide](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World/blob/v1.0.0/docs/ml-guide.md)
builds a link-level dataset with train, validation and test scenes, padding, an
optional cache and a `DataLoader`.

Records are stored in groups of 25 scenes, and the library keeps the metadata of the
four most recently used groups in memory (about 11 MB each). Reading records in a fully
random order across many groups reloads group metadata again and again (about 0.06 s per
reload). For shuffled training, shuffle the order of groups or scenes and read each
one's records together, or convert once to your own training format.

[README](../README.md) · [API and CLI](api.md) · [Scientific conventions](conventions.md)
