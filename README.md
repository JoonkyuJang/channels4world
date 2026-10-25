# channels4world library

**The environment is part of the channel.** Channels4World pairs outdoor RF
scenes with stored channels across 100,000 environments. This Python library
helps you select, read and prepare those records locally. It plans downloads,
reads values and dtypes as read-only NumPy arrays, and can extract subsets or
build memory-mapped caches; it performs no network transfers.

[Dataset card](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World) ·
[Download helper](https://cbchae.yonsei.ac.kr/channels4world/download.html) ·
[Guide](https://github.com/JoonkyuJang/channels4world/blob/main/docs/guide.md) ·
[API and CLI](https://github.com/JoonkyuJang/channels4world/blob/main/docs/api.md)

## Install

Requires Python **3.11+**, NumPy **1.26+** and PyArrow **14+**. PyTorch is optional.

```bash
python -m pip install channels4world
python -m pip install "channels4world[torch]"   # with the optional PyTorch adapter
python -m pip install huggingface_hub           # the hf command, for the downloads below
```

Hugging Face's `hf` command or `snapshot_download` fetches the planned files;
then open the local folder with the library.

## Quick start: download and read one scene

The examples follow one link throughout: base station 0 and UE 14 of
`urban_scene_019234` at 3.5 GHz, the same link the project website shows. Download BS 0
of that Tiny scene with the metadata the library needs (about 26 MB):

```bash
hf download Intelligence-Networking-Lab/Channels4World --repo-type dataset \
  --revision v1.0.0 --local-dir c4w \
  --include 'tiny/c4w-root.json' --include 'tiny/c4w-root/*' --include 'tiny/schema/*' \
  --include 'tiny/groups/test-urban-g00000/catalog/members.parquet' \
  --include 'tiny/groups/test-urban-g00000/snapshot.json' \
  --include 'tiny/records/scene=urban_scene_019234/positions.npz' \
  --include 'tiny/records/scene=urban_scene_019234/geometry.zip' \
  --include 'tiny/records/scene=urban_scene_019234/band_hz=*/component=*/bs00000.npz'
```

For all 32 BSs of the scene (about 418 MB), replace the last three `--include` patterns
with `'tiny/records/scene=urban_scene_019234/*'`. Then read the link:

```python
import channels4world as c4w

ds = c4w.open("c4w/tiny")
scene = ds.scene("urban_scene_019234")
record = scene.frequencies[3_500_000_000].base_stations[0].read()
positions = scene.read_positions()

print(scene.split, scene.profile, scene.tier)   # test urban tiny
link = record.get_link(14)                      # This example has a retained link
assert link is not None
print(link.n_paths, link.core["tau"])           # 16 retained paths and their delays, s
```

The hierarchy is **dataset → scene → carrier → base station → UE link → paths**. Scenes
also belong to a split (`ds.splits`), and `ds.scene(id)` works across splits. Scene IDs are strings, carriers are integer Hz, and
BS and UE IDs are integers. A UE ID is a row of `positions.ue`; it is not the link's row
in the record. Collections iterate over objects; `.ids` gives their keys.

## Select what to read

Small and larger tiers are TAR archives. Download the release catalog, then choose scenes,
carriers, BSs and components:

```bash
hf download Intelligence-Networking-Lab/Channels4World --repo-type dataset \
  --revision v1.0.0 --local-dir c4w --include 'c4w-root.json' --include 'c4w-root/*'
```

```python
import channels4world as c4w
from huggingface_hub import snapshot_download

ds = c4w.open("c4w")
selection = ds.select(tier="small", profile="urban", frequencies_hz=[3_500_000_000],
                      components=("core", "positions"))
report = selection.missing()
print(report)  # files to download and their total size; check it before downloading

if report.files:
    snapshot_download("Intelligence-Networking-Lab/Channels4World", repo_type="dataset",
                      revision="v1.0.0", local_dir=report.local_dir,
                      allow_patterns=report.repo_files)
assert selection.missing().complete
for record in selection.records():
    print(record.scene_id, record.frequency_hz, record.bs_id, record.n_paths)
```

`missing()` plans TAR downloads from the catalog before any group metadata is local, and
`report.hf_command(revision="v1.0.0")` prints the same plan as an `hf download` command.
Requested items the release does
not host, such as detail outside the Medium tier, are listed as unpublished rather than
dropped. Use `.local()` only to narrow a selection deliberately to what is downloaded.

Save a selection as a small JSON **recipe** to share it or to rerun it later. A recipe
records the request, not file paths, so each user plans from their own catalog:

```python
recipe = selection.to_recipe(revision="v1.0.0")
same = ds.select_recipe(recipe)            # or the path of a saved recipe file
```

```bash
c4w recipe c4w --tier small --profile urban --frequency 3500000000 \
  --components core,positions --revision v1.0.0 -o small-urban.json
c4w missing c4w --recipe small-urban.json --revision v1.0.0
```

## Prepare, transform and train

Downloaded TARs can be read directly. For repeated access, make a loose subset or an
NPY cache:

```python
c4w.extract(selection, "small-loose")
c4w.to_cache(selection, "small-npy")
cache = c4w.open("small-npy")
```

Every root opens with the same API. Optional [transforms](https://github.com/JoonkyuJang/channels4world/blob/main/docs/conventions.md) give
complex gains, path powers, delay spreads, baseband coefficients, frequency responses
and fixed-size padded arrays. A thin [PyTorch adapter](https://github.com/JoonkyuJang/channels4world/blob/main/docs/guide.md#pytorch) turns a
selection into a dataset for `DataLoader`.

## Documentation

- [Guide](https://github.com/JoonkyuJang/channels4world/blob/main/docs/guide.md): downloads, selection, records, geometry, recipes, extraction,
  padding and PyTorch.
- [API and CLI](https://github.com/JoonkyuJang/channels4world/blob/main/docs/api.md): every public entry point, verification and errors.
- [Scientific conventions](https://github.com/JoonkyuJang/channels4world/blob/main/docs/conventions.md): coefficients, delays, antennas and ranks.
- Tutorials on the dataset card, each a page and a runnable notebook:
  [Read one scene](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World/blob/main/docs/tutorial-library.md)
  ([notebook](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World/blob/main/docs/notebooks/01-read-one-scene-library.ipynb)) and
  [From channel records to a DataLoader](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World/blob/main/docs/ml-guide.md)
  ([notebook](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World/blob/main/docs/notebooks/03-dataloader.ipynb)).
- [Development](https://github.com/JoonkyuJang/channels4world/blob/main/docs/development.md): source layout, tests, building the package and the root format.

The [dataset card](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World) holds the full schema, the catalogs
and NumPy-only instructions.
The library version, the root format (version 1) and the dataset release are separate
identifiers.

## Help and contributing

For usage questions and bug reports, use the [GitHub issues page](https://github.com/JoonkyuJang/channels4world/issues).
Contributors can start with the Development guide listed under Documentation.

## Maintainer, licenses and citation

Copyright 2026 Joonkyu Jang. Maintained by **Joonkyu Jang**; see
[COPYRIGHT](https://github.com/JoonkyuJang/channels4world/blob/v1.0.0/COPYRIGHT).

The software is **Apache-2.0**; see [LICENSE](https://github.com/JoonkyuJang/channels4world/blob/v1.0.0/LICENSE).
The Channels4World data is separately **CC BY 4.0**. For data attribution and the
submitted dataset-paper citation (arXiv preprint pending), see the
[dataset card](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World#license-and-citation).
