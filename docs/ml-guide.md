# From channel records to a DataLoader

This guide turns Channels4World records into fixed-size arrays and a PyTorch `DataLoader`,
with train, validation and test sets split by scene. It ends at one batch whose contents
are explained, so you can adapt the inputs and targets to your model, or swap these
batches into an existing training loop. The [notebook version](../examples/02-dataloader.ipynb)
runs top to bottom.

The example uses the 100 urban scenes of Tiny (90 train, 10 test), BS 0 at 3.5 GHz. Change the
selection to scale up; the steps stay the same. Install the library with `pip install channels4world`
(Python 3.11+), then `pip install torch huggingface_hub`.

```python
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset

import channels4world as c4w
from channels4world import transforms as T

REPO, REVISION = "Intelligence-Networking-Lab/Channels4World", "v1.0.0"
DATA = Path("c4w")
```

## 1. Choose the records

One **sample** here is one retained link: a (scene, carrier, BS, UE) channel. A record holds
every retained link of one BS at one carrier, so records are read whole and split into
links. First download Tiny's catalog and the metadata of the five groups that hold its urban
scenes (about 42 MB), so the library can plan exactly which record files the selection needs:
here about 0.4 GB.

```python
snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=DATA, allow_patterns=[
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
assert selection.missing().complete
```

## 2. Split by scene

Keep the release's train and test split: test scenes are places no training scene shows.
Take validation scenes from the training split, by scene and never by link, so links of one
place never sit on both sides. Hashing the scene ID makes the choice reproducible, and
saving the lists records it.

```python
def is_validation(scene_id, fraction=0.1):
    return int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 1000 < fraction * 1000

splits = {"train": [], "validation": [], "test": []}
for scene in selection.scenes():
    part = "test" if scene.split == "test" else ("validation" if is_validation(scene.id) else "train")
    splits[part].append(scene.id)
print({k: len(v) for k, v in splits.items()})
Path("splits.json").write_text(json.dumps(splits, indent=1))
```

## 3. From a record to arrays

Each record becomes arrays with one row per link. This is where your own preprocessing goes:
`inputs` here are the UE position relative to the BS and the BS height; `targets` are the
strongest 16 paths' delays (ns), powers (dB) and arrival angles, padded to fixed size with a
mask that marks stored paths. Replace either function to fit your task. `LINKS_PER_RECORD`
bounds memory by sampling links reproducibly.

```python
LINKS_PER_RECORD = 512
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
            "ids": np.array(ids, dtype=object)}
```

## 4. Cache, then normalize with training statistics only

Converting records takes a while, so save the arrays once. The cache key covers the release,
the selection, the scene lists of the split and the preprocessing version: change any of them
(including the validation fraction) and a new cache is built. Loading checks that every cached
row comes from a scene of its part.
Normalization is fitted on the training split and then applied unchanged to validation and
test, so no information leaks from held-out scenes. Target statistics use stored paths only,
and padded entries stay zero. Keep the statistics to turn predictions back into units.

```python
recipe = c4w.recipe.dumps(selection.to_recipe(revision=REVISION))
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
    part["y"] = np.where(part["mask"][..., None], y, 0).astype(np.float32)
```

## 5. A Dataset and a DataLoader

Every sample has the same shape, so PyTorch's default collate function batches them. Shuffle
the training set; keep validation and test in order.

```python
class LinkDataset(Dataset):
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
    print(f"{name:5s} {tuple(value.shape)} {value.dtype}")
```

## 6. What a batch holds

- `x` (batch, 4): the UE position relative to the BS, in metres, and the BS height,
  normalized with training statistics.
- `y` (batch, 16, 5): up to 16 paths per link, strongest first; per path the delay in ns, the
  power in dB relative to unit power, the cosine and sine of the arrival azimuth and the
  arrival zenith in radians, each normalized with the training split's stored paths.
- `mask` (batch, 16): true where a path is stored. Padded paths are zeros: use the mask in
  every loss and metric so they never count as weak paths.

The saved `ids` (scene, carrier, BS, UE) tie each row back to its record.

```python
paths = data["train"]["mask"].sum(axis=1)
print("stored paths per link: min", paths.min(), "mean", round(float(paths.mean()), 1), "max", paths.max())

def masked_mse(prediction, target, mask):
    error = ((prediction - target) ** 2).mean(dim=-1)          # (batch, 16)
    return (error * mask).sum() / mask.sum().clamp(min=1)

# Predicting the training mean (zero after normalization) gives a masked MSE near 1.
print("masked MSE of predicting the mean:",
      round(float(masked_mse(torch.zeros_like(batch["y"]), batch["y"], batch["mask"])), 3))
```

## Adapt it

- **Other inputs:** add the carrier, the scene geometry (`scene.open_geometry()`), or the
  channel at the other carrier for cross-band tasks; keep them per link.
- **Other targets:** use the complex coefficients, departure angles, the frequency response
  (`T.frequency_response`) or the interaction detail (`components=("core", "detail")`,
  `T.pad_interactions`).
- **More data:** widen the selection (`tier="small"` on the release catalog, other profiles or
  BSs) and raise `LINKS_PER_RECORD`; the [library guide](guide.md) plans TAR
  downloads, and `c4w.to_cache` keeps records memory-mapped for repeated reads.
- **Shuffling at scale:** records are stored in groups of 25 scenes; shuffle groups or scenes
  and read each one's records together, as above.
- **Records instead of links:** `channels4world.torch.RecordDataset(selection, transform=...)`
  yields one record per item and reopens the dataset in each worker process.

*Notebook: [02-dataloader.ipynb](../examples/02-dataloader.ipynb). The page and the notebook hold the same code.*

[Library documentation](../README.md)
