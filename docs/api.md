# API and CLI

Reference for the channels4world library. Examples assume the named data is local.
The dataset card holds the full field and schema tables.

## Open and navigate

```python
import channels4world as c4w

ds = c4w.open(path)
```

`path` is a local release, Tiny, extracted subset or NPY-cache root. Opening reads
the manifest and referenced root tables. Group metadata is loaded and checked once,
lazily. Payload reads are explicit. The library does not perform network transfers.

| Object | Public access |
|---|---|
| `Dataset` | `path`, `kind`, `release`, `tiers`, `profiles`, `frequencies_hz`, `splits`, `scenes`, `scene(scene_id)`, `select(...)`, `select_recipe(recipe)` |
| `Split` | `scenes`, `select(...)` |
| `Scene` | `id`, `split`, `profile`, `tier`, `frequencies`, `read_positions`, `open_geometry`, `extract_geometry`, `records` |
| `Frequency` | `base_stations` |
| `BaseStation` | `id`, `frequency_hz`, `scene`, `read` |

Collections look up by ID (`ds.splits["train"]`), iterate over objects and expose
`.ids`. Scene/split IDs are strings, frequency keys are integer Hz and BS keys are
integers. `ds.scene(id)` works across splits. `kind` is `release`, `subset` (including
Tiny) or `cache`; the scene tier is independent of root kind and local availability.

## Select and inspect availability

```python
selection = ds.select(
    split=None, tier=None, profile=None, scene_ids=None, frequencies_hz=None, bs_ids=None,
    components=("core", "positions", "geometry"),
)
```

Filters are keyword arguments. `profile` takes one name or a list, spelled as in the
catalog (`ds.profiles` lists them; for example `metropolitan`, while scene IDs use
`metro_`). Components are `core`, `detail`, `positions` and
`geometry`; requesting detail also requests core. Tiers select nested scene
populations and preserve the chosen component request.

| Selection method/property | Meaning |
|---|---|
| `scene_ids` | Selected IDs in pinned order |
| `scenes()` | Selected scene objects; `records()` on each preserves the selection filters |
| `records(verify="basic")` | Read selected channel records, scene order then ascending carrier/BS |
| `missing()` | Downloadable missing files plus requested items not published/included |
| `local()` | Explicitly narrow to locally usable items |
| `summary()` | Counts and published bytes of what would be read or extracted |
| `to_recipe(revision=None)` | The selection as a shareable recipe (below) |

`missing()` has a catalog-only planning route when group metadata/schemas are absent.
It reports matching payload files and required metadata, then becomes more precise
once metadata is local. TARs are download units; a loose root can initially plan
other files in a touched group. Rerun after downloading. Iteration, extraction,
cache conversion and local filtering use the required group metadata/schemas.
Positions/geometry-only selections do not require channel core payloads.

`local()` keeps channel records only when every requested channel component is
usable; positions and geometry are independent. It can omit scenes. Ordinary
selection does not silently filter unavailable detail.

Requests for items the opened root does not host are planned as unpublished, never as
errors. In release 1.0.0 that includes detail for scenes outside the Medium tier: their
records are catalogued in the group metadata, but their archives are not part of the
release, so `missing()`, `summary()`, preparation plans and CLI `verify` list them as
unpublished. Reading one directly raises `NotPublishedError`.

### Selection recipes

A recipe is a small JSON file that records a selection and the release it refers to,
so it can be shared and replayed. It holds no paths or commands: the recipient plans
downloads from their own catalog, so one recipe works for the release root and for Tiny.

```python
recipe = selection.to_recipe(revision="v1.0.0")   # dict; c4w.recipe.dumps(recipe) -> JSON text
selection = ds.select_recipe("selection.json")    # a path or a parsed dict
```

```json
{
  "format": "channels4world.selection",
  "format_version": 1,
  "release": "1.0.0",
  "revision": "v1.0.0",
  "selection": {"tier": "small", "profiles": ["urban"], "components": ["core", "positions"]}
}
```

`selection` may contain `tier`, `split`, `profiles`, `scene_ids`, `frequencies_hz`,
`bs_ids` and `components` (required), with the same meanings as `select()`; `revision`
is optional. `select_recipe` refuses a recipe for another release and checks every
filter against the catalog as `select()` does. Unknown fields, wrong types, duplicate
keys and NaN are refused (`RecipeError`, a `ValueError`); nothing in a recipe is
executed. `c4w.recipe` also offers `load(path_or_dict)`, `loads(text)`, `validate(obj)`
and `dumps(recipe)`. A `local()` selection cannot be exported: it depends on what is
downloaded on one machine.

### MissingReport

| Member | Meaning |
|---|---|
| `files` | Missing paths relative to the opened root |
| `bytes` | Sum of the reported missing file sizes |
| `unpublished` | Requested items absent from the release/subset: unpublished components, scenes with no channel records when core is requested, and explicitly requested carriers or BS IDs a selected scene does not have. `summary()`, preparation plans and CLI `verify` count the same items |
| `complete` | True only when both `files` and `unpublished` are empty |
| `bool(report)` | True when `files` is nonempty; does not include unpublished requests |
| `repo_files` | Missing paths with the published HF root prefix applied |
| `local_dir` | Local folder corresponding to the HF repository top level |
| `hf_command(revision=None)` | Quoted POSIX `hf download` command; requires a nonempty file plan |

HF helpers require publication metadata and a matching download layout. They refuse
user-created subsets/caches or unpublished roots. Pass the immutable dataset
revision used for the catalog. Reports check presence; reading verifies contents.

## Read arrays, links and geometry

| Call | Result |
|---|---|
| `bs.read(detail=False, verify="basic")` | `Record` with core; optionally detail for the same scene/carrier/BS |
| `scene.records(detail=False, verify="basic")` | Records in ascending carrier/BS order |
| `selected_scene.records(detail=None, verify="basic")` | Records under selection filters; `None` uses its detail request |
| `scene.read_positions(verify="basic")` | `Positions.bs` and `.ue` arrays, metres |
| `scene.open_geometry(verify="basic")` | In-memory `zipfile.ZipFile`; close it or use `with` |
| `scene.extract_geometry(dest, verify="basic")` | Write geometry under `dest/<scene_id>/`; return its XML path |

`Record` exposes `id`, `scene_id`, `frequency_hz`, `bs_id`, `header`, `n_ues`,
`n_links`, `n_paths`, `ue_ids`, `core` and optional `detail`. Arrays are read-only
and retain stored values/dtypes. `get_link(ue_id)` returns a `Link` for a retained
link, `None` for a valid UE without one, or raises for an invalid UE ID. `links()`
iterates retained links. `copy()` returns dictionaries of writable array copies.

A `Link` exposes `ue_id`, `n_paths`, `core` and optional `detail`; interaction offsets
are rebased to that link. Use the UE ID to index positions; retained-link row indices
are different. Consult the dataset schema for field meanings and shapes.

## Extract and cache

```python
c4w.extract(selection, dest, verify="basic", check_space=True,
            skip_existing=False, confirm=None)
c4w.to_cache(selection, dest, verify="basic", check_space=True, confirm=None)
```

Both return the destination `Path`, which opens with `c4w.open`. Extraction copies
published NPZ/ZIP records into a loose subset and accepts a release/loose source,
not an NPY cache. Cache conversion stores memory-mappable numeric arrays and copies
geometry ZIPs. Original group metadata and included-record selection accompany
subsets/caches. Stored values, precision and ragged associations are preserved.

Extraction refuses an existing destination unless `skip_existing=True` is explicit.
Matching existing payloads are then kept according to the verify level. With
`skip_existing=True` the selection must include every record the destination already
holds: it resumes or extends an earlier extraction (for example, select both carriers to
add the second one) and never drops earlier contents. A narrower selection is refused.
A subset or cache lists only the carriers it actually contains.
Cache conversion requires a new destination. Unpublished requested items are counted
in the preparation plan and omitted from output: inspect `missing().complete` when
complete coverage is required.

`confirm`, when supplied, receives a `Plan` and returns whether to proceed. Python
has no default prompt. `Plan` exposes `action`, `dest`, `scenes`, `records`, `files`,
`bytes`, `free_bytes`, `unpublished`, `skipped_existing` and `describe()`. Estimates
exclude filesystem overhead. The CLI prints the estimate and asks before writing;
`--yes` skips that prompt. `check_space` is independent of verification. Root JSON
is written last.

## Verification

All payload reads/preparation accept `none`, `basic` or `full`; default `basic`.
Group metadata is checked against the root when first loaded at every level.

| Level | Published NPZ/ZIP | NPY cache |
|---|---|---|
| `none` | Decode payloads without payload validation | No payload checks |
| `basic` | Exact published bytes, declared arrays, dtypes and shapes | Dtypes and shapes |
| `full` | Basic plus scientific/release-rule checks and published content-proof agreement | Array agreement with the published content proof plus scientific checks |

Full checks apply to components loaded in that call; unrelated components are not
loaded for validation. Geometry has no additional scientific checks, so its full
level equals basic. A full core+detail read checks their association together:
`bs.read(detail=True, verify="full")` or CLI verification of both components.
Extraction/cache conversion checks individual records and does not repeat that
pairwise association check.

## Optional transforms and Torch

Import transforms with `from channels4world import transforms as T`:

| Function | Output |
|---|---|
| `complex_gain(record)` | `(paths,)`, complex64 |
| `path_power_db(record)` | `(paths,)`, float64, dB relative to unit power |
| `delay_spread(record)` | `(links,)`, float64 seconds; NaN for zero total retained power |
| `baseband_coefficients(record, delay_reference="absolute")` | `(paths,)`, complex128 |
| `frequency_response(record, offsets_hz, delay_reference="absolute", max_elements=1 << 22)` | `(links, len(offsets_hz))`, complex128 |
| `pad_paths(core, name_or_array, max_paths=16)` | `(links, 16, ...)` values and `(links, 16)` bool mask |
| `pad_interactions(detail, name_or_array, max_interactions=4)` | `(paths, 4, ...)` values and `(paths, 4)` bool mask |

These pure transforms return new writable arrays. `first_retained` is the other
delay-reference option. Frequency inputs are offsets from the record's carrier.
See [conventions](conventions.md) for the scientific meaning and precision limits.

`pad_paths` and `pad_interactions` turn the stored offset-based layout into fixed-size
rows. They take any mapping of arrays (`record.core`, `record.detail`, `link.detail`, an
opened NPZ or a restored HF Datasets row), keep the stored dtype, fill padding with zeros
and return a mask that tells stored entries from padding. Row *i* of `pad_paths` is UE
`core["link_ue_index"][i]`. Pass the output of `pad_interactions` to `pad_paths` for
`(links, 16, 4, ...)`. They never truncate: a row longer than the requested width raises
`ValueError`. `T.MAX_PATHS` (16) and `T.MAX_INTERACTIONS` (4) are the release limits.

`channels4world.torch.RecordDataset(selection, transform=None, verify="basic")`
is a map-style adapter exposing `len(data)` and `data[index]`. It builds a metadata
record list, then reads lazily. PyTorch is optional. Use single-record DataLoader
items, list collation for ragged records or an explicit transform. Copy read-only
stored arrays before `torch.from_numpy`. See the [guide](guide.md#pytorch).

## Command line

```text
c4w --version
c4w info     PATH
c4w scenes   PATH [selection]
c4w missing  PATH [selection] [--paths-only] [--revision REV]
c4w recipe   PATH [selection] [--revision REV] [--output FILE]
c4w extract  PATH DEST [selection] [--verify LEVEL] [--skip-existing] [--yes] [--no-space-check]
c4w cache    PATH DEST [selection] [--verify LEVEL] [--yes] [--no-space-check]
c4w verify   PATH [selection] [--level basic|full]
c4w geometry PATH SCENE_ID DEST
```

Selection flags: `--split`, `--tier`, repeatable `--profile NAME`, `--scenes ID,ID` or
`--scenes @FILE`, repeatable `--frequency HZ` and `--bs N`, and comma-separated
`--components`, or `--recipe FILE` in place of all of them. Frequencies are integer Hz.
`recipe` writes the selection as a recipe; `--output` never overwrites a file. `missing` exits zero only when everything
requested is published and present; its download hint uses `--revision REV`, the
recipe's revision, or a `REVISION` placeholder to replace. `verify` exits non-zero for integrity failures or
files that are not downloaded; requested items that are not published are listed but
are not failures. CLI `verify` defaults to `full`; preparation defaults to `basic`.
`c4w --version` prints the library version, which Python reads as `c4w.__version__`; it is
separate from the root format and the dataset release.

## Errors

All library-specific errors derive from `C4WError`:

| Error | Meaning |
|---|---|
| `UnknownIDError` | Unknown split, profile, tier, scene, frequency, BS or invalid UE ID |
| `NotPublishedError` | Requested component absent from the release/subset |
| `NotDownloadedError` | Published file absent locally; `.paths` lists missing paths |
| `IntegrityError` | Published bytes/array checks fail |
| `FormatError` | Root/format is unreadable, incomplete or unsupported |
| `InsufficientSpaceError` | Preparation estimate exceeds available destination space |

`RecipeError` (an invalid selection recipe) derives from `ValueError`.

[README](../README.md) · [Guide](guide.md) · [Development](development.md)
