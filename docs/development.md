# Development

The channels4world library is maintained by **Joonkyu Jang**. Source:
`https://github.com/JoonkyuJang/channels4world`; package: `https://pypi.org/project/channels4world/`.

## Source layout

```text
src/channels4world/
  __init__.py     public entry points and errors
  dataset.py      navigation, selection, records, links, positions, missing reports
  recipe.py       selection recipes (JSON): validation, reading and writing
  _root.py        root tables, lazy group metadata, TAR/loose/NPY access
  _contract.py    record contract: headers, arrays, content checks
  _paths.py       safe relative paths and exclusive, link-refusing writes
  _errors.py      public exception classes
  prepare.py      extract and to_cache
  transforms.py   optional scientific transforms and fixed-size padding
  torch.py        thin optional PyTorch adapter; the module does not import torch
  cli.py          the c4w command
```

Dependency direction is `dataset` → `_root` → `_contract`; `prepare` uses `dataset`.
Keep transforms and the PyTorch adapter independent of ordinary reads. Build output,
package metadata and caches stay out of the source tree.

## Local development

From a source checkout:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

Building needs setuptools **77+** for the SPDX license string. Requirements are Python
**3.11+**, NumPy **1.26+**, PyArrow **14+** and, for the optional extra, PyTorch **2.1+**.
The tests run on Python 3.11 with the minimum NumPy and PyArrow and on Python 3.12, 3.13
and 3.14 with current releases; the PyTorch tests are skipped when PyTorch is not installed.

`pytest` builds a small synthetic release in the published format (`tests/synth.py`):
TARs of NPZ and ZIP members, group catalogs and snapshots, the public schema documents and
a release root. Keep fixtures with the tests.

## Building the package

```bash
python -m pip install build
python -m build                       # dist/: wheel and sdist
python -m venv /tmp/c4w-clean && /tmp/c4w-clean/bin/pip install dist/*.whl pytest
/tmp/c4w-clean/bin/c4w --version
tar -xzf dist/*.tar.gz -C /tmp && /tmp/c4w-clean/bin/python -m pytest /tmp/channels4world-*/tests
```

The sdist carries the tests (with their synthetic-release helper and fixtures) and these
docs, so the last line checks the installed wheel without the source tree on the path.

## Root format and versions

The library version, the root format and the dataset release are separate identifiers.
The root format is `channels4world.root` version 1.

A root is `c4w-root.json` plus the scenes and files tables it references, and, for
subsets and caches, the IDs of the records they include. The root JSON is written last.
Group metadata loads and is checked lazily. Extracted subsets keep the original group
metadata and payload bytes; NPY caches keep values, dtypes and the offset layout. See
the [API](api.md) and the dataset card's schema reference for the full contract.

Published roots also record where they sit in a Hugging Face repository (`hf_repo`,
`hf_path`): the release at the repository top level and Tiny under `tiny/`. These fields
let `MissingReport` map missing files to repository paths; subsets and caches you make
yourself leave them empty, so they never advertise their files as repository downloads. A
root may leave out archives that its groups catalogue (release 1.0.0 leaves out detail
outside the Medium tier); the library then plans those records as unpublished. Editing a
root file changes its hash, so update every pin that refers to it. The library reads
published roots; it does not build them.

## Licenses

The software is Apache-2.0; the full text is in [LICENSE](../LICENSE) and the notice in
[COPYRIGHT](../COPYRIGHT). Joonkyu Jang is the copyright holder and maintainer. The
Channels4World dataset is separately CC BY 4.0; the software notice does not cover it.

[README](../README.md) · [Guide](guide.md) · [API](api.md) · [Conventions](conventions.md)
