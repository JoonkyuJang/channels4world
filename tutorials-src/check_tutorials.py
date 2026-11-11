#!/usr/bin/env python3
"""Execute the library notebooks and optional dataset NumPy notebook against a synthetic release in the published layout.

The release is small but has the real repository layout: a release root at the top
level with group TARs under payload/, and Tiny as loose files under tiny/ with its own
catalog. It holds the tutorials' scene and group names, and UE 14 has a link to BS 0 at
both carriers. ``snapshot_download`` is replaced by a copy from that synthetic repository
that applies ``allow_patterns`` the way huggingface_hub does (fnmatch on repository
paths), so the notebooks' download patterns are checked too. Nothing touches the network.

Values printed by a run are synthetic. Never publish executed outputs from this check;
run the notebooks on the Hub release to make published outputs.

Usage: python3 tutorials-src/check_tutorials.py [LIBRARY_DIR] [HF_CARD_DIR] [--keep DIR]
(LIBRARY_DIR is the library checkout: its src/ and tests/synth.py are used.)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

REPO, REVISION = "Intelligence-Networking-Lab/Channels4World", "v1.0.0"
LEAD, LEAD_GROUP, UE_ID = "urban_scene_019234", "test-urban-g00000", 14
NOTEBOOKS = ("01-read-one-scene-library.ipynb", "02-read-one-scene-numpy.ipynb", "03-dataloader.ipynb")

# Group name, split, profile, scenes. Every scene is in Tiny except those of the last group,
# which is Small only, so tier and profile filters have something to exclude.
GROUPS = [
    ("test-urban-g00000", "test", "urban", [LEAD, "urban_scene_019235"]),
    ("train-urban-g00000", "train", "urban", [f"urban_scene_{i:06d}" for i in range(20)]),
    ("test-metro-g00000", "test", "metropolitan", ["metro_scene_054046"]),
    ("train-suburban-g00000", "train", "suburban", ["suburban_scene_000100"]),
    ("train-urban-g00001", "train", "urban", ["urban_scene_000500"]),
]
SMALL_ONLY = {"train-urban-g00001"}

BUILDING_OBJ = """# box
v 10 10 0
v 30 10 0
v 30 40 0
v 10 40 0
v 10 10 25
v 30 10 25
v 30 40 25
v 10 40 25
vn 0 0 1
f 5//1 6//1 7//1 8//1
f 1/1 2/1 6/1
f 1 6 5
"""


def build_repo(lib: Path, repo: Path) -> None:
    sys.path[:0] = [str(lib / "src"), str(lib / "tests")]
    import numpy as np
    import synth
    import channels4world as c4w

    synth.N_BS, synth.N_UES = 4, 64
    synth.GROUPS = [{"name": n, "split": s, "profile": p, "scenes": ids, "detail": True,
                     "inc": "medium" if n in SMALL_ONLY else "small"} for n, s, p, ids in GROUPS]
    synth.REALISTIC = {n: n for n, *_ in GROUPS}
    synth.TIERS = {sid: ("small" if n in SMALL_ONLY else "tiny") for n, _, _, ids in GROUPS for sid in ids}

    original = synth.core_and_detail

    def with_ue(rng, **kw):  # every record links UE 14, as the lead link does at both carriers
        while True:
            core, detail = original(rng, **kw)
            if UE_ID in core["link_ue_index"]:
                return core, detail
    synth.core_and_detail = with_ue

    class ZipWithBuilding(synth.zipfile.ZipFile):  # synthetic geometry gains one building mesh
        def writestr(self, name, data, *a, **kw):
            super().writestr(name, data, *a, **kw)
            if name == "meshes/ground.obj":
                super().writestr("meshes/building_0.obj", BUILDING_OBJ)

    class _Zip:
        ZipFile, ZIP_DEFLATED = ZipWithBuilding, synth.zipfile.ZIP_DEFLATED
    synth.zipfile = _Zip

    synth.make_release(repo, realistic_names=True, layout="release", hf_repo=REPO, hf_path="")
    tiny = c4w.extract(c4w.open(repo).select(tier="tiny", components=("core", "detail", "positions", "geometry")),
                       repo / "tiny", verify="full")
    synth.set_root_publication(tiny, hf_repo=REPO, hf_path="tiny")
    assert c4w.open(repo / "tiny").scene(LEAD).tier == "tiny"
    del np


FAKE_HUB = '''
import fnmatch as _fn, shutil as _sh
from pathlib import Path as _P
import huggingface_hub as _hub
_SOURCE = _P({source!r})
_FILES = sorted(p.relative_to(_SOURCE).as_posix() for p in _SOURCE.rglob("*") if p.is_file())
COPIED = []
def _snapshot_download(repo_id, *, repo_type=None, revision=None, local_dir=None, allow_patterns=None, **kw):
    assert (repo_id, repo_type, revision) == ({repo!r}, "dataset", {revision!r}), (repo_id, repo_type, revision)
    assert local_dir is not None and allow_patterns, "tutorials download explicit patterns into a folder"
    patterns = [allow_patterns] if isinstance(allow_patterns, str) else list(allow_patterns)
    patterns = [p + "*" if p.endswith("/") else p for p in patterns]
    for rel in _FILES:
        if any(_fn.fnmatch(rel, p) for p in patterns):
            out = _P(local_dir) / rel
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                _sh.copyfile(_SOURCE / rel, out)
                COPIED.append(rel)
    return str(local_dir)
_hub.snapshot_download = _snapshot_download
'''


ML_INVARIANTS = '''
import numpy as _np
for _part, _d in data.items():
    assert _np.all(_d["y"][~_d["mask"]] == 0), f"{_part}: padding must stay zero"
    assert _np.isfinite(_d["x"]).all() and _np.isfinite(_d["y"]).all(), _part
    assert set(_d["ids"][:, 0]) <= set(splits[_part]), f"{_part}: rows from other scenes"
_sets = [set(v) for v in splits.values()]
assert all(not (a & b) for i, a in enumerate(_sets) for b in _sets[i + 1:]), "a scene in two parts"
_tr = data["train"]
assert _np.allclose(_tr["x"].mean(axis=0), 0, atol=1e-3), "inputs normalized with training statistics"
assert _np.allclose(_tr["y"][_tr["mask"]].mean(axis=0), 0, atol=1e-3), "targets: stored paths only"
print("INVARIANTS OK", key)
'''


def execute(path: Path, workdir: Path, repo: Path, lib: Path, extra: tuple[str, ...] = (),
            replace: tuple[str, str] | None = None) -> nbformat.NotebookNode:
    nb = nbformat.read(path, as_version=4)
    if replace:
        hits = [c for c in nb.cells if c.cell_type == "code" and replace[0] in c.source]
        assert len(hits) == 1, f"{replace[0]!r} must occur in exactly one cell"
        hits[0].source = hits[0].source.replace(*replace)
    for i, code in enumerate(extra):
        cell = nbformat.v4.new_code_cell(code)
        cell.id = f"check-extra-{i}"
        nb.cells.append(cell)
    setup = nbformat.v4.new_code_cell(FAKE_HUB.format(source=str(repo), repo=REPO, revision=REVISION))
    setup.id = "check-setup"
    nb.cells.insert(0, setup)
    nb.cells.append(nbformat.v4.new_code_cell("import json as _j; print('COPIED=' + _j.dumps(COPIED))"))
    nb.cells[-1].id = "check-copied"
    old = os.environ.copy()
    os.environ["PYTHONPATH"] = str(lib / "src")
    os.environ.pop("MPLBACKEND", None)  # keep the inline backend so figures become outputs
    try:
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        try:
            NotebookClient(nb, timeout=600, km=km,
                           resources={"metadata": {"path": str(workdir)}}).execute()
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return nb


def text_of(nb: nbformat.NotebookNode) -> str:
    out = []
    for cell in nb.cells:
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                out.append(o["text"])
            elif "data" in o and "text/plain" in o["data"]:
                out.append(o["data"]["text/plain"])
    return "\n".join(out)


def copied(nb: nbformat.NotebookNode) -> list[str]:
    line = [l for l in text_of(nb).splitlines() if l.startswith("COPIED=")][-1]
    return json.loads(line[len("COPIED="):])


def images(nb: nbformat.NotebookNode) -> int:
    return sum("image/png" in o.get("data", {}) for c in nb.cells for o in c.get("outputs", []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("library", type=Path, nargs="?", default=Path(__file__).resolve().parent.parent)
    ap.add_argument("card", type=Path, nargs="?", help="include the dataset card's NumPy notebook")
    ap.add_argument("--keep", type=Path, help="write executed notebooks here (synthetic outputs; never publish)")
    a = ap.parse_args()
    paths = {
        NOTEBOOKS[0]: a.library.resolve() / "examples/01-read-one-scene-library.ipynb",
        NOTEBOOKS[2]: a.library.resolve() / "examples/02-dataloader.ipynb",
    }
    if a.card:
        paths[NOTEBOOKS[1]] = a.card.resolve() / "docs/notebooks" / NOTEBOOKS[1]
    failures = []
    with tempfile.TemporaryDirectory(prefix="c4w-tutorials-") as tmp:
        tmp = Path(tmp)
        repo = tmp / "repo"
        build_repo(a.library.resolve(), repo)
        runs = {}
        for name in paths:
            work = tmp / name.split(".")[0]
            work.mkdir()
            try:
                nb = execute(paths[name], work, repo, a.library.resolve(),
                         extra=(ML_INVARIANTS,) if name == NOTEBOOKS[2] else ())
            except Exception as exc:  # nbclient raises CellExecutionError with the traceback
                failures.append(f"{name}: {exc}")
                continue
            runs[name] = (nb, work)
            if a.keep:
                a.keep.mkdir(parents=True, exist_ok=True)
                nbformat.write(nb, a.keep / name)

        shared = tmp / "shared"  # one folder, another order: each notebook must cope with the others' files
        shared.mkdir()
        for name in (NOTEBOOKS[1], NOTEBOOKS[0], NOTEBOOKS[2]):
            if name not in paths:
                continue
            try:
                execute(paths[name], shared, repo, a.library.resolve())
                print("ok   " + name + " also runs after the others in a shared folder")
            except Exception as exc:
                failures.append(f"{name} in a shared folder: {exc}")

        def check(cond: bool, label: str) -> None:
            print(("ok   " if cond else "FAIL ") + label)
            if not cond:
                failures.append(label)

        if NOTEBOOKS[0] in runs:
            nb, _ = runs[NOTEBOOKS[0]]
            t, got = text_of(nb), copied(nb)
            lead = f"tiny/records/scene={LEAD}/"
            records = sorted(p for p in got if p.startswith(lead))
            check(records == sorted([lead + "geometry.zip", lead + "positions.npz"]
                                    + [f"{lead}band_hz={hz}/component={c}/bs00000.npz"
                                       for hz in (3_500_000_000, 7_000_000_000) for c in ("core", "detail")]),
                  "library tutorial downloads exactly the lead scene's positions, geometry and BS 0 records")
            check(not any(p.startswith("tiny/records/") and not p.startswith(lead) for p in got),
                  "library tutorial downloads no other scene")
            check(f"tiny/groups/{LEAD_GROUP}/snapshot.json" in got and
                  not any(p.startswith("tiny/groups/") and f"/{LEAD_GROUP}/" not in p for p in got),
                  "library tutorial downloads only the lead group's metadata")
            check(f"Link(UE {UE_ID}," in t, "library tutorial reads the lead link")
            check("1 building meshes" in t and images(nb) == 3, "library tutorial draws the scene and both profiles")
            check("link power" in t and "7 GHz" in t, "library tutorial compares the carriers")
        if NOTEBOOKS[1] in runs:
            nb, _ = runs[NOTEBOOKS[1]]
            t, got = text_of(nb), copied(nb)
            check(all(p.startswith(f"tiny/records/scene={LEAD}/") for p in got) and len(got) == 6,
                  "NumPy tutorial downloads only the lead scene's six record files")
            check(f"UE {UE_ID}: link row" in t and images(nb) == 3, "NumPy tutorial finds the link and plots")
        if NOTEBOOKS[0] in runs and NOTEBOOKS[1] in runs:
            lib_table = [l for l in text_of(runs[NOTEBOOKS[0]][0]).splitlines() if l.startswith("rank ")]
            np_table = [l for l in text_of(runs[NOTEBOOKS[1]][0]).splitlines() if l.startswith("rank ")]
            check(bool(lib_table) and lib_table == np_table, "library and NumPy tutorials print the same path table")
        if NOTEBOOKS[2] in runs:
            nb, work = runs[NOTEBOOKS[2]]
            t, got = text_of(nb), copied(nb)
            urban_tiny = [s for n, _, p, ids in GROUPS if p == "urban" and n not in SMALL_ONLY for s in ids]
            want = {f"tiny/records/scene={s}/positions.npz" for s in urban_tiny} | {
                f"tiny/records/scene={s}/band_hz=3500000000/component=core/bs00000.npz" for s in urban_tiny}
            check({p for p in got if p.startswith("tiny/records/")} == want,
                  "ML guide downloads BS 0 core at 3.5 GHz and positions of exactly the urban Tiny scenes")
            check(not any("-metro-" in p or "-suburban-" in p for p in got), "ML guide downloads no other profile")
            splits = json.loads((work / "splits.json").read_text())
            check(sorted(splits["test"]) == sorted(GROUPS[0][3]) and splits["validation"]
                  and not set(splits["validation"]) & set(splits["train"]),
                  "ML guide splits by scene, keeps the test split and holds out validation scenes")
            shapes = re.findall(r"^(x|y|mask)\s+\((\d+)((?:, \d+)*)\) torch\.(\w+)$", t, re.M)
            check([(n, rest, dt) for n, _, rest, dt in shapes] == [("x", ", 4", "float32"), ("y", ", 16, 5", "float32"),
                                                                 ("mask", ", 16", "bool")]
                  and len({b for _, b, _, _ in shapes}) == 1, "ML guide batches fixed-size tensors")
            check(len(list((work / "cache").glob("*/train.npz"))) == 1, "ML guide caches once under one key")
            check("INVARIANTS OK" in t, "ML guide: padding stays zero, no scene in two parts, training-only statistics")
            # A different validation assignment must build a new cache, never reuse the old arrays.
            try:
                nb2 = execute(paths[NOTEBOOKS[2]], work, repo, a.library.resolve(), extra=(ML_INVARIANTS,),
                              replace=("fraction=0.1", "fraction=0.3"))
                splits2 = json.loads((work / "splits.json").read_text())
                ok = ("INVARIANTS OK" in text_of(nb2) and splits2["validation"] != splits["validation"]
                      and len(list((work / "cache").glob("*/train.npz"))) == 2)
            except Exception as exc:  # noqa: BLE001
                ok = False
                failures.append(f"ML guide rerun with another split: {exc}")
            check(ok, "ML guide: another validation assignment builds a new cache under a new key")
    for f in failures:
        print("FAIL", f.splitlines()[0][:400] if "\n" not in f[:400] else f[:2000])
    print("all notebook checks passed" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
