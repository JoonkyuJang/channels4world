from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

import channels4world as c4w
from channels4world import _contract as C
from channels4world import prepare
from channels4world import transforms as T
from channels4world.cli import main as cli
from channels4world.torch import RecordDataset
from synth import FREQS, N_UES, make_release

F35, F7 = FREQS
ALL = ("core", "detail", "positions", "geometry")


@pytest.fixture(scope="session")
def release(tmp_path_factory) -> Path:
    return make_release(tmp_path_factory.mktemp("release") / "c4w")


@pytest.fixture(scope="session")
def bad_release(tmp_path_factory) -> Path:
    return make_release(tmp_path_factory.mktemp("bad") / "c4w", seed=1, bad_core=("a0", F35, 2),
                        zero_link=("a1", F7, 0))


@pytest.fixture
def copy(tmp_path, release) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(release, dst)
    return dst


# ----------------------------------------------------------------------------- navigation
def test_open_navigation_and_lookup(release):
    ds = c4w.open(release)
    assert ds._root._groups == {}  # open reads only the root
    assert ds.splits.ids == ["train", "test"]
    assert [s.id for s in ds.splits["train"].scenes] == ["a0", "a1", "b0", "b1"]  # pinned order, yields objects
    scene = ds.scene("b0")
    assert ds._root._groups == {}  # scene lookup needs no group metadata
    assert (scene.split, scene.profile, scene.tier) == ("train", "suburban", "medium")
    assert ds.splits["train"].scenes["a0"].id == "a0"
    f = ds.scene("a0").frequencies
    assert f.ids == [F35, F7]
    assert f[F35].base_stations.ids == [0, 1, 2]
    with pytest.raises(TypeError, match="integer Hz"):
        f[3.5e9]
    with pytest.raises(c4w.UnknownIDError):
        f[F35].base_stations[9]
    with pytest.raises(c4w.UnknownIDError):
        ds.scene("nope")
    assert 3.5e9 not in f and F35 in f
    assert "tiny (1)" in repr(ds)


def test_tiers_are_nested(release):
    ds = c4w.open(release)
    assert ds.select(tier="tiny").scene_ids == ["a0"]
    assert ds.select(tier="small").scene_ids == ["a0", "a1", "c0"]
    assert ds.select(tier="medium").scene_ids == ["a0", "a1", "b0", "c0"]
    assert ds.select(tier="full").scene_ids == ["a0", "a1", "b0", "b1", "c0"]
    assert ds.select(split="train", tier="small").scene_ids == ["a0", "a1"]
    with pytest.raises(TypeError):
        ds.select(frequencies_hz=[3.5e9])
    with pytest.raises(ValueError):
        ds.select(components=("core", "pressure"))


# ----------------------------------------------------------------------------- reading
def test_read_record_link_and_positions(release):
    ds = c4w.open(release)
    scene = ds.scene("a0")
    rec = scene.frequencies[F35].base_stations[1].read(detail=True)
    assert (rec.scene_id, rec.frequency_hz, rec.bs_id) == ("a0", F35, 1)
    assert rec.core["tau"].dtype == np.float32 and not rec.core["tau"].flags.writeable
    with pytest.raises(ValueError):
        rec.core["tau"][0] = 1.0
    assert np.all(np.diff(rec.ue_ids) > 0)
    ue = int(rec.ue_ids[1])
    link = rec.get_link(ue)
    off = rec.core["link_path_offsets"]
    assert link.ue_id == ue and link.n_paths == off[2] - off[1]
    np.testing.assert_array_equal(link.core["tau"], rec.core["tau"][off[1]:off[2]])
    assert link.detail["path_interaction_offsets"][0] == 0
    absent = next(u for u in range(N_UES) if u not in set(rec.ue_ids.tolist()))
    assert rec.get_link(absent) is None
    with pytest.raises(c4w.UnknownIDError):
        rec.get_link(N_UES)
    with pytest.raises(TypeError):
        rec.get_link(1.0)
    writable = rec.copy()
    writable["core"]["tau"][0] = 0
    pos = scene.read_positions()
    assert pos.bs.shape == (3, 3) and pos.ue[rec.ue_ids].shape == (rec.n_links, 3)


def test_geometry_access(tmp_path, release):
    scene = c4w.open(release).scene("a1")
    with scene.open_geometry() as z:
        assert "scene.xml" in z.namelist()
    xml = scene.extract_geometry(tmp_path / "geom")
    assert xml == tmp_path / "geom" / "a1" / "scene.xml" and (xml.parent / "meshes/ground.obj").exists()


# ----------------------------------------------------------------------------- verification
def _corrupt_member(root: Path, scene: str, hz: int, comp: str, bs: int) -> None:
    ds = c4w.open(root)
    g = ds.scene(scene)._group
    rid = g.records_of_scene(scene)[(comp, hz, bs)]
    m = g.members[rid]
    with (root / m["object_path"]).open("r+b") as fh:
        fh.seek(m["data_offset"] + m["member_bytes"] // 2)
        b = fh.read(1)
        fh.seek(-1, 1)
        fh.write(bytes([b[0] ^ 0xFF]))


def test_verify_levels(copy, bad_release):
    _corrupt_member(copy, "a0", F35, "core", 0)
    bs = c4w.open(copy).scene("a0").frequencies[F35].base_stations[0]
    with pytest.raises(c4w.IntegrityError, match="published member"):
        bs.read()
    ds = c4w.open(bad_release)
    bad = ds.scene("a0").frequencies[F35].base_stations[2]
    bad.read(verify="basic")  # bytes are as published; release rules are only checked by "full"
    with pytest.raises(c4w.IntegrityError, match="retained-empty"):
        bad.read(verify="full")
    zero = ds.scene("a1").frequencies[F7].base_stations[0].read(verify="full")  # zero-link record is valid
    assert zero.n_links == 0 and zero.n_paths == 0
    with pytest.raises(ValueError):
        bad.read(verify="paranoid")


def test_group_metadata_is_checked_against_root(copy):
    p = copy / "groups/g-a/snapshot.json"
    p.write_bytes(p.read_bytes().replace(b'"version":1', b'"version": 1', 1))
    ds = c4w.open(copy)
    with pytest.raises(c4w.IntegrityError, match="differs from the release root"):
        ds.scene("a0").frequencies  # noqa: B018


# ----------------------------------------------------------------------------- snapshot reuse
def _original_select(payload: bytes, rid: str) -> dict:
    matches = [r for r in C.strict_json(payload)["records"] if r["record_id"] == rid]
    if len(matches) != 1:
        raise KeyError(rid)
    return matches[0]


def test_snapshot_index_matches_original_and_is_immutable(release, monkeypatch):
    body = (release / "groups/g-a/snapshot.json").read_bytes()
    snap = C.Snapshot(body, C.sha256(body), release)
    for rid, *_ in snap.identities:
        assert snap.row(rid) == _original_select(body, rid)
    rid = snap.identities[0][0]
    snap.row(rid)["record"]["identity"]["scene_id"] = "mutated"
    assert snap.row(rid) == _original_select(body, rid)
    with pytest.raises(KeyError):
        snap.row("0" * 64)
    parsed = []
    real = C.strict_json
    monkeypatch.setattr(C, "strict_json", lambda p: (parsed.append(len(p)), real(p))[1])
    ds = c4w.open(release)
    for rec in ds.select(scene_ids=["a0"], components=("core", "detail")).records():
        pass
    per_read = [n for n in parsed if n != len(body)]
    assert parsed.count(len(body)) == 1  # the snapshot is parsed once (at construction)
    assert max(per_read) < len(body) / 10


def _validate_detail_loop(detail, core):  # straightforward reference implementation
    offsets = detail["path_interaction_offsets"]
    if not np.array_equal(np.diff(offsets), core["interaction_count"]):
        raise ValueError("interaction count association mismatch")
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        flags = 0
        for code in detail["interaction_type"][start:stop]:
            flags |= 1 << int(code)
        if flags != int(core["interaction_flags"][i]):
            raise ValueError("interaction flags association mismatch")


def test_vectorized_detail_check_matches_original():
    from synth import core_and_detail
    rng = np.random.default_rng(5)
    for trial in range(200):
        core, det = core_and_detail(rng, zero_links=trial % 17 == 0)
        core, det = dict(core), dict(det)
        if trial % 3 == 1 and len(core["interaction_flags"]):
            flags = core["interaction_flags"].copy()
            flags[rng.integers(len(flags))] ^= 2
            core["interaction_flags"] = flags
        if trial % 5 == 2 and len(core["interaction_count"]):
            counts = core["interaction_count"].copy()
            counts[rng.integers(len(counts))] += 1
            core["interaction_count"] = counts
        results = []
        for fn in (_validate_detail_loop, C.validate_detail):
            try:
                fn(det, core)
                results.append(None)
            except ValueError as exc:
                results.append(str(exc))
        assert results[0] == results[1]


# ----------------------------------------------------------------------------- partial availability
def test_partial_downloads_and_unpublished_detail(copy):
    detail_tar = "payload/main/7ghz/detail/inc-small/train/g-a-7ghz-detail.tar"
    (copy / detail_tar).unlink()
    ds = c4w.open(copy)
    sel = ds.select(scene_ids=["a0", "a1", "b0"], components=("core", "detail", "positions"))
    assert sel.scene_ids == ["a0", "a1", "b0"]  # nothing is silently dropped
    report = sel.missing()
    assert report.files == [detail_tar]
    assert any("detail b0" in u for u in report.unpublished)
    assert "payload/main/7ghz/detail/inc-small/train/g-a-7ghz-detail.tar" in __import__("shlex").split(report.hf_command())
    bs7 = ds.scene("a0").frequencies[F7].base_stations[0]
    with pytest.raises(c4w.NotDownloadedError) as err:
        bs7.read(detail=True)
    assert err.value.paths == [detail_tar]
    bs7.read()  # core alone is still available
    with pytest.raises(c4w.NotPublishedError, match="not published"):
        ds.scene("b0").frequencies[F35].base_stations[0].read(detail=True)
    local = sel.local()
    keys = [(r.scene_id, r.frequency_hz, r.bs_id) for r in local.records()]
    assert keys and all(hz == F35 and s in ("a0", "a1") for s, hz, _ in keys)
    core_only = ds.select(scene_ids=["a0", "b0"], components=("core",)).local()
    assert len(list(core_only.records())) == 2 * 2 * 3


def test_positions_and_geometry_without_channel_payloads(copy, tmp_path):
    for tar in copy.glob("payload/main/[37]*ghz/**/*.tar"):
        tar.unlink()
    ds = c4w.open(copy)
    scene = ds.scene("b1")
    assert scene.read_positions().ue.shape == (N_UES, 3)
    assert scene.extract_geometry(tmp_path / "g").exists()
    report = ds.select(scene_ids=["b1"], components=("core",)).missing()
    assert len(report.files) == 2 and all("core" in p for p in report.files)
    assert not ds.select(scene_ids=["b1"], components=("positions", "geometry")).missing()


def test_missing_group_metadata_is_reported(copy):
    (copy / "groups/g-b/snapshot.json").unlink()
    ds = c4w.open(copy)
    report = ds.select(scene_ids=["b0"], components=("core",)).missing()
    assert "groups/g-b/snapshot.json" in report.files
    with pytest.raises(c4w.NotDownloadedError):
        ds.scene("b0").frequencies  # noqa: B018


# ----------------------------------------------------------------------------- extraction
def _arrays_equal(a: c4w.Record, b: c4w.Record) -> None:
    assert set(a.core) == set(b.core)
    for k in a.core:
        np.testing.assert_array_equal(a.core[k], b.core[k])
        assert a.core[k].dtype == b.core[k].dtype
    assert (a.detail is None) == (b.detail is None)
    for k in a.detail or ():
        np.testing.assert_array_equal(a.detail[k], b.detail[k])


def test_extract_reopen_and_subset_semantics(release, tmp_path):
    ds = c4w.open(release)
    sel = ds.select(tier="small", frequencies_hz=[F35], bs_ids=[0, 2], components=ALL)
    out = c4w.extract(sel, tmp_path / "sub", verify="full")
    sub = c4w.open(out)
    assert sub.kind == "subset" and sub.select().scene_ids == ["a0", "a1", "c0"]
    assert sub.scene("a1").frequencies.ids == [F35]
    assert sub.scene("a1").frequencies[F35].base_stations.ids == [0, 2]
    for scene in sub.select(components=ALL).scenes():
        for rec in scene.records():
            src = ds.scene(rec.scene_id).frequencies[rec.frequency_hz].base_stations[rec.bs_id].read(detail=True)
            _arrays_equal(src, rec)
    assert not sub.select(components=ALL).missing()
    assert (out / "records/scene=a1/band_hz=3500000000/component=core/bs00002.npz").is_file()
    victim = "records/scene=a1/band_hz=3500000000/component=core/bs00002.npz"
    (out / victim).unlink()
    assert sub.select(components=ALL).missing().files == [victim]
    with pytest.raises(c4w.UnknownIDError):
        sub.scene("b0")
    with pytest.raises(c4w.NotPublishedError, match="not included in this subset"):
        c4w.extract(ds.select(scene_ids=["a0"], frequencies_hz=[F35], components=("core",)), tmp_path / "core-only")
        c4w.open(tmp_path / "core-only").scene("a0").frequencies[F35].base_stations[0].read(detail=True)


def test_tiny_uses_the_subset_representation(release, tmp_path):
    tiny = c4w.open(c4w.extract(c4w.open(release).select(tier="tiny", components=ALL), tmp_path / "tiny"))
    assert tiny.kind == "subset" and tiny.select().scene_ids == ["a0"] and tiny.scene("a0").tier == "tiny"
    assert len(list(tiny.select(components=ALL).records())) == 2 * 3


def test_extract_safety(release, tmp_path, monkeypatch):
    ds = c4w.open(release)
    sel = ds.select(tier="tiny", components=("core", "positions"))
    dest = tmp_path / "x"
    c4w.extract(sel, dest)
    with pytest.raises(FileExistsError):
        c4w.extract(sel, dest)
    plans = []
    c4w.extract(sel, dest, skip_existing=True, confirm=lambda p: (plans.append(p), True)[1])
    assert plans[0].skipped_existing == 2 * 3 + 1 and c4w.open(dest).kind == "subset"
    monkeypatch.setattr(prepare, "_free", lambda d: 10)
    with pytest.raises(c4w.InsufficientSpaceError):
        c4w.extract(sel, tmp_path / "y")
    c4w.extract(sel, tmp_path / "y", check_space=False)  # explicit bypass
    monkeypatch.undo()
    assert c4w.extract(sel, tmp_path / "z", confirm=lambda p: False) and not (tmp_path / "z" / "c4w-root.json").exists()

    def crash(*a, **k):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(prepare, "_write_root", crash)
    with pytest.raises(RuntimeError):
        c4w.extract(sel, tmp_path / "w")
    with pytest.raises(c4w.FormatError, match="interrupted"):
        c4w.open(tmp_path / "w")


def test_extract_refuses_missing_inputs(copy, tmp_path):
    (copy / "payload/main/3.5ghz/core/inc-small/train/g-a-3.5ghz-core.tar").unlink()
    sel = c4w.open(copy).select(scene_ids=["a0"])
    with pytest.raises(c4w.NotDownloadedError):
        c4w.extract(sel, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    out = c4w.extract(sel.local(), tmp_path / "out2")
    assert {r.frequency_hz for r in c4w.open(out).select().records()} == {F7}


# ----------------------------------------------------------------------------- cache
def test_cache_roundtrip_lazy_open_and_verify(release, tmp_path):
    ds = c4w.open(release)
    sel = ds.select(tier="small", components=ALL)
    out = c4w.to_cache(sel, tmp_path / "npy", verify="full")
    cache = c4w.open(out)
    assert cache.kind == "cache" and cache._root._groups == {}
    n = 0
    for scene in cache.select(components=ALL).scenes():
        np.testing.assert_array_equal(scene.read_positions(verify="full").ue, ds.scene(scene.id).read_positions().ue)
        for rec in scene.records(verify="full"):
            src = ds.scene(rec.scene_id).frequencies[rec.frequency_hz].base_stations[rec.bs_id].read(detail=True)
            _arrays_equal(src, rec)
            assert not rec.core["tau"].flags.writeable
            n += 1
    assert n == 3 * 2 * 3
    assert cache.scene("a0").open_geometry().namelist()
    tau = out / "cache/g-a/core-3500000000/tau.npy"
    m = np.load(tau, mmap_mode="r+")
    m[0] += 1.0
    m.flush()
    del m
    cache = c4w.open(out)
    bs = cache.scene("a0").frequencies[F35].base_stations[0]
    bs.read(verify="basic")  # basic on a cache checks dtypes and shapes only
    with pytest.raises(c4w.IntegrityError, match="content proof"):
        bs.read(verify="full")


def test_cache_from_a_loose_subset(release, tmp_path):
    sub = c4w.extract(c4w.open(release).select(tier="tiny", components=("core", "positions")), tmp_path / "sub")
    cache = c4w.open(c4w.to_cache(c4w.open(sub).select(components=("core", "positions")), tmp_path / "npy"))
    assert len(list(cache.select().records())) == 2 * 3


# ----------------------------------------------------------------------------- transforms
def test_transforms(release):
    rec = c4w.open(release).scene("a1").frequencies[F7].base_stations[2].read()
    fc = rec.frequency_hz
    a = rec.core["a_real"].astype(np.float64) + 1j * rec.core["a_imag"].astype(np.float64)
    tau = rec.core["tau"].astype(np.float64)
    off = rec.core["link_path_offsets"]
    g = T.complex_gain(rec)
    assert g.dtype == np.complex64 and np.array_equal(g.real, rec.core["a_real"])
    np.testing.assert_allclose(T.path_power_db(rec), 10 * np.log10(np.abs(a) ** 2))
    ds_ = T.delay_spread(rec)
    for i in range(rec.n_links):
        w, t = np.abs(a[off[i]:off[i + 1]]) ** 2, tau[off[i]:off[i + 1]]
        mean = (w * t).sum() / w.sum()
        np.testing.assert_allclose(ds_[i], np.sqrt((w * (t - mean) ** 2).sum() / w.sum()), rtol=1e-9, atol=1e-18)
    offsets = np.array([-60e3, -30e3, 0.0, 15e3, 120e3])
    for ref in ("absolute", "first_retained"):
        H = T.frequency_response(rec, offsets, delay_reference=ref)
        assert H.shape == (rec.n_links, len(offsets))
        for i in range(rec.n_links):
            t = tau[off[i]:off[i + 1]]
            t = t - t.min() if ref == "first_retained" else t
            for k, df in enumerate(offsets):
                expect = np.sum(a[off[i]:off[i + 1]] * np.exp(-2j * np.pi * (fc + df) * t))
                np.testing.assert_allclose(H[i, k], expect, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(T.frequency_response(rec, offsets, ref, max_elements=7), H)
        b = T.baseband_coefficients(rec, ref)
        np.testing.assert_allclose(H[:, 2], np.add.reduceat(b, off[:-1]) if rec.n_paths else 0, rtol=1e-12)
    np.testing.assert_allclose(np.abs(T.frequency_response(rec, offsets, "absolute")),
                               np.abs(T.frequency_response(rec, offsets, "first_retained")), rtol=1e-9)
    with pytest.raises(ValueError):
        T.frequency_response(rec, offsets, delay_reference="normalized")


def test_transforms_on_zero_link_record(bad_release):
    rec = c4w.open(bad_release).scene("a1").frequencies[F7].base_stations[0].read()
    assert T.frequency_response(rec, [0.0, 1e3]).shape == (0, 2)
    assert T.delay_spread(rec).shape == (0,) and T.complex_gain(rec).shape == (0,)


# ----------------------------------------------------------------------------- PyTorch adapter
def _first_ue(rec):
    return int(rec.ue_ids[0])


def test_record_dataset_is_picklable_and_works_in_workers(release):
    sel = c4w.open(release).select(tier="tiny", components=("core", "detail"))
    data = RecordDataset(sel)
    assert len(data) == 2 * 3 and data[0].detail is not None
    clone = pickle.loads(pickle.dumps(RecordDataset(sel, transform=_first_ue)))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(2) as pool:
        got = pool.map(clone.__getitem__, range(len(clone)))
    assert got == [_first_ue(data[i]) for i in range(len(data))]


def test_record_dataset_with_dataloader(release):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    data = RecordDataset(c4w.open(release).select(tier="tiny", components=("core",)))
    assert len(next(iter(DataLoader(data, batch_size=4, collate_fn=list)))) == 4
    assert isinstance(next(iter(DataLoader(data, batch_size=None))), c4w.Record)
    del torch


# ----------------------------------------------------------------------------- CLI
def test_cli(release, tmp_path, capsys):
    assert cli(["info", str(release)]) == 0
    assert "test-1" in capsys.readouterr().out
    assert cli(["scenes", str(release), "--tier", "small"]) == 0
    assert capsys.readouterr().out.splitlines()[1:] == ["a0\ttrain\tmetropolitan\ttiny", "a1\ttrain\tmetropolitan\tsmall",
                                                        "c0\ttest\turban\tsmall"]
    assert cli(["missing", str(release), "--tier", "small"]) == 0
    sub = tmp_path / "sub"
    assert cli(["extract", str(release), str(sub), "--tier", "tiny", "--frequency", str(F35), "--yes"]) == 0
    assert "Continue" not in capsys.readouterr().out
    assert cli(["extract", str(release), str(tmp_path / "s2"), "--tier", "tiny"]) == 1  # stdin is not a terminal
    assert cli(["cache", str(sub), str(tmp_path / "npy"), "--yes", "--no-space-check"]) == 0
    assert cli(["verify", str(tmp_path / "npy"), "--level", "full"]) == 0
    assert "passed full" in capsys.readouterr().out
    assert cli(["geometry", str(release), "c0", str(tmp_path / "geom")]) == 0
    assert capsys.readouterr().out.strip().endswith("c0/scene.xml")
    assert cli(["scenes", str(release), "--tier", "huge"]) == 2


def test_cli_refused_resumed_extract_is_cancelled(release, tmp_path, capsys, monkeypatch):
    import builtins
    import io
    import sys

    dest = tmp_path / "existing"
    args = ["extract", str(release), str(dest), "--tier", "tiny", "--frequency", str(F35)]
    assert cli([*args, "--yes"]) == 0
    before = {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    capsys.readouterr()

    class Terminal:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", Terminal())
    monkeypatch.setattr(builtins, "input", lambda prompt: "n")
    assert cli([*args, "--skip-existing"]) == 1
    assert "Cancelled." in capsys.readouterr().out
    assert {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file()} == before

    monkeypatch.setattr(sys, "stdin", io.StringIO())
    assert cli([*args, "--skip-existing"]) == 1
    capture = capsys.readouterr()
    assert "Cancelled." in capture.out and "Not a terminal" in capture.err
    assert {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file()} == before


# ----------------------------------------------------------------------------- safety, publication and locations
def _rewrite_member_path(root: Path, record_index: int, new_path: str) -> None:
    import pyarrow.parquet as pq
    from synth import TIERS, write_release_root
    p = root / "groups/g-a/catalog/members.parquet"
    t = pq.read_table(p)
    rows = t.to_pylist()
    rows[record_index]["member_path"] = new_path
    pq.write_table(type(t).from_pylist(rows, schema=t.schema), p)
    write_release_root(root, tiers=TIERS, release="test-1")  # a crafted but self-consistent root


@pytest.mark.parametrize("bad", ["../../escape.npz", "/tmp/escape.npz", "records/../../escape.npz"])
def test_metadata_paths_cannot_escape(copy, tmp_path, bad):
    _rewrite_member_path(copy, 0, bad)
    ds = c4w.open(copy)
    with pytest.raises(c4w.FormatError, match="unsafe path"):
        c4w.extract(ds.select(scene_ids=["a0"]), tmp_path / "out" / "dest")
    assert not list(tmp_path.rglob("escape.npz")) and not Path("/tmp/escape.npz").exists()


def test_writes_never_follow_symlinks(release, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    sel = c4w.open(release).select(tier="tiny", components=("positions",))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "records").symlink_to(outside, target_is_directory=True)
    with pytest.raises(c4w.FormatError, match="symbolic link"):
        c4w.extract(sel, dest, skip_existing=True)
    assert not any(outside.iterdir())
    (dest / "records").unlink()
    (dest / "records/scene=a0").mkdir(parents=True)
    victim = outside / "victim"
    victim.write_text("keep")
    (dest / "records/scene=a0/positions.npz.part").symlink_to(victim)  # stale temp name pointing outside
    c4w.extract(sel, dest, skip_existing=True)
    assert victim.read_text() == "keep" and (dest / "records/scene=a0/positions.npz").is_file()
    scene = c4w.open(release).scene("a0")
    geo = tmp_path / "geo"
    geo.mkdir()
    (geo / "a0").symlink_to(outside, target_is_directory=True)
    with pytest.raises(c4w.FormatError, match="symbolic link"):
        scene.extract_geometry(geo)
    assert victim.read_text() == "keep" and len(list(outside.iterdir())) == 1


def test_geometry_archive_entries_cannot_escape(release, tmp_path, monkeypatch):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("scene.xml", "<scene/>")
        z.writestr("../evil.txt", "x")
    scene = c4w.open(release).scene("a0")
    monkeypatch.setattr(type(scene), "open_geometry", lambda self, verify="basic": zipfile.ZipFile(io.BytesIO(buf.getvalue())))
    with pytest.raises(c4w.FormatError, match="unsafe path"):
        scene.extract_geometry(tmp_path / "geo")
    assert not list(tmp_path.rglob("*.txt")) and not list(tmp_path.rglob("scene.xml"))


def test_local_filters_every_component(copy, tmp_path):
    (copy / "payload/main/shared/geometry/inc-small/train/g-a-geometry.tar").unlink()
    ds = c4w.open(copy)
    sel = ds.select(scene_ids=["a0", "a1", "b0"], components=ALL)
    loc = sel.local()
    assert loc.scene_ids == ["a0", "a1", "b0"]
    assert loc.missing().complete  # nothing it still needs is absent or unpublished
    sub = c4w.open(c4w.extract(loc, tmp_path / "sub"))
    with pytest.raises(c4w.NotPublishedError):
        sub.scene("a0").open_geometry()
    sub.scene("b0").open_geometry().close()
    assert {r.scene_id for r in sub.select(components=ALL).records()} == {"a0", "a1"}  # b0 has no detail
    assert sub.scene("b0").read_positions().ue.shape == (N_UES, 3)
    (copy / "groups/g-b/snapshot.json").unlink()
    ds = c4w.open(copy)
    assert ds.select(scene_ids=["b0"], components=("geometry",)).local().scene_ids == []
    assert ds.select(scene_ids=["b0"], components=("core",)).local().scene_ids == []


def test_cache_parts_load_independently(release, tmp_path):
    out = c4w.to_cache(c4w.open(release).select(tier="tiny", components=ALL), tmp_path / "npy")
    (out / "cache/g-a/detail-7000000000/index.json").unlink()
    cache = c4w.open(out)
    bs = cache.scene("a0").frequencies[F7].base_stations[1]
    with pytest.raises(c4w.NotDownloadedError):
        bs.read(detail=True)
    assert bs.read().n_links >= 0  # the core part is unaffected and state was not poisoned
    report = cache.select(components=ALL).missing()
    assert report.files == ["cache/g-a/detail-7000000000/index.json"]


def test_missing_shared_schema_is_reported(copy):
    (copy / "schema/arrays.json").unlink()
    ds = c4w.open(copy)
    report = ds.select(scene_ids=["a0"]).missing()
    assert "schema/arrays.json" in report.files and "groups/g-a/snapshot.json" not in report.files
    with pytest.raises(c4w.NotDownloadedError) as err:
        ds.scene("a0").read_positions()
    assert err.value.paths == ["schema/arrays.json"]


def test_records_pickle_with_read_only_arrays(release, tmp_path):
    ds = c4w.open(release)
    for rec in (ds.scene("a0").frequencies[F35].base_stations[0].read(detail=True),
                c4w.open(c4w.to_cache(ds.select(tier="tiny", components=("core",)), tmp_path / "npy"))
                .scene("a0").frequencies[F35].base_stations[0].read()):
        back = pickle.loads(pickle.dumps(rec))
        _arrays_equal(rec, back)
        assert not back.core["tau"].flags.writeable and back.get_link(int(back.ue_ids[0])) is not None


def test_record_dataset_default_route_with_workers(release):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    data = RecordDataset(c4w.open(release).select(tier="small", components=("core", "detail")))
    single = [r.id for r in DataLoader(data, batch_size=None)]
    for ctx in ("fork", "spawn"):
        multi = list(DataLoader(data, batch_size=None, num_workers=2, multiprocessing_context=ctx))
        assert [r.id for r in multi] == single and not multi[0].core["tau"].flags.writeable
    batches = list(DataLoader(data, batch_size=4, collate_fn=list, num_workers=2))
    assert [r.id for b in batches for r in b] == single
    del torch


def test_missing_report_distinguishes_unpublished(release, capsys):
    report = c4w.open(release).select(scene_ids=["b0"], components=("core", "detail")).missing()
    assert not report and not report.complete and report.unpublished
    assert cli(["missing", str(release), "--scenes", "b0", "--components", "core,detail"]) == 1
    assert "not published" in capsys.readouterr().out


def test_selection_operations_require_group_metadata(copy, tmp_path):
    (copy / "groups/g-b/snapshot.json").unlink()
    ds = c4w.open(copy)
    geo = ds.select(scene_ids=["b0"], components=("geometry",))
    assert ds.scene("b0").open_geometry().namelist()  # direct reads need only the members table
    assert geo.local().scene_ids == []                 # selection-level: one rule for every component
    report = geo.missing()
    assert report.files == ["groups/g-b/snapshot.json"] and not report.complete
    with pytest.raises(c4w.NotDownloadedError) as err:
        c4w.extract(geo, tmp_path / "out")
    assert err.value.paths == ["groups/g-b/snapshot.json"] and not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="nothing to write"):
        c4w.extract(geo.local(), tmp_path / "empty")
    (copy / "schema/arrays.json").unlink()
    with pytest.raises(c4w.NotDownloadedError) as err:
        c4w.extract(ds.select(scene_ids=["a0"], components=("geometry",)), tmp_path / "out2")
    assert err.value.paths == ["schema/arrays.json"]


def test_cache_geometry_does_not_touch_numeric_parts(release, tmp_path):
    out = c4w.to_cache(c4w.open(release).select(tier="tiny", components=("core", "geometry")), tmp_path / "npy")
    (out / "groups/g-a/snapshot.json").unlink()
    cache = c4w.open(out)
    assert cache.scene("a0").open_geometry().namelist()
    sel = cache.select(scene_ids=["a0"], components=("geometry",))
    assert sel.missing().files == ["groups/g-a/snapshot.json"] and sel.local().scene_ids == []


def test_download_commands_follow_the_published_location(tmp_path):
    root = make_release(tmp_path / "repo" / "releases" / "main", hf_path="releases/main")
    tar = "payload/main/3.5ghz/core/inc-small/train/g-a-3.5ghz-core.tar"
    (root / tar).unlink()
    report = c4w.open(root).select(scene_ids=["a0"], frequencies_hz=[F35], components=("core",)).missing()
    assert report.files == [tar] and report.repo_files == ["releases/main/" + tar]
    assert report.local_dir == tmp_path / "repo"
    import shlex
    argv = shlex.split(report.hf_command(revision="v1.0"))
    assert argv == ["hf", "download", "org/Channels4World", "--repo-type", "dataset", "--revision", "v1.0",
                    "--local-dir", str(tmp_path / "repo"), "--include", "releases/main/" + tar]
    assert "--revision" not in shlex.split(report.hf_command())
    moved = tmp_path / "elsewhere"
    shutil.copytree(root, moved)
    with pytest.raises(ValueError, match="not at"):
        c4w.open(moved).select(scene_ids=["a0"], frequencies_hz=[F35], components=("core",)).missing().local_dir
    complete = c4w.open(root).select(scene_ids=["a1"], components=("positions",)).missing()
    with pytest.raises(ValueError, match="nothing to download"):
        complete.hf_command()


def test_user_subsets_are_not_published_roots(release, tmp_path):
    from synth import set_root_publication
    out = c4w.extract(c4w.open(release).select(tier="tiny", components=ALL), tmp_path / "tiny")
    victim = "records/scene=a0/positions.npz"
    (out / victim).unlink()
    report = c4w.open(out).select(components=ALL).missing()
    assert report.files == [victim]
    with pytest.raises(ValueError, match="not a published"):
        report.hf_command()
    set_root_publication(out, hf_repo="org/Channels4World", hf_path="")
    assert c4w.open(out).select(components=ALL).missing().repo_files == [victim]
    set_root_publication(out, hf_repo="org/Channels4World", hf_path="tiny")
    report = c4w.open(out).select(components=ALL).missing()
    assert report.repo_files == ["tiny/" + victim] and report.local_dir == tmp_path
    assert c4w.open(out).select(components=("core",)).local().scene_ids == ["a0"]


def test_hf_command_quotes_every_argument(tmp_path):
    import shlex
    from channels4world.dataset import MissingReport
    odd = tmp_path / "it's here" / "tiny"
    odd.mkdir(parents=True)
    report = MissingReport(["records/scene=o'neil/positions.npz"], 1, [], "org/it's", "tiny", odd)
    assert shlex.split(report.hf_command(revision="rev; rm -rf x")) == [
        "hf", "download", "org/it's", "--repo-type", "dataset", "--revision", "rev; rm -rf x",
        "--local-dir", str(tmp_path / "it's here"), "--include", "tiny/records/scene=o'neil/positions.npz"]


def test_cli_download_hint_respects_publication(release, tmp_path, capsys):
    out = c4w.extract(c4w.open(release).select(tier="small", components=ALL), tmp_path / "sub")
    for f in list((out / "records").rglob("*.npz"))[:25]:
        f.unlink()
    assert cli(["missing", str(out), "--components", "core,detail,positions,geometry"]) == 1
    text = capsys.readouterr().out
    assert "not a published" in text and "snapshot_download" not in text


def test_library_is_independent_of_archive_layout(tmp_path):
    """Archive paths can change without a library change: locations come from the catalogs."""
    import shlex
    cur = c4w.open(make_release(tmp_path / "nested"))
    new = c4w.open(make_release(tmp_path / "release", layout="release"))
    for scene in ("a0", "b1"):
        a = cur.scene(scene).frequencies[F35].base_stations[1].read()
        b = new.scene(scene).frequencies[F35].base_stations[1].read()
        _arrays_equal(a, b)
    tar = tmp_path / "release/payload/3.5ghz/core/inc1-small/g-a-3.5ghz-core.tar"
    tar.unlink()
    report = new.select(scene_ids=["a0"], frequencies_hz=[F35], components=("core",)).missing()
    assert report.files == ["payload/3.5ghz/core/inc1-small/g-a-3.5ghz-core.tar"]
    assert "payload/3.5ghz/core/inc1-small/g-a-3.5ghz-core.tar" in shlex.split(report.hf_command())


def test_reads_from_hugging_face_cache_layout(release, tmp_path):
    """snapshot_download's default cache: every file is a symlink into a separate blobs folder."""
    import os
    snap, blobs = tmp_path / "hub/snapshots/rev", tmp_path / "hub/blobs"
    blobs.mkdir(parents=True)
    for i, f in enumerate(p for p in release.rglob("*") if p.is_file()):
        blob = blobs / f"blob{i}"
        shutil.copy2(f, blob)
        link = snap / f.relative_to(release)
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blob, link.parent), link)
    ds = c4w.open(snap)
    rec = ds.scene("a0").frequencies[F35].base_stations[0].read(detail=True, verify="full")
    _arrays_equal(rec, c4w.open(release).scene("a0").frequencies[F35].base_stations[0].read(detail=True))
    assert ds.scene("a0").read_positions().ue.shape == (N_UES, 3)
    tampered = snap / "schema/arrays.json"
    target = tampered.resolve()
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="size/hash"):
        c4w.open(snap).scene("a1").frequencies[F35].base_stations[0].read()


# ----------------------------------------------------------------------------- availability, filters and group reuse
def test_skip_existing_extends_but_never_drops(release, tmp_path):
    ds = c4w.open(release)
    dest = tmp_path / "sub"
    c4w.extract(ds.select(tier="small", frequencies_hz=[F35], components=("core", "positions")), dest)
    with pytest.raises(FileExistsError, match="outside this selection"):  # 7 GHz alone would drop 3.5 GHz
        c4w.extract(ds.select(tier="small", frequencies_hz=[F7], components=("core", "positions")), dest,
                    skip_existing=True)
    assert c4w.open(dest).frequencies_hz == [F35]  # the refused call changed nothing
    plans = []
    both = ds.select(tier="small", components=("core", "positions"))
    c4w.extract(both, dest, skip_existing=True, confirm=lambda p: (plans.append(p), True)[1])
    assert plans[0].skipped_existing == 3 * 3 + 3  # 3.5 GHz core records and positions kept as they were
    sub = c4w.open(dest)
    assert sub.frequencies_hz == [F35, F7]
    assert {r.frequency_hz for r in sub.select().records()} == {F35, F7}
    c4w.extract(both, dest, skip_existing=True)  # resuming the same selection is still fine
    import json
    other = tmp_path / "other"
    shutil.copytree(dest, other)
    manifest = json.loads((other / "c4w-root.json").read_text())
    manifest["collection_id"] = "another-collection"
    (other / "c4w-root.json").write_text(json.dumps(manifest))
    with pytest.raises(FileExistsError, match="different"):
        c4w.extract(both, other, skip_existing=True)


def test_prepared_roots_list_only_their_carriers(release, tmp_path):
    ds = c4w.open(release)
    sel = ds.select(tier="small", frequencies_hz=[F35], components=("core", "positions"))
    for out in (c4w.extract(sel, tmp_path / "loose"), c4w.to_cache(sel, tmp_path / "npy")):
        sub = c4w.open(out)
        assert sub.frequencies_hz == [F35]
        with pytest.raises(c4w.UnknownIDError, match="no frequency"):
            sub.select(frequencies_hz=[F7])
    geo = c4w.open(c4w.extract(ds.select(tier="tiny", components=("geometry",)), tmp_path / "geo"))
    assert geo.frequencies_hz == []


def test_absent_requested_records_are_reported(release, capsys):
    ds = c4w.open(release)
    report = ds.select(scene_ids=["a0"], bs_ids=[99], components=("core",)).missing()
    assert not report.complete and not report.files
    assert report.unpublished == [f"core a0 {F35} Hz BS 99", f"core a0 {F7} Hz BS 99"]
    partial = ds.select(scene_ids=["a0"], frequencies_hz=[F35], bs_ids=[0, 99], components=("core",))
    assert partial.missing().unpublished == [f"core a0 {F35} Hz BS 99"]
    assert [r.bs_id for r in partial.records()] == [0]  # iteration reads what exists
    assert ds.select(scene_ids=["a0"], bs_ids=[0, 2], components=("core",)).missing().complete
    assert ds.select(scene_ids=["a0"], frequencies_hz=[F35], components=("core",)).missing().complete
    assert cli(["missing", str(release), "--scenes", "a0", "--bs", "99", "--components", "core"]) == 1
    assert "BS 99" in capsys.readouterr().out


def test_profile_filter(release, capsys):
    ds = c4w.open(release)
    assert ds.profiles == ["metropolitan", "suburban", "urban"]  # the release catalog's spelling
    assert ds.select(profile="metropolitan").scene_ids == ["a0", "a1"]
    assert ds.select(profile=["urban", "metropolitan"]).scene_ids == ["a0", "a1", "c0"]
    assert ds.select(profile="suburban", tier="medium").scene_ids == ["b0"]
    assert ds.splits["train"].select(profile="urban").scene_ids == []
    with pytest.raises(c4w.UnknownIDError, match="available: metropolitan, suburban, urban"):
        ds.select(profile="metro")  # scene IDs say metro_; the profile name does not
    with pytest.raises(TypeError):
        ds.select(profile=[1])
    assert "profile='metropolitan'" in repr(ds.select(profile="metropolitan"))
    assert cli(["scenes", str(release), "--profile", "urban", "--profile", "suburban"]) == 0
    assert [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()[1:]] == ["b0", "b1", "c0"]


def test_cli_missing_pins_the_revision(tmp_path, capsys):
    root = make_release(tmp_path / "repo")
    tar = "payload/main/3.5ghz/core/inc-small/train/g-a-3.5ghz-core.tar"
    (root / tar).unlink()
    args = ["missing", str(root), "--scenes", "a0", "--frequency", str(F35), "--components", "core"]
    assert cli(args) == 1
    out = capsys.readouterr().out
    assert "--revision REVISION" in out and "Replace REVISION" in out
    assert cli(args + ["--revision", "v1.0.0"]) == 1
    out = capsys.readouterr().out
    assert "--revision v1.0.0" in out and "Replace REVISION" not in out


def test_cli_verify_separates_unpublished_from_failures(copy, capsys):
    args = ["verify", str(copy), "--scenes", "b0", "--components", "core,detail", "--level", "basic"]
    assert cli(args) == 0  # g-b publishes no detail: reported, not a failure
    out = capsys.readouterr().out
    assert "NOT PUBLISHED" in out and "FAIL" not in out and "0 failed" in out
    _corrupt_member(copy, "a0", F35, "core", 0)
    assert cli(["verify", str(copy), "--scenes", "a0", "--components", "core", "--level", "basic"]) == 1
    assert "FAIL a0" in capsys.readouterr().out
    (copy / "payload/main/7ghz/core/inc-small/train/g-a-7ghz-core.tar").unlink()
    assert cli(["verify", str(copy), "--scenes", "a1", "--frequency", str(F7), "--components", "core"]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_reloaded_groups_are_not_validated_again(release, monkeypatch):
    from channels4world import _root
    monkeypatch.setattr(_root, "GROUP_CACHE", 1)
    calls = []
    real = C.validate_header
    monkeypatch.setattr(C, "validate_header", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ds = c4w.open(release)
    order = ["a0", "c0", "a1", "b0", "a0", "c0"]  # every step evicts the previous group
    for s in order:
        ds.scene(s).frequencies[F35].base_stations[0].read()
    first_pass = len(calls)
    for s in order:
        ds.scene(s).frequencies[F35].base_stations[0].read()
    assert len(calls) == first_pass  # reloads re-check hashes but do not re-validate rows
    assert len(ds._root._validated_snapshots) == 3


def test_reloaded_group_still_checks_snapshot_bytes(copy, monkeypatch):
    from channels4world import _root
    monkeypatch.setattr(_root, "GROUP_CACHE", 1)
    ds = c4w.open(copy)
    ds.scene("a0").frequencies[F35].base_stations[0].read()
    ds.scene("c0").frequencies[F35].base_stations[0].read()  # evicts g-a
    p = copy / "groups/g-a/snapshot.json"
    p.write_bytes(p.read_bytes().replace(b'"version":1', b'"version": 1', 1))
    with pytest.raises(c4w.IntegrityError, match="differs from the release root"):
        ds.scene("a0").frequencies[F35].base_stations[0].read()


def _mixed_subset(release, tmp_path):
    """a0 keeps only positions, a1 keeps positions and 3.5 GHz core."""
    ds = c4w.open(release)
    full = c4w.extract(ds.select(scene_ids=["a0", "a1"], frequencies_hz=[F35], components=("positions", "core")),
                       tmp_path / "full")
    for f in (full / "records/scene=a0").rglob("*/component=core/*.npz"):
        f.unlink()
    mixed = c4w.open(full).select(components=("positions", "core")).local()
    return c4w.open(c4w.extract(mixed, tmp_path / "mixed"))


def test_scene_without_channel_records_is_reported(release, tmp_path):
    mx = _mixed_subset(release, tmp_path)
    assert mx.frequencies_hz == [F35]
    for kw in ({}, {"bs_ids": [99]}, {"frequencies_hz": [F35]}, {"bs_ids": [0, 99]}):
        report = mx.select(scene_ids=["a0"], components=("core",), **kw).missing()
        assert not report.complete and report.unpublished == ["core a0: no channel records in this subset"]
    assert mx.select(scene_ids=["a1"], components=("core",)).missing().complete
    assert mx.select(scene_ids=["a0"], components=("positions",)).missing().complete


def test_absent_requests_are_counted_by_every_consumer(release, tmp_path, capsys):
    ds = c4w.open(release)
    sel = ds.select(scene_ids=["a0"], frequencies_hz=[F35], bs_ids=[0, 99], components=("core",))
    assert len(sel.missing().unpublished) == 1
    assert sel.summary()["unpublished"] == 1
    plans = []
    c4w.extract(sel, tmp_path / "x", confirm=lambda p: (plans.append(p), False)[1])
    assert plans[0].unpublished == 1 and "1 requested items are not published" in plans[0].describe()
    assert cli(["verify", str(release), "--scenes", "a0", "--frequency", str(F35), "--bs", "0", "--bs", "99",
                "--components", "core", "--level", "basic"]) == 0
    out = capsys.readouterr().out
    assert f"NOT PUBLISHED core a0 {F35} Hz BS 99" in out and "1 items passed" in out and "1 requested items" in out
    local = sel.local()  # local() narrows on purpose: nothing is reported as absent
    assert local.missing().complete and local.summary()["unpublished"] == 0


def _leave_out_of_root(root: Path, predicate) -> list[str]:
    """Drop files from the root table as the release does for Large/Full detail (archives stay
    catalogued in their groups). Returns the dropped paths."""
    import json
    import pyarrow.parquet as pq
    from channels4world._root import FILES_SCHEMA, ROOT_FILE
    from channels4world.prepare import _table_bytes
    manifest = json.loads((root / ROOT_FILE).read_text())
    ref = manifest["tables"]["files"]
    rows = pq.read_table(root / ref["path"]).to_pylist()
    dropped = [r["path"] for r in rows if predicate(r)]
    body = _table_bytes([r for r in rows if not predicate(r)], FILES_SCHEMA)
    (root / ref["path"]).write_bytes(body)
    manifest["tables"]["files"] = dict(ref, sha256=C.sha256(body), bytes=len(body))
    (root / ROOT_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return dropped


def test_detail_left_out_of_the_root_is_unpublished_not_an_error(copy, capsys):
    dropped = _leave_out_of_root(copy, lambda f: f["kind"] == "payload" and f["component"] == "detail"
                                 and f["group"] == "g-c")
    assert len(dropped) == 2  # both carriers; group metadata still lists the records
    ds = c4w.open(copy)
    sel = ds.select(scene_ids=["a0", "c0"], components=("core", "detail"))
    report = sel.missing()  # used to raise NotPublishedError from Root.locate
    assert not report.files
    detail_c0 = [u for u in report.unpublished if u.startswith("detail c0")]
    assert len(detail_c0) == 2 * 3 and not any("a0" in u for u in report.unpublished)
    s = sel.summary()
    assert s["unpublished"] == 6 and s["records"] == 2 * 2 * 3
    keys = {(r.scene_id, r.frequency_hz) for r in sel.local().records()}
    assert keys == {("a0", F35), ("a0", F7)}  # c0 has core but its requested detail is not hosted
    with pytest.raises(c4w.NotPublishedError, match="not part of this root"):
        ds.scene("c0").frequencies[F35].base_stations[0].read(detail=True)  # direct reads still raise
    ds.scene("c0").frequencies[F35].base_stations[0].read()  # core is unaffected
    assert cli(["missing", str(copy), "--scenes", "c0", "--components", "core,detail"]) == 1
    assert "not published" in capsys.readouterr().out
    assert cli(["verify", str(copy), "--scenes", "c0", "--components", "core,detail", "--level", "basic"]) == 0
    out = capsys.readouterr().out
    assert out.count("NOT PUBLISHED") == 6 and "0 failed" in out


# ----------------------------------------------------------------------------- recipes
def test_recipe_roundtrip_and_cli(release, tmp_path, capsys):
    import json
    ds = c4w.open(release)
    sel = ds.select(tier="small", split="train", profile=["metropolitan"], frequencies_hz=[F7, F35],
                    bs_ids=[2, 0], components=("detail", "positions"))
    recipe = sel.to_recipe(revision="v1.0.0")
    assert recipe == {"format": "channels4world.selection", "format_version": 1, "release": "test-1",
                      "revision": "v1.0.0",
                      "selection": {"tier": "small", "split": "train", "profiles": ["metropolitan"],
                                    "frequencies_hz": [F35, F7], "bs_ids": [0, 2],
                                    "components": ["core", "detail", "positions"]}}
    path = tmp_path / "sel.json"
    path.write_text(c4w.recipe.dumps(recipe))
    again = ds.select_recipe(path)
    assert again.scene_ids == sel.scene_ids and again.components == sel.components
    assert [(r.scene_id, r.frequency_hz, r.bs_id) for r in again.records()] == \
        [(r.scene_id, r.frequency_hz, r.bs_id) for r in sel.records()]
    assert ds.select_recipe(recipe).missing().files == sel.missing().files
    # scene lists keep their order; unset filters are omitted
    by_id = ds.select(scene_ids=["c0", "a0"], components=("core",)).to_recipe()
    assert by_id["selection"] == {"scene_ids": ["c0", "a0"], "components": ["core"]}
    with pytest.raises(ValueError, match="local"):
        sel.local().to_recipe()
    # CLI: write, then plan from it; the recipe's revision pins the command
    out = tmp_path / "cli.json"
    assert cli(["recipe", str(release), "--profile", "urban", "--components", "core,positions",
                "--revision", "v1.0.0", "-o", str(out)]) == 0
    assert json.loads(out.read_text())["selection"] == {"profiles": ["urban"], "components": ["core", "positions"]}
    capsys.readouterr()
    assert cli(["scenes", str(release), "--recipe", str(out)]) == 0
    assert capsys.readouterr().out.splitlines()[1:] == ["c0\ttest\turban\tsmall"]
    assert cli(["scenes", str(release), "--recipe", str(out), "--tier", "tiny"]) == 2
    assert "--recipe replaces" in capsys.readouterr().err
    assert cli(["recipe", str(release), "--tier", "tiny", "-o", str(out)]) == 2  # never overwrites


def test_recipe_missing_command_uses_its_revision(tmp_path, capsys):
    root = make_release(tmp_path / "c4w", hf_repo="org/C4W", hf_path="")
    (root / "payload/main/3.5ghz/core/inc-small/test/g-c-3.5ghz-core.tar").unlink()
    recipe = c4w.open(root).select(scene_ids=["c0"], frequencies_hz=[F35],
                                   components=("core",)).to_recipe(revision="v1.0.0")
    path = tmp_path / "r.json"
    path.write_text(c4w.recipe.dumps(recipe))
    assert cli(["missing", str(root), "--recipe", str(path)]) == 1
    out = capsys.readouterr().out
    assert "--revision v1.0.0" in out and "REVISION" not in out.replace("--revision", "")


@pytest.mark.parametrize("bad, match", [
    ({"format": "other"}, "format"),
    ({"format_version": 2}, "format_version"),
    ({"format_version": True}, "format_version"),
    ({"extra": 1}, "unknown recipe field"),
    ({"revision": "v1; rm -rf /"}, "revision"),
    ({"revision": "../x"}, "revision"),
    ({"selection": {"components": ["core"], "rows": [1]}}, "unknown selection field"),
    ({"selection": {"components": []}}, "non-empty"),
    ({"selection": {"tier": "small"}}, "components is required"),
    ({"selection": {"components": ["core", "audio"]}}, "unknown component"),
    ({"selection": {"components": ["core"], "bs_ids": [0, True]}}, "integers"),
    ({"selection": {"components": ["core"], "frequencies_hz": [3.5e9]}}, "integers"),
    ({"selection": {"components": ["core"], "scene_ids": ["../etc"]}}, "plain names"),
    ({"selection": {"components": ["core"], "profiles": "urban"}}, "list"),
])
def test_recipe_validation_rejects(bad, match):
    base = {"format": "channels4world.selection", "format_version": 1, "release": "test-1",
            "selection": {"components": ["core"]}}
    with pytest.raises(c4w.RecipeError, match=match):
        c4w.recipe.validate({**base, **bad})


def test_recipe_parsing_and_release_check(release):
    ds = c4w.open(release)
    ok = '{"format": "channels4world.selection", "format_version": 1, "release": "test-1", ' \
         '"selection": {"components": ["detail"]}}'
    assert c4w.recipe.loads(ok)["selection"]["components"] == ["core", "detail"]  # detail implies core
    with pytest.raises(c4w.RecipeError, match="duplicate"):
        c4w.recipe.loads(ok[:-1] + ', "release": "x"}')
    with pytest.raises(c4w.RecipeError, match="JSON"):
        c4w.recipe.loads(ok.replace('"selection"', '"revision": NaN, "selection"'))
    other = dict(c4w.recipe.loads(ok), release="2.0.0")
    with pytest.raises(c4w.RecipeError, match="release 2.0.0"):
        ds.select_recipe(other)
    with pytest.raises(c4w.UnknownIDError):  # catalog checks still apply
        ds.select_recipe({**c4w.recipe.loads(ok), "selection": {"tier": "huge", "components": ["core"]}})


# ----------------------------------------------------------------------------- fixed-size arrays
def test_pad_paths_and_interactions(release):
    rec = c4w.open(release).scene("a0").frequencies[F35].base_stations[1].read(detail=True)
    off = rec.core["link_path_offsets"]
    tau, mask = T.pad_paths(rec.core, "tau")
    assert tau.shape == mask.shape == (rec.n_links, 16) and tau.dtype == np.float32
    for i in range(rec.n_links):
        n = off[i + 1] - off[i]
        assert mask[i].sum() == n and not mask[i, n:].any()
        np.testing.assert_array_equal(tau[i, :n], rec.core["tau"][off[i]:off[i + 1]])
        assert not tau[i, n:].any()
    v, vmask = T.pad_interactions(rec.detail, "interaction_vertex")
    assert v.shape == (rec.n_paths, 4, 3) and vmask.shape == (rec.n_paths, 4)
    np.testing.assert_array_equal(vmask.sum(1), rec.core["interaction_count"])
    per_link, per_link_mask = T.pad_paths(rec.core, v)
    assert per_link.shape == (rec.n_links, 16, 4, 3)
    link = rec.get_link(int(rec.ue_ids[0]))
    lv, lm = T.pad_interactions(link.detail, "interaction_vertex")
    np.testing.assert_array_equal(per_link[0, :link.n_paths], lv)
    assert per_link_mask[0].sum() == link.n_paths
    # The card's NumPy snippet gives the same arrays
    def card_pad(core, name, max_paths=16):
        offsets = np.asarray(core["link_path_offsets"])
        values = np.asarray(core[name])
        counts = np.diff(offsets)
        rows = np.repeat(np.arange(counts.size), counts)
        cols = np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts)
        padded = np.zeros((counts.size, max_paths), values.dtype)
        padded[rows, cols] = values
        return padded, np.arange(max_paths) < counts[:, None]
    for got, want in zip(T.pad_paths(rec.core, "a_real"), card_pad(rec.core, "a_real")):
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="more than the padded width"):
        T.pad_paths(rec.core, "tau", max_paths=1)
    with pytest.raises(ValueError, match="first axis"):
        T.pad_paths(rec.core, rec.core["tau"][:-1])


def test_pad_paths_on_zero_link_record(bad_release):
    rec = c4w.open(bad_release).scene("a1").frequencies[F7].base_stations[0].read()
    tau, mask = T.pad_paths(rec.core, "tau")
    assert tau.shape == mask.shape == (0, 16)


def test_recipe_cases_shared_with_the_website_helper():
    """tests/data/recipe-cases.json is also used by the website's download helper
    (src/selector/recipe-cases.json); both must reach the same verdicts."""
    import json
    cases = json.loads((Path(__file__).parent / "data/recipe-cases.json").read_text())
    for case in cases:
        if case["valid"]:
            c4w.recipe.loads(case["text"])
        else:
            with pytest.raises(c4w.RecipeError):
                c4w.recipe.loads(case["text"])


def test_website_helper_recipes_are_accepted_unchanged():
    """Recipes written by the website's download helper (a fixed sample) parse and
    re-serialize byte for byte, so either side can produce them."""
    import json
    for text in json.loads((Path(__file__).parent / "data/website-recipes.json").read_text()):
        assert c4w.recipe.dumps(c4w.recipe.loads(text)) == text
