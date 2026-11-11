#!/usr/bin/env python3
"""Run every code block of the card and its guides against the synthetic release.

Uses check_tutorials.py's synthetic repository (published layout, tutorial scene and
group names). Python blocks run in order, one namespace per page, in a folder where the
whole repository sits at c4w/. Hub access is replaced: snapshot_download and
hf_hub_download copy from the synthetic repository after applying the patterns,
load_dataset(REPO, config) reads the card's configs from it, and hf:// URLs map to it.
Shell blocks: every ``hf download --include`` pattern must match published files (the
one-scene command exactly BS 0 of the lead scene and the library's metadata, the same
command in the card and the library README), and the ``c4w`` commands and read_archives.py
run for real; ``pip install`` lines are skipped. The library README and guide run the same
way as the card's pages. Values printed are synthetic.

Usage: python3 check_snippets.py LIBRARY_DIR HF_CARD_DIR
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_tutorials as T  # noqa: E402

PAGES = ["README.md", "docs/library-guide.md", "docs/without-library.md", "docs/hf-datasets.md"]
LIB_PAGES = ["README.md", "docs/guide.md"]  # relative to LIBRARY_DIR
HF_PREFIX = f"hf://datasets/{T.REPO}@{T.REVISION}/"
CONFIGS = {"scenes": "c4w-root/scenes.parquet", "files": "c4w-root/files.parquet",
           "tiny_scenes": "tiny/c4w-root/scenes.parquet"}


def blocks(text: str) -> list[tuple[str, str]]:
    return re.findall(r"```(\w*)\n(.*?)```", text, re.S)


def install_fakes(repo: Path, copied: list[str]) -> None:
    import datasets
    import huggingface_hub as hub

    files = sorted(p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file())

    def check_repo(repo_id, repo_type, revision):
        assert (repo_id, repo_type, revision) == (T.REPO, "dataset", T.REVISION), (repo_id, repo_type, revision)

    def snapshot_download(repo_id, *, repo_type=None, revision=None, local_dir=None, allow_patterns=None, **kw):
        check_repo(repo_id, repo_type, revision)
        pats = [allow_patterns] if isinstance(allow_patterns, str) else list(allow_patterns or [])
        assert local_dir is not None and pats, "card examples download explicit patterns into a folder"
        hits = [f for f in files if any(fnmatch.fnmatch(f, p) for p in pats)]
        assert hits, f"no published file matches {pats[:3]}"
        for f in hits:
            out = Path(local_dir) / f
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / f, out)
            copied.append(f)
        return str(local_dir)

    def hf_hub_download(repo_id, filename, *, repo_type=None, revision=None, local_dir=None, **kw):
        check_repo(repo_id, repo_type, revision)
        assert filename in files, filename
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / filename, out)
        return str(out)

    real = datasets.load_dataset

    def local(path):
        return str(repo / path[len(HF_PREFIX):]) if isinstance(path, str) and path.startswith(HF_PREFIX) else path

    def load_dataset(path, name=None, *, split=None, revision=None, data_files=None, **kw):
        if path == T.REPO:
            assert revision == T.REVISION and name in CONFIGS, (name, revision)
            return real("parquet", data_files={"catalog": str(repo / CONFIGS[name])}, split=split, **kw)
        if isinstance(data_files, dict):
            data_files = {k: ([local(x) for x in v] if isinstance(v, list) else local(v)) for k, v in data_files.items()}
        return real(path, name, split=split, data_files=data_files, **kw)

    hub.snapshot_download, hub.hf_hub_download = snapshot_download, hf_hub_download
    datasets.load_dataset = load_dataset


def check_hf_command(cmd: str, files: list[str]) -> tuple[bool, str]:
    args = shlex.split(cmd.replace("\\\n", " "))
    incl = [args[i + 1] for i, a in enumerate(args) if a == "--include"]
    if args[:3] != ["hf", "download", T.REPO] or "--revision" not in args or not incl:
        return False, "not a pinned hf download with --include patterns"
    if args[args.index("--revision") + 1] != T.REVISION:
        return False, "wrong revision"
    empty = [p for p in incl if not any(fnmatch.fnmatch(f, p) for f in files)]
    if empty:
        return False, f"patterns match nothing: {empty}"
    got = sorted({f for f in files if any(fnmatch.fnmatch(f, p) for p in incl)})
    return True, f"{len(incl)} pattern(s), {len(got)} file(s)"


def first_includes(text: str) -> list[str]:
    for lang, code in blocks(text):
        if lang == "bash" and "hf download" in code:
            args = shlex.split(code.replace("\\\n", " "))
            return [args[i + 1] for i, a in enumerate(args) if a == "--include"]
    return []


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit("Usage: check_snippets.py LIBRARY_DIR HF_CARD_DIR")
    lib = Path(sys.argv[1]).resolve()
    card = Path(sys.argv[2]).resolve()
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="c4w-snippets-") as tmp:
        tmp = Path(tmp)
        repo = tmp / "repo"
        T.build_repo(lib, repo)
        files = sorted(p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file())
        copied: list[str] = []
        install_fakes(repo, copied)
        env = dict(os.environ, PYTHONPATH=str(lib / "src"))
        pages = [(card, page, page) for page in PAGES] + [(lib, page, "library/" + page) for page in LIB_PAGES]
        for base, rel, page in pages:
            work = tmp / ("work-" + page.replace("/", "-"))
            work.mkdir()
            shutil.copytree(repo, work / "c4w")
            shutil.copy(card / "docs" / "read_archives.py", work / "read_archives.py")
            os.chdir(work)
            ns: dict = {"__name__": "__main__"}
            for i, (lang, code) in enumerate(blocks((base / rel).read_text())):
                label = f"{page}[{i}] {code.strip().splitlines()[0][:60] if code.strip() else ''}"
                if lang == "python":
                    out = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(out):
                            exec(compile(code, label, "exec"), ns)
                        print("ok  ", label)
                    except Exception:
                        print("FAIL", label)
                        print(traceback.format_exc(limit=3))
                        failures.append(label)
                elif lang == "bash":
                    for cmd in re.split(r"(?<!\\)\n", code.strip()):
                        cmd = re.sub(r"\s+#.*$", "", cmd).strip()
                        if not cmd:
                            continue
                        if re.match(r"(python -m )?pip install ", cmd):
                            print("skip", f"{page}[{i}] $ {cmd[:70]}  (install)")
                            continue
                        if cmd.startswith("hf download"):
                            ok, why = check_hf_command(cmd, files)
                        else:
                            argv = shlex.split(cmd.replace("\\\n", " "))
                            if argv[0] == "c4w":
                                argv = [sys.executable, "-c", "import sys; from channels4world.cli import main; "
                                        "sys.exit(main())"] + argv[1:]
                            elif argv[0] == "python":
                                argv[0] = sys.executable
                            r = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True)
                            ok, why = r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
                            why = why[0][:100]
                        print("ok  " if ok else "FAIL", f"{page}[{i}] $ {cmd[:70]}  ({why})")
                        if not ok:
                            failures.append(f"{page}[{i}] {cmd[:70]}: {why}")
            os.chdir(tmp)
        # The card's one-scene command (also the library README's first download) must fetch
        # exactly BS 0 of the lead scene and the metadata the library needs.
        incl = first_includes((card / "README.md").read_text())
        same = incl == first_includes((lib / "README.md").read_text())
        print("ok  " if same else "FAIL", "the card and the library README share the one-scene command")
        if not same:
            failures.append("one-scene command differs between the card and the library README")
        got = {f for f in files if any(fnmatch.fnmatch(f, p) for p in incl)}
        lead = f"tiny/records/scene={T.LEAD}/"
        want_records = {f for f in files if f.startswith(lead) and
                        (f.endswith(("/positions.npz", "/geometry.zip")) or f.endswith("/bs00000.npz"))}
        extra = {f for f in got - want_records if f.startswith("tiny/records/") or
                 (f.startswith("tiny/groups/") and f"/{T.LEAD_GROUP}/" not in f) or not f.startswith("tiny/")}
        ok = bool(want_records) and want_records <= got and not extra
        print("ok  " if ok else "FAIL", "the one-scene command fetches BS 0 of the lead scene and its metadata only")
        if not ok:
            failures.append(f"one-scene command: extra {sorted(extra)[:5]}")
        lead_dir = tmp / "lead"
        for f in got:
            (lead_dir / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / f, lead_dir / f)
        sys.path.insert(0, str(lib / "src"))
        import channels4world as c4w
        rec = c4w.open(lead_dir / "tiny").scene(T.LEAD).frequencies[3_500_000_000].base_stations[0].read(detail=True)
        ok = rec.get_link(T.UE_ID) is not None
        print("ok  " if ok else "FAIL", "the one-scene download alone is enough for a library read")
        if not ok:
            failures.append("one-scene library read")
    print("all snippet checks passed" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
