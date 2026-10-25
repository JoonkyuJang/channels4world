"""``c4w`` command line: a thin wrapper over the Python API."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from . import recipe as R
from ._errors import C4WError, NotDownloadedError, NotPublishedError
from .dataset import COMPONENTS, BaseStation, DEFAULT_COMPONENTS, Dataset, Selection, open as c4w_open
from .prepare import Plan, extract, to_cache


def _selection_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("selection")
    g.add_argument("--split")
    g.add_argument("--tier")
    g.add_argument("--profile", action="append", help="generation profile as spelled in the catalog (repeatable)")
    g.add_argument("--scenes", help="scene IDs, comma-separated, or @file with one ID per line")
    g.add_argument("--frequency", type=int, action="append", metavar="HZ", help="carrier in Hz (repeatable)")
    g.add_argument("--bs", type=int, action="append", metavar="N", help="base station ID (repeatable)")
    g.add_argument("--components",
                   help=f"comma-separated from {','.join(COMPONENTS)} (default: {','.join(DEFAULT_COMPONENTS)})")
    g.add_argument("--recipe", metavar="FILE",
                   help="selection recipe (JSON) instead of the options above; see 'c4w recipe'")


def _select(ds: Dataset, a: argparse.Namespace) -> Selection:
    if a.recipe:
        used = [opt for opt, v in (("--split", a.split), ("--tier", a.tier), ("--profile", a.profile),
                                   ("--scenes", a.scenes), ("--frequency", a.frequency), ("--bs", a.bs),
                                   ("--components", a.components)) if v]
        if used:
            raise ValueError(f"--recipe replaces the other selection options; remove {', '.join(used)}")
        return ds.select_recipe(a.recipe)
    scenes = None
    if a.scenes:
        if a.scenes.startswith("@"):
            with open(a.scenes[1:]) as fh:
                scenes = [line.strip() for line in fh if line.strip()]
        else:
            scenes = [s for s in a.scenes.split(",") if s]
    comps = DEFAULT_COMPONENTS if a.components is None else tuple(
        c.strip() for c in a.components.split(",") if c.strip())
    return ds.select(split=a.split, tier=a.tier, profile=a.profile, scene_ids=scenes, frequencies_hz=a.frequency,
                     bs_ids=a.bs, components=comps)


def _confirm(assume_yes: bool):
    def ask(plan: Plan) -> bool:
        print(plan.describe())
        if assume_yes:
            return True
        if not sys.stdin.isatty():
            print("Not a terminal: pass --yes to proceed.", file=sys.stderr)
            return False
        return input("Continue [y/N] ").strip().lower() in ("y", "yes")
    return ask


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="c4w", description="Channels4World dataset tools")
    ap.add_argument("--version", action="version", version=f"channels4world {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="describe a release, Tiny, subset or cache")
    p.add_argument("path")

    p = sub.add_parser("scenes", help="list selected scenes")
    p.add_argument("path")
    _selection_args(p)

    p = sub.add_parser("missing", help="list files to download for a selection")
    p.add_argument("path")
    _selection_args(p)
    p.add_argument("--paths-only", action="store_true", help="print only the missing paths")
    p.add_argument("--revision", help="dataset revision the catalog was downloaded from (pins the printed command; "
                                      "defaults to the recipe's revision)")
    p.description = "Exit status 0 when everything requested is published and downloaded, else 1."

    p = sub.add_parser("recipe", help="write a selection as a shareable recipe (JSON)")
    p.add_argument("path")
    _selection_args(p)
    p.add_argument("--revision", help="Hugging Face revision the catalog came from, recorded in the recipe")
    p.add_argument("--output", "-o", metavar="FILE", help="write to FILE instead of standard output")
    p.description = ("Records the selection and the release it refers to, not file paths: the recipient "
                     "plans downloads from their own catalog with 'c4w missing PATH --recipe FILE'.")

    for name, helptext in (("extract", "copy a selection to loose files"), ("cache", "convert a selection to NPY")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("path")
        p.add_argument("dest")
        _selection_args(p)
        p.add_argument("--verify", choices=("none", "basic", "full"), default="basic")
        p.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
        p.add_argument("--no-space-check", action="store_true", help="skip the free-space check")
        if name == "extract":
            p.add_argument("--skip-existing", action="store_true", help="keep matching files already in DEST")

    p = sub.add_parser("verify", help="read every selected item at a verify level")
    p.add_argument("path")
    _selection_args(p)
    p.add_argument("--level", choices=("basic", "full"), default="full")

    p = sub.add_parser("geometry", help="extract one scene's geometry; prints the scene XML path")
    p.add_argument("path")
    p.add_argument("scene_id")
    p.add_argument("dest")

    a = ap.parse_args(argv)
    try:
        return _run(a)
    except (C4WError, FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"c4w: {exc}", file=sys.stderr)
        return 2


def _run(a: argparse.Namespace) -> int:
    ds = c4w_open(a.path)
    if a.cmd == "info":
        print(ds)
        return 0
    if a.cmd == "geometry":
        print(ds.scene(a.scene_id).extract_geometry(a.dest))
        return 0
    sel = _select(ds, a)
    if a.cmd == "recipe":
        text = R.dumps(sel.to_recipe(revision=a.revision))
        if a.output:
            with open(a.output, "x", encoding="utf-8") as fh:
                fh.write(text)
            print(f"Wrote {a.output}")
        else:
            print(text, end="")
        return 0
    if a.cmd == "scenes":
        print("scene_id\tsplit\tprofile\ttier")
        for s in sel.scenes():
            print(f"{s.id}\t{s.split}\t{s.profile}\t{s.tier}")
        return 0
    if a.cmd == "missing":
        report = sel.missing()
        if a.paths_only:
            print("\n".join(report.files))
        else:
            print(report)
            if report.files:
                pinned = a.revision or (R.load(a.recipe).get("revision") if a.recipe else None)
                rev = pinned or "REVISION"
                try:
                    report.repo_files  # raises if this root is not published as-is
                    hint = (report.hf_command(revision=rev) if len(report.files) <= 20 else
                            f"(many files) huggingface_hub.snapshot_download(repo_id=..., revision={rev!r}, "
                            "local_dir=report.local_dir, allow_patterns=report.repo_files)")
                    if not pinned:
                        hint += ("\n  Replace REVISION with the dataset revision you downloaded the catalog from "
                                 "(or pass --revision).")
                except ValueError as exc:
                    hint = str(exc)
                print("\nDownload with:\n  " + hint)
        return 0 if report.complete else 1
    if a.cmd in ("extract", "cache"):
        confirm = _confirm(a.yes)
        cancelled = False
        if a.cmd == "extract":
            # The API returns DEST on refusal for compatibility. An existing
            # subset can already have a root, so the CLI must track refusal.
            def confirm_extract(plan: Plan) -> bool:
                nonlocal cancelled
                accepted = confirm(plan)
                cancelled = not accepted
                return accepted

            confirmation = confirm_extract
        else:
            confirmation = confirm
        kwargs = dict(verify=a.verify, check_space=not a.no_space_check, confirm=confirmation)
        if a.cmd == "extract":
            out = extract(sel, a.dest, skip_existing=a.skip_existing, **kwargs)
        else:
            out = to_cache(sel, a.dest, **kwargs)
        if not cancelled and (out / "c4w-root.json").exists():
            print(f"Done: {out}")
            return 0
        print("Cancelled.")
        return 1
    if a.cmd == "verify":
        # Integrity failures and missing downloads fail the command; requested items that are
        # simply not published (or not included in a subset) are reported but do not.
        counts = {"passed": 0, "failed": 0, "missing": 0, "unpublished": 0}

        def run(label: str, check) -> None:
            try:
                check()
                counts["passed"] += 1
            except NotPublishedError as exc:
                counts["unpublished"] += 1
                print(f"NOT PUBLISHED {label}: {exc}")
            except NotDownloadedError as exc:
                counts["missing"] += 1
                print(f"MISSING {label}: {exc}")
            except C4WError as exc:
                counts["failed"] += 1
                print(f"FAIL {label}: {exc}")

        for scene in sel.scenes():
            for label in sel._absent_requests(scene):
                counts["unpublished"] += 1
                print(f"NOT PUBLISHED {label}")
            if "positions" in sel.components:
                run(scene.id, lambda s=scene: s.read_positions(verify=a.level))
            if "geometry" in sel.components:
                run(scene.id, lambda s=scene: s.open_geometry(verify=a.level).close())
            if "core" in sel.components:
                detail = "detail" in sel.components
                for k in sel._keys(scene):
                    run(f"{scene.id} {k.hz} Hz BS {k.bs}",
                        lambda s=scene, k=k: BaseStation(s, k.hz, k.bs).read(detail=detail, verify=a.level))
        line = f"{counts['passed']} items passed {a.level} verification, {counts['failed']} failed"
        if counts["missing"]:
            line += f", {counts['missing']} not downloaded"
        if counts["unpublished"]:
            line += f"; {counts['unpublished']} requested items are not published (not an error)"
        print(line)
        return 1 if counts["failed"] or counts["missing"] else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
