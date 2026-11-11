#!/usr/bin/env python3
"""Check local page/notebook paths and anchors, including staged GitHub/HF links.

This is an offline check. External URLs outside these source trees are reported,
not fetched. New public URLs only become live after publication.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

GH = "https://github.com/JoonkyuJang/channels4world/"
HF = "https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World"
LINK = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


def text(path: Path) -> str:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        return "\n".join("".join(c['source']) for c in nb['cells'] if c['cell_type'] == 'markdown')
    # Do not treat example code as prose links or headings.
    return re.sub(r"```.*?```", "", path.read_text(), flags=re.S)


def anchors(body: str) -> set[str]:
    seen, result = {}, set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", body, re.M):
        heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        number = seen.get(slug, 0)
        seen[slug] = number + 1
        result.add(slug + (f"-{number}" if number else ""))
    result.update(re.findall(r'(?:id|name)=[\"\']([^\"\']+)[\"\']', body))
    return result


def check(root: Path, card: Path | None = None) -> tuple[list[str], int, set[str]]:
    root = root.resolve()
    card = card.resolve() if card else None
    pages = [root / "README.md"]
    for folder in ("docs", "examples", "tutorials-src"):
        pages += list((root / folder).rglob("*.md")) + list((root / folder).rglob("*.ipynb"))
    if card:
        pages += [card / "README.md"] + list((card / "docs").rglob("*.md")) + list((card / "docs").rglob("*.ipynb"))
    failures, checked, external = [], 0, set()
    for source in pages:
        for raw in LINK.findall(text(source)):
            parsed = urlsplit(raw)
            target = None
            if not parsed.scheme:
                target = source.parent / unquote(parsed.path) if parsed.path else source
            elif raw.startswith((GH + "blob/main/", GH + "tree/main/")):
                target = root / unquote(parsed.path.split("/main/", 1)[1])
            elif raw.startswith(GH) and parsed.path.rstrip("/") == "/JoonkyuJang/channels4world":
                target = root / "README.md"
            elif card and raw.startswith(HF + "/blob/main/"):
                target = card / unquote(parsed.path.split("/main/", 1)[1])
            elif card and raw.startswith(HF) and parsed.path.rstrip("/") == "/datasets/Intelligence-Networking-Lab/Channels4World":
                target = card / "README.md"
            else:
                external.add(raw)
                continue
            if target.is_dir():
                target = target / "README.md"
            checked += 1
            if not target.is_file():
                failures.append(f"{source}: missing target {raw}")
            elif parsed.fragment and target.suffix in {".md", ".ipynb"}:
                if unquote(parsed.fragment) not in anchors(text(target)):
                    failures.append(f"{source}: missing anchor {raw}")
    return failures, checked, external


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-card", type=Path)
    args = parser.parse_args()
    failures, checked, external = check(Path(__file__).resolve().parent.parent, args.hf_card)
    for failure in failures:
        print("FAIL", failure)
    print(f"{checked} local paths/anchors checked; {len(failures)} failure(s).")
    print(f"{len(external)} distinct external URLs not fetched (offline check).")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
