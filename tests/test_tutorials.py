"""Fast, network-free tutorial structure and generation checks (no extra dependencies)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tutorials-src' / f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_generated_library_tutorials_are_current():
    build = module('build_tutorials')
    for path, expected in build.outputs(ROOT).items():
        assert path.read_text() == expected, f'Regenerate {path}'


def test_pages_and_notebooks_share_code_and_clean_outputs():
    build = module('build_tutorials')
    for _, name, cells, surface in build.TUTORIALS:
        nb = build.notebook(cells, name, surface)
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        assert [''.join(c['source']) for c in code] == [t for k, t in cells if k == 'code']
        assert len({c['id'] for c in nb['cells']}) == len(nb['cells'])
        for index, cell in enumerate(code):
            assert cell['outputs'] == [] and cell['execution_count'] is None
            compile(''.join(cell['source']), f'{name}:cell-{index}', 'exec')
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                links = module('check_links').LINK.findall(''.join(cell['source']))
                assert all(t.startswith(('https://', '#')) for t in links)
        if surface == 'huggingface':
            assert not any('import channels4world' in ''.join(c['source']) for c in code)


def test_library_paths_and_anchors():
    failures, count, _ = module('check_links').check(ROOT)
    assert count > 0
    assert failures == []


def test_hf_compatibility_keeps_code_and_is_idempotent(tmp_path):
    build = module('build_tutorials')
    path = tmp_path / 'docs/notebooks/01-read-one-scene-library.ipynb'
    path.parent.mkdir(parents=True)
    original = build.notebook(build.LIB_TUTORIAL, path.name, 'library')
    path.write_text(json.dumps(original))
    output = build.outputs(ROOT, tmp_path)
    updated = json.loads(output[path])
    assert updated['cells'][1:] == original['cells']
    assert updated['cells'][0]['id'] == 'library-tutorial-moved'
    path.write_text(output[path])
    assert build.outputs(ROOT, tmp_path)[path] == output[path]
