# Tutorial maintenance

`build_tutorials.py` is the single editable source for the tutorial cells. It produces
pages and output-free notebooks from the same cells. Library tutorials are published
in this repository. The NumPy-only example is published with the dataset on Hugging
Face; its cells live here so the two ways of reading one link can be tested together.
Do not independently edit the generated pages or notebooks.

From the repository root:

```bash
python -m pip install -e ".[test,torch]" -r tutorials-src/requirements.txt
python tutorials-src/build_tutorials.py
python tutorials-src/build_tutorials.py --check
python tutorials-src/check_links.py
python tutorials-src/check_tutorials.py
python -m pytest
```

The execution check uses a small synthetic release and replaces Hub downloads with
local copies. It verifies download selection, plots, scene splits, training-only
normalization, masked padding and cache invalidation. It runs notebooks separately
and in a shared working directory. It uses the invoking Python interpreter as the
notebook kernel and makes no network transfers. Synthetic outputs are test evidence,
not publishable examples. Runtime checks against the real Hub release are separate.

## Dataset card coordination

With a local dataset-card checkout or staging tree:

```bash
python tutorials-src/build_tutorials.py --hf-card /path/to/hf-card
python tutorials-src/build_tutorials.py --hf-card /path/to/hf-card --check
python tutorials-src/check_links.py --hf-card /path/to/hf-card
python tutorials-src/check_tutorials.py . /path/to/hf-card
python tutorials-src/check_snippets.py . /path/to/hf-card
```

This also generates the NumPy page and notebook on the card, replaces old library
tutorial pages with pointers to GitHub, and labels existing HF library notebooks as
frozen compatibility copies without changing their code. Their previous URLs keep
working. No second editable library tutorial is maintained on HF, and this command
never uploads files or deletes a notebook.

When first publishing this split, publish and verify the GitHub paths before updating
HF navigation. Keep the dataset's release tag unchanged. Update the HF card's tutorial
index separately when adding tutorials; `--check` catches drift in generated files,
and `check_links.py` verifies local targets and anchors across both trees.
