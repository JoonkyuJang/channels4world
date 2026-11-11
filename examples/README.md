# Examples

Start with [Read one scene](../docs/tutorial-library.md)
([notebook](01-read-one-scene-library.ipynb)): download one scene's base station 0
records, inspect geometry and positions, and compare one link at 3.5 and 7 GHz.
It downloads about 26 MB.

Then try [From channel records to a DataLoader](../docs/ml-guide.md)
([notebook](02-dataloader.ipynb)): select Tiny's urban scenes, split by scene,
pad paths, cache preprocessing and form PyTorch batches. It downloads about
42 MB of metadata and 0.4 GB of records.

The [pure NumPy example](https://huggingface.co/datasets/Intelligence-Networking-Lab/Channels4World/blob/main/docs/tutorial-numpy.md)
remains with the dataset on Hugging Face. It reads the same link without installing
channels4world.

## Run a notebook

Use Python 3.11 or later. Install `channels4world`, `huggingface_hub` and `matplotlib`
for the first notebook; install `torch` for the DataLoader notebook. Open the notebook
in Jupyter or VS Code with that Python environment, then run its cells in order.
Data downloads use the dataset's `v1.0.0` release and write into `c4w/` under the
notebook's working directory. The DataLoader example also writes `cache/` and
`splits.json`; reuse that folder to reuse its cache.

The notebooks ship without outputs. For changes, edit the cells in
[`tutorials-src/build_tutorials.py`](../tutorials-src/build_tutorials.py), then regenerate
both the page and notebook. See [tutorial maintenance](../tutorials-src/README.md)
for build, link and execution checks. No data downloads or notebook execution happen
when the library is installed or imported.
