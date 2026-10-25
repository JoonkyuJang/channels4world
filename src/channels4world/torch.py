"""Thin PyTorch adapter.

``RecordDataset`` is a map-style dataset (``__len__`` / ``__getitem__``), which is all
``torch.utils.data.DataLoader`` needs, so this module does not import torch itself.

    DataLoader(ds, batch_size=None)                 # one record per item, untouched
    DataLoader(ds, batch_size=8, collate_fn=list)   # ragged records as plain lists
    DataLoader(RecordDataset(sel, transform=to_tensors), batch_size=8)  # fixed-size tensors

PyTorch's default collate function cannot batch ``Record`` objects (it raises a TypeError),
so use one of the forms above. Stored arrays are read-only; copy them before
``torch.from_numpy`` (transforms already return new writable arrays).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .dataset import BaseStation, Dataset, Selection


class RecordDataset:
    """Channel records of a selection, indexable for PyTorch's ``DataLoader``.

    The record list is built once from metadata (no payload reads). Each worker process
    reopens the dataset lazily, so ``num_workers > 0`` works without sharing open files.
    """

    def __init__(self, selection: Selection, transform: Callable[[Any], Any] | None = None,
                 verify: str = "basic") -> None:
        self._dataset: Dataset = selection.dataset
        self._detail = "detail" in selection.components
        self._keys = [(scene.id, k.hz, k.bs) for scene in selection.scenes() for k in selection._keys(scene)]
        self.transform = transform
        self.verify = verify

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int) -> Any:
        scene_id, hz, bs = self._keys[index]
        rec = BaseStation(self._dataset.scene(scene_id), hz, bs).read(detail=self._detail, verify=self.verify)
        return rec if self.transform is None else self.transform(rec)
