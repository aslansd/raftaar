"""Raftaar — know what your demonstrations will teach, before you train."""

__version__ = "0.3.1"

from .metrics import scan
from .synth import DatasetSpec, FaultSpec, build_dataset, load_dataset

# `provenance` is deliberately not imported here: it is optional, and importing
# it eagerly would be the first step toward the core install growing a
# dependency it does not need. `from raftaar import provenance` works.

__all__ = [
    "scan",
    "DatasetSpec",
    "FaultSpec",
    "build_dataset",
    "load_dataset",
    "__version__",
]
