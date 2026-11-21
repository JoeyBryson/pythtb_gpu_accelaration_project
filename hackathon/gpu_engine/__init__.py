"""GPU engine for PythTB's topology pipeline (written at the hackathon, 21-22 Nov 2026).

Sits alongside PythTB rather than modifying it. Modules:

- backend:   device / precision selection and GPU-correct timing
- reference: the slow, obviously-correct CPU baseline (plain PythTB) that every
             optimised version is compared against
"""

from .backend import Backend, get_backend, sync, timed

__all__ = ["Backend", "get_backend", "sync", "timed"]
