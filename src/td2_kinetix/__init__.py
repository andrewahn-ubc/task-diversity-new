"""Reproducible Kinetix continual-learning experiment support.

The package is deliberately independent of :mod:`td2`: Kinetix v0.1.0 uses
Python 3.11, JAX 0.4.35, and NumPy 1.26, while the Banyan experiment has a
newer, incompatible runtime.  Imports which require Kinetix or JAX therefore
live inside the runtime entry points rather than at module import time.
"""

from .config import KINETIX_COMMIT, PROTOCOL, PROTOCOL_ID

__all__ = ["KINETIX_COMMIT", "PROTOCOL", "PROTOCOL_ID"]
