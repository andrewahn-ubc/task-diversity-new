#!/usr/bin/env python3
"""Fail-fast validation of the immutable cluster Python/JAX runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


EXPECTED = {
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
    "flax": "0.12.8",
    "optax": "0.2.8",
    "chex": "0.1.92",
    "numpy": "2.5.1",
    "PyYAML": "6.0.3",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"expected Python 3.13, got {sys.version.split()[0]}")

    mismatches = []
    for package, expected in EXPECTED.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            mismatches.append(f"{package}={actual} (expected {expected})")
    if mismatches:
        raise RuntimeError("runtime version mismatch: " + ", ".join(mismatches))

    # Import the exact Banyan modules used here so its declared Python >=3.13
    # compatibility is exercised during setup rather than first discovered by
    # a production job.
    import banyan_grid  # noqa: F401
    import td2  # noqa: F401
    from banyan_grid.utils.banyan import get_map  # noqa: F401

    import jax
    import jax.numpy as jnp

    backend = jax.default_backend()
    devices = jax.devices()
    if args.require_gpu and backend != "gpu":
        raise RuntimeError(f"JAX selected {backend!r}, expected 'gpu'; devices={devices!r}")
    if args.require_gpu and not devices:
        raise RuntimeError("JAX reported no GPU devices")

    # Force both compilation and execution; device discovery alone does not
    # catch every CUDA-library mismatch.
    x = jnp.arange(256, dtype=jnp.float32).reshape(16, 16)
    value = jax.device_get(jnp.sum(x @ x.T))
    if not bool(jnp.isfinite(value)):
        raise RuntimeError("GPU smoke calculation produced a non-finite value")

    print(f"python={sys.version.split()[0]} jax={jax.__version__} backend={backend}")
    for device in devices:
        print(f"device={device}")
    print(f"smoke_value={float(value):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
