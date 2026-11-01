"""Preflight completed fixed checkpoints used by terminal held-out branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from td2.config import load_config
from td2.plan import main_work_items, plan_identity


class SourceAuditError(RuntimeError):
    """A requested source checkpoint is absent or not the exact terminal state."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceAuditError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise SourceAuditError(f"JSON value is not an object: {path}")
    return value


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise SourceAuditError(f"{label}: expected {expected!r}, got {actual!r}")


def _source_stream(
    root: Path, *, mode: str, algorithm: str, diversity_n2: int, seed: int
) -> Path:
    if mode == "fixed":
        prefix = "fixed"
    elif mode == "fixed_2b_fresh":
        prefix = "fixed_2b_fresh"
    else:
        raise SourceAuditError(f"unsupported fixed source mode: {mode}")
    return root / prefix / algorithm / f"diversity_{diversity_n2}" / f"seed_{seed}"


def audit_sources(
    data_root: str | Path,
    conditions: Sequence[tuple[int, int, str, str | Path, str | Path]],
) -> int:
    """Validate every DONE marker, pointer, blob, identity, and terminal counter."""

    data_root = Path(data_root).expanduser().resolve()
    expected_keys = {(1, 256), (2, 256), (2, 512), (2, 1024)}
    actual_keys = {(horizon, width) for horizon, width, *_ in conditions}
    if actual_keys != expected_keys or len(conditions) != len(expected_keys):
        raise SourceAuditError(
            f"conditions must be exactly {sorted(expected_keys)}; got {sorted(actual_keys)}"
        )

    audited = 0
    for horizon, width, mode, config_path, source_root in conditions:
        config_path = Path(config_path).expanduser().resolve()
        source_root = Path(source_root).expanduser().resolve()
        config = load_config(config_path)
        _require_equal(
            f"{config_path} GRU width", config.network.recurrent_hidden_dim, width
        )
        _require_equal(f"{horizon}B source mode", mode, "fixed" if horizon == 1 else "fixed_2b_fresh")
        items = main_work_items(config)
        _require_equal(f"{config_path} stream count", len(items), 45)
        config_digest = plan_identity(config)
        if horizon == 1:
            target_updates = config.derived.total_training_updates_per_stream
        else:
            target_updates = 2 * config.derived.total_training_updates_per_stream
        target_optimizer_steps = (
            target_updates
            * config.derived.training_per_distribution.optimizer_steps_per_update
        )

        for item in items:
            _require_equal("source algorithm", item.algorithm, "ppo_cbp")
            seed_root = data_root / f"seed_{item.seed}"
            manifest_path = seed_root / "manifest.json"
            sidecar_path = seed_root / "manifest.sha256"
            if not manifest_path.is_file() or not sidecar_path.is_file():
                raise SourceAuditError(f"seed task bundle is incomplete: {seed_root}")
            manifest_digest = _sha256(manifest_path)
            expected_manifest_digest = sidecar_path.read_text(
                encoding="ascii"
            ).split()[0]
            _require_equal(
                f"manifest checksum {manifest_path}",
                manifest_digest,
                expected_manifest_digest,
            )
            manifest = _read_object(manifest_path)
            _require_equal(
                f"manifest data seed {manifest_path}",
                manifest.get("data_seed"),
                item.seed,
            )

            stream = _source_stream(
                source_root,
                mode=mode,
                algorithm=item.algorithm,
                diversity_n2=item.diversity_n2,
                seed=item.seed,
            )
            done_path = stream / "DONE"
            pointer_path = stream / "resume" / "latest.json"
            if not done_path.is_file() or not pointer_path.is_file():
                raise SourceAuditError(
                    f"source lacks DONE or resume/latest.json: {stream}"
                )
            done = _read_object(done_path)
            pointer = _read_object(pointer_path)
            metadata = pointer.get("metadata")
            if not isinstance(metadata, dict):
                raise SourceAuditError(f"checkpoint metadata is invalid: {pointer_path}")
            _require_equal(f"DONE metadata {stream}", done, metadata)

            blob_name = pointer.get("blob")
            digest = pointer.get("sha256")
            byte_count = pointer.get("bytes")
            if not isinstance(blob_name, str) or Path(blob_name).name != blob_name:
                raise SourceAuditError(f"invalid checkpoint blob name: {pointer_path}")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SourceAuditError(f"invalid checkpoint digest: {pointer_path}")
            blob_path = pointer_path.parent / blob_name
            if not blob_path.is_file():
                raise SourceAuditError(f"checkpoint blob is missing: {blob_path}")
            _require_equal(
                f"checkpoint byte count {blob_path}",
                blob_path.stat().st_size,
                byte_count,
            )

            expected_metadata = {
                "mode": mode,
                "algorithm": item.algorithm,
                "diversity_n": math.isqrt(item.diversity_n2),
                "diversity_n2": item.diversity_n2,
                "seed": item.seed,
                "distribution": 1,
                "source_distribution": 1,
                "update_in_phase": target_updates,
                "global_update": target_updates,
                "optimizer_steps": target_optimizer_steps,
                "schedule_step": target_optimizer_steps,
            }
            for key, expected in expected_metadata.items():
                _require_equal(f"{stream} metadata.{key}", metadata.get(key), expected)
            identity = metadata.get("protocol_identity")
            if not isinstance(identity, dict):
                raise SourceAuditError(f"protocol identity is invalid: {pointer_path}")
            _require_equal(
                f"{stream} config identity",
                identity.get("config_sha256"),
                config_digest,
            )
            _require_equal(
                f"{stream} data identity",
                identity.get("data_manifest_sha256"),
                manifest_digest,
            )
            _require_equal(
                f"{stream} data seed", identity.get("data_seed"), item.seed
            )
            if horizon == 2:
                fresh = identity.get("fixed_2b_fresh")
                if not isinstance(fresh, dict):
                    raise SourceAuditError(
                        f"clean-2B protocol identity is missing: {pointer_path}"
                    )
                _require_equal(
                    f"{stream} clean-2B target",
                    fresh.get("target_updates"),
                    target_updates,
                )
            audited += 1
    return audited


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--condition",
        action="append",
        nargs=5,
        required=True,
        metavar=("HORIZON_B", "GRU_WIDTH", "SOURCE_MODE", "CONFIG", "ROOT"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    conditions: list[tuple[int, int, str, Path, Path]] = []
    for horizon_text, width_text, mode, config, root in args.condition:
        try:
            horizon = int(horizon_text)
            width = int(width_text)
        except ValueError as error:
            raise SystemExit(
                f"invalid horizon/width in --condition: {horizon_text} {width_text}"
            ) from error
        conditions.append((horizon, width, mode, Path(config), Path(root)))
    audited = audit_sources(args.data_root, conditions)
    print(f"validated {audited} exact terminal source checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
