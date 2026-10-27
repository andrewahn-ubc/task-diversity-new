from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    ROOT / "scripts" / "submit_all.sh",
    ROOT / "scripts" / "submit_fixed_2b.sh",
    ROOT / "scripts" / "submit_core_aux_2b.sh",
)


def test_submission_scripts_only_name_the_krallen_allocation() -> None:
    submission_files = tuple((ROOT / "scripts").rglob("*.sh")) + tuple(
        (ROOT / "slurm").rglob("*.sbatch")
    )
    named_allocations: set[str] = set()
    for path in submission_files:
        named_allocations.update(
            re.findall(r"\bdef-[A-Za-z0-9_.-]+", path.read_text(encoding="utf-8"))
        )

    assert named_allocations == {"def-krallen"}


def test_launchers_do_not_allow_account_overrides() -> None:
    for path in LAUNCHERS:
        source = path.read_text(encoding="utf-8")
        assert source.count("\nACCOUNT=def-krallen\n") == 1
        assert "--account-a" not in source
        assert "--account-b" not in source
        assert "ACCOUNT_A" not in source
        assert "ACCOUNT_B" not in source
