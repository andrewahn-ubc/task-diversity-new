from __future__ import annotations

import pytest

from scripts.check_runtime import EXPECTED, _matches_expected_version


@pytest.mark.parametrize("expected", EXPECTED.values())
def test_compute_canada_local_builds_match_pinned_public_versions(
    expected: str,
) -> None:
    assert _matches_expected_version(expected, expected)
    assert _matches_expected_version(f"{expected}+computecanada", expected)
    assert _matches_expected_version(f"{expected}+computecanada.1", expected)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("0.2.9+computecanada", "0.2.8"),
        ("0.2.8.post1+computecanada", "0.2.8"),
        ("0.2.8rc1+computecanada", "0.2.8"),
        ("0.2.8.dev1+computecanada", "0.2.8"),
        ("1!0.2.8+computecanada", "0.2.8"),
        ("not-a-version", "0.2.8"),
    ],
)
def test_runtime_version_check_rejects_other_or_invalid_public_versions(
    actual: str,
    expected: str,
) -> None:
    assert not _matches_expected_version(actual, expected)
