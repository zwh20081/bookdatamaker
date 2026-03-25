"""Tests for submission helpers."""

from bookdatamaker.tools.submission_tools import (
    build_page_access_summary,
    compute_remaining_submissions,
    validate_dataset_messages,
)


def test_validate_dataset_messages_success() -> None:
    assert validate_dataset_messages(["q", "a"]) is None
    assert validate_dataset_messages(["q1", "a1", "q2", "a2"]) is None


def test_validate_dataset_messages_errors() -> None:
    assert "empty" in validate_dataset_messages([])
    assert "minimum 2" in validate_dataset_messages(["q"])
    assert "even length" in validate_dataset_messages(["q", "a", "q"])


def test_compute_remaining_submissions_clamps_at_zero() -> None:
    assert compute_remaining_submissions(0, 10) == 10
    assert compute_remaining_submissions(8, 10) == 2
    assert compute_remaining_submissions(10, 10) == 0
    assert compute_remaining_submissions(12, 10) == 0


def test_build_page_access_summary_merges_known_pages_and_counts() -> None:
    summary = build_page_access_summary(
        counts={2: 3, 5: 1},
        page_numbers=[1, 2, 3, 4],
        last_submission_page="2",
        limit=3,
    )

    assert summary["total_pages"] == 5
    assert summary["total_submissions"] == 4
    assert summary["page_submission_counts"][1] == 0
    assert summary["page_submission_counts"][2] == 3
    assert summary["page_submission_counts"][5] == 1
    assert summary["least_submitted_pages"][:2] == [1, 3]
    assert summary["last_submission_page"] == 2


def test_build_page_access_summary_clamps_negative_limit_to_zero() -> None:
    summary = build_page_access_summary(
        counts={1: 2},
        page_numbers=[1, 2],
        last_submission_page=None,
        limit=-5,
    )

    assert summary["least_submitted_pages"] == []
    assert summary["most_submitted_pages"] == []
    assert summary["recommended_pages"] == []
