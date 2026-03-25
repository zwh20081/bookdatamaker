"""Shared helpers for dataset submission and related validation."""

from __future__ import annotations

from typing import Any, Optional


def validate_dataset_messages(messages: list[str]) -> Optional[str]:
    """Validate dataset submission message array.

    Returns:
        ``None`` when valid; otherwise an error message string.
    """
    if not messages:
        return "messages array cannot be empty"
    if len(messages) < 2:
        return "messages must contain at least one user-assistant pair (minimum 2 messages)"
    if len(messages) % 2 != 0:
        return "messages must have even length (alternating user-assistant pairs)"
    return None


def compute_remaining_submissions(submitted_count: int, target_count: int) -> int:
    """Compute remaining submissions with lower bound zero."""
    remaining = target_count - submitted_count
    return remaining if remaining > 0 else 0


def build_page_access_summary(
    *,
    counts: dict[int, int],
    page_numbers: list[int],
    last_submission_page: Any,
    limit: int = 5,
) -> dict[str, Any]:
    """Build a unified page access summary payload.

    Args:
        counts: Mapping of page -> submission count.
        page_numbers: All known page numbers in the current document.
        last_submission_page: Last submitted page from session metadata/state.
        limit: Number of least/most submitted pages to include.

    Returns:
        Summary dictionary used by both generator and MCP server.
    """
    normalized_counts: dict[int, int] = {}

    for page_num in page_numbers:
        try:
            normalized_counts[int(page_num)] = 0
        except (TypeError, ValueError):
            continue

    for page_num, count in counts.items():
        try:
            normalized_counts[int(page_num)] = int(count)
        except (TypeError, ValueError):
            continue

    total_pages = len(normalized_counts)
    total_submissions = sum(normalized_counts.values())
    average = total_submissions / total_pages if total_pages else 0.0

    safe_limit = max(0, int(limit))
    least_sorted = sorted(normalized_counts.items(), key=lambda item: (item[1], item[0]))
    most_sorted = sorted(normalized_counts.items(), key=lambda item: (-item[1], item[0]))

    least_pages = [page for page, _ in least_sorted[:safe_limit]]
    most_pages = [page for page, _ in most_sorted[:safe_limit]]

    summary: dict[str, Any] = {
        "total_pages": total_pages,
        "total_submissions": total_submissions,
        "average_submissions_per_page": average,
        "page_submission_counts": normalized_counts,
        "least_submitted_pages": least_pages,
        "most_submitted_pages": most_pages,
        "recommended_pages": [
            page for page in least_pages if normalized_counts.get(page, 0) <= average
        ],
    }

    if last_submission_page is not None:
        try:
            summary["last_submission_page"] = int(last_submission_page)
        except (TypeError, ValueError):
            summary["last_submission_page"] = last_submission_page

    return summary
