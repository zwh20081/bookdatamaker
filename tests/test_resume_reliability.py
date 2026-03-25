"""Tests for resume-related message safety helpers.

These tests target helper behavior that supports checkpoint resume stability,
without claiming to cover the full end-to-end resume workflow.
"""

from bookdatamaker.llm.message_utils import safe_prune_messages
from bookdatamaker.tools.submission_tools import compute_remaining_submissions


def test_resume_message_history_without_system_gets_pruned_safely() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        {"role": "user", "content": "continue"},
    ]

    pruned = safe_prune_messages(messages, keep_last=2)

    # No crash and tool pair remains structurally consistent.
    assert isinstance(pruned, list)
    assert any(msg.get("role") == "tool" for msg in pruned)


def test_resume_remaining_computation_is_non_negative() -> None:
    # Resume flows rely on remaining count to build continuation hints;
    # this should never become negative even with inconsistent checkpoints.
    assert compute_remaining_submissions(5, 3) == 0
