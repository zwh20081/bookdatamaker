"""Tests for shared page tool handlers."""

from bookdatamaker.tools.page_tools import PAGE_TOOL_NAMES, execute_page_tool
from bookdatamaker.utils.page_manager import PageManager


def _sample_page_manager() -> PageManager:
    return PageManager(
        {
            1: "alpha topic\nline2",
            2: "beta topic\nline2",
            3: "gamma",
            4: "delta",
            5: "epsilon",
            6: "zeta",
        }
    )


def test_page_tool_name_set_contains_core_tools() -> None:
    assert {
        "get_current_page",
        "jump_to_page",
        "next_page",
        "previous_page",
        "get_page_range",
        "get_page_context",
        "search_text",
    } <= PAGE_TOOL_NAMES


def test_execute_get_current_page_success() -> None:
    pm = _sample_page_manager()

    result = execute_page_tool(pm, "get_current_page", {}, default_search_max_results=100)

    assert result["ok"] is True
    assert result["page_info"]["page_number"] == 1
    assert result["current_position"] == 1


def test_execute_jump_to_page_not_found() -> None:
    pm = _sample_page_manager()

    result = execute_page_tool(
        pm,
        "jump_to_page",
        {"page_number": 999},
        default_search_max_results=100,
    )

    assert result["ok"] is False
    assert "not found" in result["error"].lower()


def test_execute_get_page_range_clamps_to_five_pages() -> None:
    pm = _sample_page_manager()

    result = execute_page_tool(
        pm,
        "get_page_range",
        {"start_page": 1, "end_page": 10},
        default_search_max_results=100,
    )

    assert result["ok"] is True
    assert sorted(result["pages_data"].keys()) == [1, 2, 3, 4, 5]
    assert result["current_position"] == 5


def test_execute_search_text_respects_default_max_results() -> None:
    pm = _sample_page_manager()

    result = execute_page_tool(
        pm,
        "search_text",
        {"query": "topic"},
        default_search_max_results=1,
    )

    assert result["ok"] is True
    assert len(result["results"]) == 1


def test_execute_unsupported_tool_returns_error() -> None:
    pm = _sample_page_manager()

    result = execute_page_tool(pm, "unknown_tool", {}, default_search_max_results=100)

    assert result["ok"] is False
    assert "unsupported" in result["error"].lower()
