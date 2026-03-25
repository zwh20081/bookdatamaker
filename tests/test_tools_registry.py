"""Tests for shared tool schema registry."""

from bookdatamaker.tools.registry import (
    build_mcp_tool_specs,
    build_openai_tool_defs,
)


def test_openai_tool_defs_include_core_tools() -> None:
    tools = build_openai_tool_defs(
        include_image_tools=False,
        include_get_image=False,
        minimax_extra=None,
        search1api_extra=None,
    )

    names = [tool["function"]["name"] for tool in tools]
    assert "submit_dataset" in names
    assert "exit" in names
    assert "get_current_page" in names
    assert "next_page" in names
    assert "previous_page" in names
    assert "jump_to_page" in names
    assert "get_page_context" in names
    assert "get_page_range" in names
    assert "search_text" in names
    assert "get_page_access_summary" in names


def test_openai_tool_defs_image_flags_control_get_image() -> None:
    with_image = build_openai_tool_defs(
        include_image_tools=True,
        include_get_image=True,
        minimax_extra=None,
        search1api_extra=None,
    )
    without_get_image = build_openai_tool_defs(
        include_image_tools=True,
        include_get_image=False,
        minimax_extra=None,
        search1api_extra=None,
    )

    names_with = [tool["function"]["name"] for tool in with_image]
    names_without = [tool["function"]["name"] for tool in without_get_image]

    assert "list_page_images" in names_with
    assert "get_image" in names_with
    assert "list_page_images" in names_without
    assert "get_image" not in names_without


def test_openai_tool_defs_append_external_tools() -> None:
    minimax_extra = [
        {
            "type": "function",
            "function": {
                "name": "minimax_web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    tools = build_openai_tool_defs(
        include_image_tools=False,
        include_get_image=False,
        minimax_extra=minimax_extra,
        search1api_extra=None,
    )

    names = [tool["function"]["name"] for tool in tools]
    assert "minimax_web_search" in names


def test_mcp_tool_specs_core_modes() -> None:
    base_specs = build_mcp_tool_specs(
        include_page_manager_tools=False,
        include_image_tools=False,
    )
    page_specs = build_mcp_tool_specs(
        include_page_manager_tools=True,
        include_image_tools=False,
    )
    image_specs = build_mcp_tool_specs(
        include_page_manager_tools=False,
        include_image_tools=True,
    )

    base_names = [spec["name"] for spec in base_specs]
    page_names = [spec["name"] for spec in page_specs]
    image_names = [spec["name"] for spec in image_specs]

    assert base_names == ["submit_dataset", "exit"]
    assert "get_current_page" in page_names
    assert "search_text" in page_names
    assert "get_page_access_summary" in page_names
    assert "list_page_images" in image_names
    assert "get_image" in image_names


def test_openai_search_text_exposes_max_results() -> None:
    tools = build_openai_tool_defs(
        include_image_tools=False,
        include_get_image=False,
        minimax_extra=None,
        search1api_extra=None,
    )

    search_tool = next(tool for tool in tools if tool["function"]["name"] == "search_text")
    properties = search_tool["function"]["parameters"]["properties"]
    assert "max_results" in properties
    assert properties["max_results"]["default"] == 20


def test_shared_tool_descriptions_align_between_openai_and_mcp() -> None:
    openai_tools = build_openai_tool_defs(
        include_image_tools=True,
        include_get_image=True,
        minimax_extra=None,
        search1api_extra=None,
    )
    mcp_specs = build_mcp_tool_specs(
        include_page_manager_tools=True,
        include_image_tools=True,
    )

    openai_desc = {tool["function"]["name"]: tool["function"]["description"] for tool in openai_tools}
    mcp_desc = {spec["name"]: spec["description"] for spec in mcp_specs}

    for name in {
        "submit_dataset",
        "exit",
        "get_current_page",
        "next_page",
        "previous_page",
        "search_text",
        "get_page_access_summary",
        "list_page_images",
        "get_image",
    }:
        assert openai_desc[name] == mcp_desc[name]
