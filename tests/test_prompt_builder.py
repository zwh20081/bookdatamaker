"""Tests for prompt builder."""

from bookdatamaker.llm.prompt_builder import create_system_prompt


def test_prompt_builder_includes_core_and_custom_prompt() -> None:
    prompt = create_system_prompt(
        start_page=10,
        thread_id=2,
        total_pages=300,
        datasets_per_thread=20,
        extracted_dir=None,
        has_minimax_mcp=False,
        has_search1api_mcp=False,
        custom_prompt="Please focus on domain terms.",
    )

    assert "Starting page: 10 | Total pages: 300 | Target: 20 conversations | Thread: 2" in prompt
    assert "submit_dataset(messages), exit(reason)" in prompt
    assert "Please focus on domain terms." in prompt


def test_prompt_builder_toggles_image_and_search_sections() -> None:
    prompt_with_all = create_system_prompt(
        start_page=1,
        thread_id=0,
        total_pages=10,
        datasets_per_thread=3,
        extracted_dir="dummy",  # truthy is enough
        has_minimax_mcp=True,
        has_search1api_mcp=True,
        custom_prompt=None,
    )

    prompt_with_search1_only = create_system_prompt(
        start_page=1,
        thread_id=0,
        total_pages=10,
        datasets_per_thread=3,
        extracted_dir="dummy",  # truthy is enough
        has_minimax_mcp=False,
        has_search1api_mcp=True,
        custom_prompt=None,
    )

    assert "Images: list_page_images(page_number)" in prompt_with_all
    assert "get_image(page_number, image_name)" not in prompt_with_all
    assert "minimax_web_search" in prompt_with_all
    assert "search1api_search" in prompt_with_all

    assert "get_image(page_number, image_name)" in prompt_with_search1_only
    assert "minimax_web_search" not in prompt_with_search1_only
    assert "search1api_search" in prompt_with_search1_only
