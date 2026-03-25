"""Prompt construction helpers for parallel dataset generation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def _load_prompt_template(filename: str) -> str:
    """Load a prompt template from ``src/bookdatamaker/prompts``."""
    return (PROMPT_DIR / filename).read_text(encoding="utf-8")


def _join_section(content: str) -> str:
    """Prefix a section with two newlines when content is non-empty."""
    return f"\n\n{content}" if content else ""


def create_system_prompt(
    *,
    start_page: int,
    thread_id: int,
    total_pages: int,
    datasets_per_thread: int,
    extracted_dir: Optional[Path],
    has_minimax_mcp: bool,
    has_search1api_mcp: bool,
    custom_prompt: Optional[str],
) -> str:
    """Build the system prompt used by each generation thread.

    This extracts prompt-building logic out of ``ParallelDatasetGenerator`` so
    content can later be migrated to templates with minimal behavior change.
    """
    image_tools_line = ""
    if extracted_dir:
        image_tools_line = "\nImages: list_page_images(page_number)"
        if not has_minimax_mcp:
            image_tools_line += ", get_image(page_number, image_name)"

    minimax_tool_lines = ""
    if has_minimax_mcp:
        minimax_tool_lines = (
            "\nInternet search: minimax_web_search — search the REAL INTERNET for external information (NOT the document)"
            "\nImage analysis: minimax_understand_image(image_url) — analyze an image file"
        )

    search1api_tool_lines = ""
    if has_search1api_mcp:
        search1api_tool_lines = (
            "\nInternet search: search1api_search(query) — search the REAL INTERNET via Search1API"
            "\nNews search: search1api_news(query) — search for latest news articles"
            "\nWeb crawl: search1api_crawl(url) — extract content from a URL"
        )

    search_guidance = ""
    if has_search1api_mcp and has_minimax_mcp:
        search_guidance = _load_prompt_template("combined_search_guidance.md")
    elif has_minimax_mcp:
        search_guidance = _load_prompt_template("minimax_guidance.md")
    elif has_search1api_mcp:
        search_guidance = _load_prompt_template("search1api_guidance.md")

    image_workflow = ""
    if extracted_dir and has_minimax_mcp:
        image_workflow = _load_prompt_template("image_workflow.md")

    custom_prompt_section = ""
    if custom_prompt:
        custom_prompt_section = f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_prompt}"

    template = _load_prompt_template("base_system_prompt.md")
    rendered = template
    replacements = {
        "{{START_PAGE}}": str(start_page),
        "{{THREAD_ID}}": str(thread_id),
        "{{TOTAL_PAGES}}": str(total_pages),
        "{{TARGET_COUNT}}": str(datasets_per_thread),
        "{{IMAGE_TOOLS_LINE}}": image_tools_line,
        "{{MINIMAX_TOOL_LINES}}": minimax_tool_lines,
        "{{SEARCH1API_TOOL_LINES}}": search1api_tool_lines,
        "{{SEARCH_GUIDANCE_SECTION}}": _join_section(search_guidance),
        "{{IMAGE_WORKFLOW_SECTION}}": _join_section(image_workflow),
        "{{CUSTOM_PROMPT_SECTION}}": custom_prompt_section,
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered
