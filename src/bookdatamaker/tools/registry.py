"""Shared tool schema registry for generation and MCP surfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOL_DESCRIPTIONS: Dict[str, str] = {
    "submit_dataset": (
        "Submit a multi-turn conversation to the dataset. "
        "Provide an array of strings alternating between user and assistant messages "
        "(must start with user, end with assistant)."
    ),
    "exit": (
        "Exit the session after completing the required number of dataset submissions. "
        "Use this when you have submitted the target number of Q&A pairs."
    ),
    "get_current_page": "Get the current page content with metadata (page number, total pages, etc.)",
    "next_page": "Move to the next page(s) and return the new page content",
    "previous_page": "Move to the previous page(s) and return the new page content",
    "jump_to_page": "Jump to a specific page by page number",
    "get_page_context": "Get current page with surrounding pages for context",
    "get_page_range": "Get content of multiple pages at once. More efficient than calling next_page repeatedly. Max 5 pages per call.",
    "search_text": "Search for text across the document. Returns matching lines with page numbers.",
    "get_page_access_summary": (
        "Get page submission statistics: which pages have been covered and which are under-explored. "
        "Use to find pages that need more attention."
    ),
    "list_page_images": (
        "List available images for a specific page with absolute file paths, "
        "including cropped figures and the full page image"
    ),
    "get_image": "Get a specific image as base64 data URL. Use list_page_images first to see available images.",
}


def build_openai_tool_defs(
    *,
    include_image_tools: bool,
    include_get_image: bool,
    minimax_extra: Optional[List[dict]] = None,
    search1api_extra: Optional[List[dict]] = None,
) -> List[dict]:
    """Build the OpenAI-style function calling tool definitions.

    This centralizes the shared core tool schemas used by both the generator
    and MCP server so they stay in sync.
    """
    tools: List[dict] = [
        _tool(
            "submit_dataset",
            TOOL_DESCRIPTIONS["submit_dataset"],
            {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Array of message strings alternating user/assistant. "
                            "Must start with user, end with assistant. "
                            "Example: ['user msg 1', 'assistant reply 1', 'user msg 2', 'assistant reply 2']"
                        ),
                    }
                },
                "required": ["messages"],
            },
        ),
        _tool(
            "exit",
            TOOL_DESCRIPTIONS["exit"],
            {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for exiting (e.g., 'Completed 10 dataset entries')",
                    }
                },
                "required": ["reason"],
            },
        ),
        _tool("get_current_page", TOOL_DESCRIPTIONS["get_current_page"], {"type": "object", "properties": {}}),
        _tool(
            "next_page",
            TOOL_DESCRIPTIONS["next_page"],
            {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of pages to move forward",
                        "default": 1,
                    }
                },
            },
        ),
        _tool(
            "previous_page",
            TOOL_DESCRIPTIONS["previous_page"],
            {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of pages to move backward",
                        "default": 1,
                    }
                },
            },
        ),
        _tool(
            "jump_to_page",
            TOOL_DESCRIPTIONS["jump_to_page"],
            {
                "type": "object",
                "properties": {"page_number": {"type": "integer", "description": "Target page number"}},
                "required": ["page_number"],
            },
        ),
        _tool(
            "get_page_context",
            TOOL_DESCRIPTIONS["get_page_context"],
            {
                "type": "object",
                "properties": {
                    "before": {
                        "type": "integer",
                        "description": "Number of pages before current",
                        "default": 1,
                    },
                    "after": {
                        "type": "integer",
                        "description": "Number of pages after current",
                        "default": 1,
                    },
                },
            },
        ),
        _tool(
            "get_page_range",
            TOOL_DESCRIPTIONS["get_page_range"],
            {
                "type": "object",
                "properties": {
                    "start_page": {"type": "integer", "description": "Start page number (inclusive)"},
                    "end_page": {
                        "type": "integer",
                        "description": "End page number (inclusive). Max 5 pages from start.",
                    },
                },
                "required": ["start_page", "end_page"],
            },
        ),
        _tool(
            "search_text",
            TOOL_DESCRIPTIONS["search_text"],
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for"},
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case-sensitive search",
                        "default": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        _tool(
            "get_page_access_summary",
            TOOL_DESCRIPTIONS["get_page_access_summary"],
            {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of least-submitted pages to highlight",
                        "default": 5,
                    }
                },
            },
        ),
    ]

    if include_image_tools:
        tools.append(
            _tool(
                "list_page_images",
                TOOL_DESCRIPTIONS["list_page_images"],
                {
                    "type": "object",
                    "properties": {
                        "page_number": {
                            "type": "integer",
                            "description": "Page number to list images for",
                        }
                    },
                    "required": ["page_number"],
                },
            )
        )

        if include_get_image:
            tools.append(
                _tool(
                    "get_image",
                    TOOL_DESCRIPTIONS["get_image"],
                    {
                        "type": "object",
                        "properties": {
                            "page_number": {"type": "integer", "description": "Page number"},
                            "image_name": {
                                "type": "string",
                                "description": "Image filename (e.g., 'images/0.jpg' for cropped image, 'page_001.png' for full page)",
                            },
                        },
                        "required": ["page_number", "image_name"],
                    },
                )
            )

    if minimax_extra:
        tools.extend(minimax_extra)
    if search1api_extra:
        tools.extend(search1api_extra)

    return tools


def build_mcp_tool_specs(include_page_manager_tools: bool, include_image_tools: bool) -> List[Dict[str, Any]]:
    """Build tool specs in a neutral format for MCP server Tool creation."""
    specs: List[Dict[str, Any]] = [
        {
            "name": "submit_dataset",
            "description": TOOL_DESCRIPTIONS["submit_dataset"],
            "inputSchema": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of message strings alternating user/assistant. Example: ['user message 1', 'assistant reply 1', 'user message 2', 'assistant reply 2']",
                    }
                },
                "required": ["messages"],
            },
        },
        {
            "name": "exit",
            "description": TOOL_DESCRIPTIONS["exit"],
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for exiting (e.g., 'Completed 10 dataset entries')",
                    }
                },
                "required": ["reason"],
            },
        },
    ]

    if include_page_manager_tools:
        specs.extend(
            [
                {
                    "name": "get_current_page",
                    "description": TOOL_DESCRIPTIONS["get_current_page"],
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "next_page",
                    "description": TOOL_DESCRIPTIONS["next_page"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "integer",
                                "description": "Number of pages to move forward",
                                "default": 1,
                            }
                        },
                    },
                },
                {
                    "name": "previous_page",
                    "description": TOOL_DESCRIPTIONS["previous_page"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "integer",
                                "description": "Number of pages to move backward",
                                "default": 1,
                            }
                        },
                    },
                },
                {
                    "name": "jump_to_page",
                    "description": TOOL_DESCRIPTIONS["jump_to_page"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "page_number": {
                                "type": "integer",
                                "description": "Target page number",
                            }
                        },
                        "required": ["page_number"],
                    },
                },
                {
                    "name": "get_page_context",
                    "description": TOOL_DESCRIPTIONS["get_page_context"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "before": {
                                "type": "integer",
                                "description": "Number of pages before current",
                                "default": 1,
                            },
                            "after": {
                                "type": "integer",
                                "description": "Number of pages after current",
                                "default": 1,
                            },
                        },
                    },
                },
                {
                    "name": "get_page_range",
                    "description": TOOL_DESCRIPTIONS["get_page_range"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "start_page": {
                                "type": "integer",
                                "description": "Start page number (inclusive)",
                            },
                            "end_page": {
                                "type": "integer",
                                "description": "End page number (inclusive). Max 5 pages from start.",
                            },
                        },
                        "required": ["start_page", "end_page"],
                    },
                },
                {
                    "name": "get_page_access_summary",
                    "description": TOOL_DESCRIPTIONS["get_page_access_summary"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of least-submitted pages to highlight",
                                "default": 5,
                            }
                        },
                    },
                },
                {
                    "name": "search_text",
                    "description": TOOL_DESCRIPTIONS["search_text"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Text to search for"},
                            "case_sensitive": {
                                "type": "boolean",
                                "description": "Case-sensitive search",
                                "default": False,
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 100,
                            },
                        },
                        "required": ["query"],
                    },
                },
            ]
        )

    if include_image_tools:
        specs.extend(
            [
                {
                    "name": "list_page_images",
                    "description": TOOL_DESCRIPTIONS["list_page_images"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "page_number": {
                                "type": "integer",
                                "description": "Page number to list images for",
                            }
                        },
                        "required": ["page_number"],
                    },
                },
                {
                    "name": "get_image",
                    "description": TOOL_DESCRIPTIONS["get_image"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "page_number": {"type": "integer", "description": "Page number"},
                            "image_name": {
                                "type": "string",
                                "description": "Image filename (e.g., 'images/0.jpg' for cropped image, 'page_001.png' for full page)",
                            },
                        },
                        "required": ["page_number", "image_name"],
                    },
                },
            ]
        )

    return specs
