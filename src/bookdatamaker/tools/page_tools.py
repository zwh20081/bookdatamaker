"""Shared page-navigation tool handlers used by generator and MCP server."""

from __future__ import annotations

from typing import Any, Dict


PAGE_TOOL_NAMES = {
    "get_current_page",
    "jump_to_page",
    "next_page",
    "previous_page",
    "get_page_range",
    "get_page_context",
    "search_text",
}


def execute_page_tool(
    page_manager: Any,
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    default_search_max_results: int,
) -> Dict[str, Any]:
    """Execute a shared page tool and return a structured result.

    Returned dictionary shape:
        {
            "ok": bool,
            "error": str (if !ok),
            "tool_name": str,
            ... tool-specific fields ...
        }
    """
    if tool_name == "get_current_page":
        page_info = page_manager.get_page_info()
        if page_info and "error" not in page_info:
            page_num = page_info["page_number"]
            return {
                "ok": True,
                "tool_name": tool_name,
                "page_info": page_info,
                "current_position": page_num,
                "pages_touched": [page_num],
            }
        return {"ok": False, "tool_name": tool_name, "error": "Could not get current page"}

    if tool_name == "jump_to_page":
        page_num = arguments["page_number"]
        content = page_manager.jump_to_page(page_num)
        if content is None:
            return {"ok": False, "tool_name": tool_name, "error": f"Page {page_num} not found"}

        page_info = page_manager.get_page_info()
        return {
            "ok": True,
            "tool_name": tool_name,
            "content": content,
            "page_info": page_info,
            "current_position": page_num,
            "pages_touched": [page_num],
        }

    if tool_name == "next_page":
        steps = arguments.get("steps", 1)
        content = page_manager.next_page(steps)
        page_info = page_manager.get_page_info()
        if content is None or not page_info:
            return {"ok": False, "tool_name": tool_name, "error": "Could not move to next page"}

        page_num = page_info["page_number"]
        return {
            "ok": True,
            "tool_name": tool_name,
            "steps": steps,
            "content": content,
            "page_info": page_info,
            "current_position": page_num,
            "pages_touched": [page_num],
        }

    if tool_name == "previous_page":
        steps = arguments.get("steps", 1)
        content = page_manager.previous_page(steps)
        page_info = page_manager.get_page_info()
        if content is None or not page_info:
            return {"ok": False, "tool_name": tool_name, "error": "Could not move to previous page"}

        page_num = page_info["page_number"]
        return {
            "ok": True,
            "tool_name": tool_name,
            "steps": steps,
            "content": content,
            "page_info": page_info,
            "current_position": page_num,
            "pages_touched": [page_num],
        }

    if tool_name == "get_page_range":
        req_start = arguments["start_page"]
        req_end = arguments["end_page"]
        if req_end - req_start + 1 > 5:
            req_end = req_start + 4

        pages_data = page_manager.get_page_range(req_start, req_end)
        if not pages_data:
            return {
                "ok": False,
                "tool_name": tool_name,
                "error": f"No pages found in range {req_start}-{req_end}",
            }

        last_page = max(pages_data.keys())
        page_manager.jump_to_page(last_page)
        return {
            "ok": True,
            "tool_name": tool_name,
            "pages_data": pages_data,
            "total_pages": page_manager.get_total_pages(),
            "current_position": last_page,
            "pages_touched": sorted(pages_data.keys()),
            "requested_start": req_start,
            "requested_end": req_end,
        }

    if tool_name == "get_page_context":
        before = arguments.get("before", 1)
        after = arguments.get("after", 1)
        current_page = page_manager.get_current_page_number()
        context = page_manager.get_context(current_page, before, after)

        pages_touched = {
            page
            for page in [context.get("current_page")]
            if page is not None
        }
        pages_touched.update(context.get("previous_pages", {}).keys())
        pages_touched.update(context.get("next_pages", {}).keys())

        return {
            "ok": True,
            "tool_name": tool_name,
            "before": before,
            "after": after,
            "context": context,
            "current_position": current_page,
            "pages_touched": sorted(pages_touched),
        }

    if tool_name == "search_text":
        query = arguments["query"]
        case_sensitive = arguments.get("case_sensitive", False)
        max_results = arguments.get("max_results", default_search_max_results)
        results = page_manager.search_text(
            query,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )

        return {
            "ok": True,
            "tool_name": tool_name,
            "query": query,
            "case_sensitive": case_sensitive,
            "max_results": max_results,
            "results": results,
            "pages_touched": sorted({result["page_number"] for result in results}),
        }

    return {
        "ok": False,
        "tool_name": tool_name,
        "error": f"Unsupported page tool: {tool_name}",
    }
