"""MCP server for paragraph and line/column navigation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from bookdatamaker.tools.image_tools import get_image_data_url, list_page_images
from bookdatamaker.tools.page_tools import PAGE_TOOL_NAMES, execute_page_tool
from bookdatamaker.tools.registry import build_mcp_tool_specs
from bookdatamaker.tools.submission_tools import (
    build_page_access_summary,
    validate_dataset_messages,
)

if TYPE_CHECKING:
    from ..utils.page_manager import PageManager
else:
    PageManager = Any


class ParagraphNavigator:
    """Navigate through paragraphs with context window."""

    def __init__(self, paragraphs: List[str]) -> None:
        """Initialize navigator with paragraphs.

        Args:
            paragraphs: List of text paragraphs
        """
        self.paragraphs = paragraphs
        self.current_index = 0

    def get_current(self) -> str:
        """Get current paragraph."""
        if 0 <= self.current_index < len(self.paragraphs):
            return self.paragraphs[self.current_index]
        return ""

    def move_forward(self, steps: int = 1) -> str:
        """Move forward by N paragraphs.

        Args:
            steps: Number of paragraphs to move forward

        Returns:
            The paragraph at the new position
        """
        self.current_index = min(self.current_index + steps, len(self.paragraphs) - 1)
        return self.get_current()

    def move_backward(self, steps: int = 1) -> str:
        """Move backward by N paragraphs.

        Args:
            steps: Number of paragraphs to move backward

        Returns:
            The paragraph at the new position
        """
        self.current_index = max(self.current_index - steps, 0)
        return self.get_current()

    def get_context(self, before: int = 1, after: int = 1) -> Dict[str, Any]:
        """Get current paragraph with surrounding context.

        Args:
            before: Number of paragraphs before current
            after: Number of paragraphs after current

        Returns:
            Dictionary with current, previous, and next paragraphs
        """
        start = max(0, self.current_index - before)
        end = min(len(self.paragraphs), self.current_index + after + 1)

        return {
            "current_index": self.current_index,
            "total_paragraphs": len(self.paragraphs),
            "current": self.get_current(),
            "previous": self.paragraphs[start : self.current_index],
            "next": self.paragraphs[self.current_index + 1 : end],
        }

    def jump_to(self, index: int) -> str:
        """Jump to specific paragraph index.

        Args:
            index: Target paragraph index

        Returns:
            The paragraph at the target position
        """
        self.current_index = max(0, min(index, len(self.paragraphs) - 1))
        return self.get_current()


class MCPServer:
    """MCP server for paragraph and line/column navigation tools."""

    def __init__(
        self, 
        paragraphs: Optional[List[str]] = None,
        page_manager: Optional["PageManager"] = None,
        db_path: Optional[str] = None,
        extracted_dir: Optional[Path] = None
    ) -> None:
        """Initialize MCP server.

        Args:
            paragraphs: List of text paragraphs (optional, for backward compatibility)
            page_manager: PageManager instance for advanced navigation (optional)
            db_path: Path to SQLite database for dataset storage (optional)
            extracted_dir: Path to extracted directory containing page_XXX/ subdirs (optional, for image access)
        """
        self.navigator = ParagraphNavigator(paragraphs) if paragraphs else None
        self.page_manager = page_manager
        self.db_path = db_path
        self.extracted_dir = Path(extracted_dir) if extracted_dir else None
        self.dataset_manager = None
        self.should_exit = False
        self.page_submission_counts: Dict[int, int] = {}
        self.last_accessed_page: Optional[int] = None
        
        # Initialize dataset manager if db_path provided
        if self.db_path:
            try:
                from ..dataset import DatasetManager
                self.dataset_manager = DatasetManager(self.db_path)
            except ImportError:
                print("Warning: Could not import DatasetManager")

        # Initialize submission counters
        self._refresh_page_submission_state()
        
        self.server = Server("document-navigator")
        self._setup_tools()

    def _format_response(
        self, 
        content: str, 
        line_number: Optional[int] = None,
        paragraph_number: Optional[int] = None,
        page_number: Optional[int] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format response with unified structure.

        Args:
            content: Main content text
            line_number: Global line number (0-indexed)
            paragraph_number: Paragraph number (0-indexed)
            page_number: Page number
            additional_info: Additional information to include

        Returns:
            Formatted response dictionary
        """
        response = {
            "line_number": line_number,
            "paragraph_number": paragraph_number,
            "page_number": page_number,
            "content": content,
        }
        
        if additional_info:
            response.update(additional_info)

        if page_number is not None and self.page_manager:
            response["page_submission_count"] = self.page_submission_counts.get(page_number, 0)
        
        return response

    def _record_page_access(self, page_numbers: List[Optional[int]]) -> None:
        """Record latest accessed page without mutating submission counts."""
        for page_num in page_numbers:
            if page_num is not None:
                self.last_accessed_page = page_num
                break

    def _refresh_page_submission_state(self) -> None:
        """Reload submission counts from dataset manager."""
        if not self.dataset_manager:
            return

        if self.page_manager:
            for page_num in getattr(self.page_manager, "page_numbers", []):
                self.page_submission_counts.setdefault(page_num, 0)

        counts = self.dataset_manager.get_page_submission_counts()
        self.page_submission_counts.update(counts)

        last_page_value = self.dataset_manager.get_session_metadata("last_submission_page")
        if last_page_value is not None:
            try:
                self.last_accessed_page = int(last_page_value)
            except ValueError:
                self.last_accessed_page = None

    @staticmethod
    def _append_registry_tools(
        tools: List[Tool],
        *,
        include_page_manager_tools: bool,
        include_image_tools: bool,
        skip_names: Optional[set[str]] = None,
    ) -> None:
        """Append registry-built tools with optional skip list."""
        skip_names = skip_names or set()
        for spec in build_mcp_tool_specs(
            include_page_manager_tools=include_page_manager_tools,
            include_image_tools=include_image_tools,
        ):
            if spec["name"] in skip_names:
                continue
            tools.append(
                Tool(
                    name=spec["name"],
                    description=spec["description"],
                    inputSchema=spec["inputSchema"],
                )
            )

    def _setup_tools(self) -> None:
        """Set up MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            tools = [
                Tool(
                    name="get_current_paragraph",
                    description="Get the current paragraph with line/paragraph/page numbers",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="move_forward",
                    description="Move forward by N paragraphs and return the new position with metadata",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "integer",
                                "description": "Number of paragraphs to move forward",
                                "default": 1,
                            }
                        },
                    },
                ),
                Tool(
                    name="move_backward",
                    description="Move backward by N paragraphs and return the new position with metadata",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "integer",
                                "description": "Number of paragraphs to move backward",
                                "default": 1,
                            }
                        },
                    },
                ),
                Tool(
                    name="get_context",
                    description="Get current paragraph with surrounding context and full metadata",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "before": {
                                "type": "integer",
                                "description": "Number of paragraphs before current",
                                "default": 1,
                            },
                            "after": {
                                "type": "integer",
                                "description": "Number of paragraphs after current",
                                "default": 1,
                            },
                        },
                    },
                ),
                Tool(
                    name="jump_to",
                    description="Jump to a specific paragraph by index and return content with metadata",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "Target paragraph index (0-based)",
                            }
                        },
                        "required": ["index"],
                    },
                ),
            ]

            # Shared core dataset tools
            self._append_registry_tools(
                tools,
                include_page_manager_tools=False,
                include_image_tools=False,
            )

            # Add page-based navigation tools if page_manager is available
            if self.page_manager:
                tools.extend([
                    Tool(
                        name="get_line",
                        description="Get content of a specific line by line number",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "line_number": {
                                    "type": "integer",
                                    "description": "Line number (0-indexed)",
                                }
                            },
                            "required": ["line_number"],
                        },
                    ),
                    Tool(
                        name="get_line_range",
                        description="Get content of a range of lines",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "start_line": {
                                    "type": "integer",
                                    "description": "Start line number (0-indexed, inclusive)",
                                },
                                "end_line": {
                                    "type": "integer",
                                    "description": "End line number (0-indexed, inclusive)",
                                },
                            },
                            "required": ["start_line", "end_line"],
                        },
                    ),
                    Tool(
                        name="get_line_with_context",
                        description="Get a line with surrounding context lines",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "line_number": {
                                    "type": "integer",
                                    "description": "Target line number (0-indexed)",
                                },
                                "before": {
                                    "type": "integer",
                                    "description": "Number of lines before",
                                    "default": 3,
                                },
                                "after": {
                                    "type": "integer",
                                    "description": "Number of lines after",
                                    "default": 3,
                                },
                            },
                            "required": ["line_number"],
                        },
                    ),
                    Tool(
                        name="get_column_range",
                        description="Get a range of columns from a specific line",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "line_number": {
                                    "type": "integer",
                                    "description": "Line number (0-indexed)",
                                },
                                "start_col": {
                                    "type": "integer",
                                    "description": "Start column (0-indexed, inclusive)",
                                },
                                "end_col": {
                                    "type": "integer",
                                    "description": "End column (0-indexed, inclusive)",
                                },
                            },
                            "required": ["line_number", "start_col", "end_col"],
                        },
                    ),
                    Tool(
                        name="get_page_info",
                        description="Get line range information for a specific page",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "page_number": {
                                    "type": "integer",
                                    "description": "Page number",
                                }
                            },
                            "required": ["page_number"],
                        },
                    ),
                ])

                # Shared core page tools
                self._append_registry_tools(
                    tools,
                    include_page_manager_tools=True,
                    include_image_tools=False,
                    skip_names={
                        "submit_dataset",
                        "exit",
                        "list_page_images",
                        "get_image",
                        "get_line",
                        "get_line_range",
                        "get_line_with_context",
                        "get_column_range",
                        "get_page_info",
                        "get_document_stats",
                    },
                )

                # Keep get_document_stats local (not in shared registry yet)
                tools.append(
                    Tool(
                        name="get_document_stats",
                        description="Get document statistics (total lines, pages, etc.)",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                        },
                    )
                )

            # Add image tools if extracted_dir is available
            if self.extracted_dir:
                self._append_registry_tools(
                    tools,
                    include_page_manager_tools=False,
                    include_image_tools=True,
                    skip_names={"submit_dataset", "exit"},
                )

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            if name == "get_current_paragraph":
                content = self.navigator.get_current()
                paragraph_num = self.navigator.current_index
                
                # If using page_manager, get additional info
                line_num = None
                page_num = None
                if self.page_manager:
                    # Find line number for this paragraph
                    para_info = self.page_manager.get_paragraph_info(paragraph_num)
                    if para_info:
                        line_num = para_info["start_line"]
                        page_num = para_info["page_number"]
                        self._record_page_access([page_num])
                
                response = self._format_response(
                    content=content,
                    line_number=line_num,
                    paragraph_number=paragraph_num,
                    page_number=page_num
                )
                return [TextContent(type="text", text=str(response))]

            elif name == "move_forward":
                steps = arguments.get("steps", 1)
                content = self.navigator.move_forward(steps)
                paragraph_num = self.navigator.current_index
                
                line_num = None
                page_num = None
                if self.page_manager:
                    para_info = self.page_manager.get_paragraph_info(paragraph_num)
                    if para_info:
                        line_num = para_info["start_line"]
                        page_num = para_info["page_number"]
                        self._record_page_access([page_num])
                
                response = self._format_response(
                    content=content,
                    line_number=line_num,
                    paragraph_number=paragraph_num,
                    page_number=page_num
                )
                return [TextContent(type="text", text=str(response))]

            elif name == "move_backward":
                steps = arguments.get("steps", 1)
                content = self.navigator.move_backward(steps)
                paragraph_num = self.navigator.current_index
                
                line_num = None
                page_num = None
                if self.page_manager:
                    para_info = self.page_manager.get_paragraph_info(paragraph_num)
                    if para_info:
                        line_num = para_info["start_line"]
                        page_num = para_info["page_number"]
                        self._record_page_access([page_num])
                
                response = self._format_response(
                    content=content,
                    line_number=line_num,
                    paragraph_number=paragraph_num,
                    page_number=page_num
                )
                return [TextContent(type="text", text=str(response))]

            elif name == "get_context":
                before = arguments.get("before", 1)
                after = arguments.get("after", 1)
                context = self.navigator.get_context(before, after)
                paragraph_num = context["current_index"]
                content = context["current"]
                
                line_num = None
                page_num = None
                if self.page_manager:
                    para_info = self.page_manager.get_paragraph_info(paragraph_num)
                    if para_info:
                        line_num = para_info["start_line"]
                        page_num = para_info["page_number"]
                        self._record_page_access([page_num])
                
                response = self._format_response(
                    content=content,
                    line_number=line_num,
                    paragraph_number=paragraph_num,
                    page_number=page_num,
                    additional_info={
                        "previous_paragraphs": context["previous"],
                        "next_paragraphs": context["next"],
                        "total_paragraphs": context["total_paragraphs"]
                    }
                )
                return [TextContent(type="text", text=str(response))]

            elif name == "jump_to":
                index = arguments["index"]
                content = self.navigator.jump_to(index)
                paragraph_num = self.navigator.current_index
                
                line_num = None
                page_num = None
                if self.page_manager:
                    para_info = self.page_manager.get_paragraph_info(paragraph_num)
                    if para_info:
                        line_num = para_info["start_line"]
                        page_num = para_info["page_number"]
                        self._record_page_access([page_num])
                
                response = self._format_response(
                    content=content,
                    line_number=line_num,
                    paragraph_number=paragraph_num,
                    page_number=page_num
                )
                return [TextContent(type="text", text=str(response))]
            
            elif name == "submit_dataset":
                messages = arguments.get("messages", [])
                validation_error = validate_dataset_messages(messages)
                if validation_error:
                    response = {
                        "status": "error",
                        "message": validation_error,
                    }
                # Save to database if dataset_manager is available
                elif self.dataset_manager:
                    try:
                        entry_id = self.dataset_manager.add_entry(messages)
                        total_entries = self.dataset_manager.count_entries()
                        
                        response = {
                            "status": "success",
                            "message": f"Multi-turn conversation saved to database. Total entries: {total_entries}",
                            "entry_id": entry_id,
                            "turns": len(messages) // 2
                        }
                    except ValueError as e:
                        response = {
                            "status": "error",
                            "message": str(e)
                        }
                else:
                    response = {
                        "status": "error",
                        "message": "Dataset manager not initialized. Please provide --db-path when starting MCP server."
                    }
                
                return [TextContent(type="text", text=str(response))]
            
            elif name == "exit":
                reason = arguments.get("reason", "No reason provided")
                
                # Get current entry count
                total = 0
                if self.dataset_manager:
                    total = self.dataset_manager.count_entries()
                
                response = {
                    "status": "exiting",
                    "reason": reason,
                    "total_entries": total,
                    "message": f"Session ending: {reason}. Total entries submitted: {total}"
                }
                
                # Set exit flag
                self.should_exit = True
                
                return [TextContent(type="text", text=str(response))]

            # Page-based navigation tools
            elif name in PAGE_TOOL_NAMES and self.page_manager:
                shared = execute_page_tool(
                    self.page_manager,
                    name,
                    arguments,
                    default_search_max_results=100,
                )
                if not shared.get("ok"):
                    return [TextContent(type="text", text=shared.get("error", "Unknown tool error"))]

                self._refresh_page_submission_state()
                self._record_page_access(shared.get("pages_touched", []))

                if name in {"get_current_page", "jump_to_page", "next_page", "previous_page"}:
                    page_info = shared["page_info"]
                    page_num = page_info.get("page_number")
                    if page_num is not None:
                        page_info["submission_count"] = self.page_submission_counts.get(page_num, 0)
                    return [TextContent(type="text", text=str(page_info))]

                if name == "get_page_range":
                    pages_data = shared["pages_data"]
                    total = shared["total_pages"]
                    parts = [
                        f"--- Page {pn} (of {total}) ---\n{pages_data[pn]}"
                        for pn in sorted(pages_data.keys())
                    ]
                    return [TextContent(type="text", text="\n\n".join(parts))]

                if name == "get_page_context":
                    context = shared["context"]
                    pages_list = shared.get("pages_touched", [])
                    context["submission_counts"] = {
                        page: self.page_submission_counts.get(page, 0)
                        for page in pages_list
                    }
                    return [TextContent(type="text", text=str(context))]

                if name == "search_text":
                    results = shared["results"]
                    for result in results:
                        para_num = self.page_manager.get_paragraph_number(result["line_number"])
                        result["paragraph_number"] = para_num
                    return [TextContent(type="text", text=str(results))]

                return [TextContent(type="text", text=f"Unsupported tool response for {name}")]

            # Line/column navigation tools
            elif name == "get_document_stats" and self.page_manager:
                stats = self.page_manager.get_statistics()
                return [TextContent(type="text", text=str(stats))]

            elif name == "get_page_access_summary" and self.page_manager:
                self._refresh_page_submission_state()
                limit = arguments.get("limit", 5)
                page_numbers = list(getattr(self.page_manager, "page_numbers", []))
                summary = build_page_access_summary(
                    counts=self.page_submission_counts,
                    page_numbers=page_numbers,
                    last_submission_page=self.last_accessed_page,
                    limit=limit,
                )
                return [TextContent(type="text", text=json.dumps(summary, ensure_ascii=False, indent=2))]

            elif name == "get_line" and self.page_manager:
                line_num = arguments["line_number"]
                content = self.page_manager.get_line(line_num)
                if content is None:
                    return [TextContent(type="text", text=f"Line {line_num} not found")]
                
                # Get page and paragraph info
                page_num, _ = self.page_manager.line_to_page.get(line_num, (None, None))
                para_num = self.page_manager.get_paragraph_number(line_num)
                self._refresh_page_submission_state()
                self._record_page_access([page_num])
                
                response = self._format_response(
                    content=content,
                    line_number=line_num,
                    paragraph_number=para_num,
                    page_number=page_num
                )
                return [TextContent(type="text", text=str(response))]

            elif name == "get_line_range" and self.page_manager:
                start = arguments["start_line"]
                end = arguments["end_line"]
                lines = self.page_manager.get_line_range(start, end)
                content = "\n".join(lines)
                
                # Get page and paragraph for start line
                page_num, _ = self.page_manager.line_to_page.get(start, (None, None))
                para_num = self.page_manager.get_paragraph_number(start)
                self._refresh_page_submission_state()
                self._record_page_access([page_num])
                
                response = self._format_response(
                    content=content,
                    line_number=start,
                    paragraph_number=para_num,
                    page_number=page_num,
                    additional_info={
                        "start_line": start,
                        "end_line": end,
                        "line_count": len(lines)
                    }
                )
                return [TextContent(type="text", text=str(response))]

            elif name == "get_line_with_context" and self.page_manager:
                line_num = arguments["line_number"]
                before = arguments.get("before", 3)
                after = arguments.get("after", 3)
                context = self.page_manager.get_line_with_context(line_num, before, after)
                
                if "error" in context:
                    return [TextContent(type="text", text=str(context))]
                
                # Already has line_number, page_number in context
                para_num = self.page_manager.get_paragraph_number(line_num)
                self._refresh_page_submission_state()
                self._record_page_access([context.get("page_number")])
                
                response = self._format_response(
                    content=context["content"],
                    line_number=line_num,
                    paragraph_number=para_num,
                    page_number=context["page_number"],
                    additional_info={
                        "before_lines": context["before_lines"],
                        "after_lines": context["after_lines"],
                        "local_line_number": context["local_line_number"],
                        "column_count": context["column_count"]
                    }
                )
                return [TextContent(type="text", text=str(response))]

            elif name == "get_column_range" and self.page_manager:
                line_num = arguments["line_number"]
                start_col = arguments["start_col"]
                end_col = arguments["end_col"]
                result = self.page_manager.get_column_range(line_num, start_col, end_col)
                
                if result is None:
                    return [TextContent(type="text", text=f"Line {line_num} not found")]
                
                page_num, _ = self.page_manager.line_to_page.get(line_num, (None, None))
                para_num = self.page_manager.get_paragraph_number(line_num)
                self._refresh_page_submission_state()
                self._record_page_access([page_num])
                
                response = self._format_response(
                    content=result,
                    line_number=line_num,
                    paragraph_number=para_num,
                    page_number=page_num,
                    additional_info={
                        "start_column": start_col,
                        "end_column": end_col
                    }
                )
                return [TextContent(type="text", text=str(response))]

            elif name == "get_page_info" and self.page_manager:
                page_num = arguments["page_number"]
                self._refresh_page_submission_state()
                info = self.page_manager.get_page_line_info(page_num)
                if info is None:
                    return [TextContent(type="text", text=f"Page {page_num} not found")]
                self._record_page_access([page_num])
                info["submission_count"] = self.page_submission_counts.get(page_num, 0)
                return [TextContent(type="text", text=str(info))]

            # Image tools
            elif name == "list_page_images" and self.extracted_dir:
                page_num = arguments["page_number"]
                result = list_page_images(self.extracted_dir, page_num)
                if not result.get("ok"):
                    return [TextContent(type="text", text=result.get("error", f"Page {page_num} not found"))]
                response = {
                    "page_number": result["page_number"],
                    "image_count": result["image_count"],
                    "images": result["images"],
                }
                return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False))]

            elif name == "get_image" and self.extracted_dir:
                page_num = arguments["page_number"]
                image_name = arguments["image_name"]
                result = get_image_data_url(self.extracted_dir, page_num, image_name)
                if not result.get("ok"):
                    return [TextContent(type="text", text=result.get("error", "Image load error"))]
                response = {
                    "page_number": result["page_number"],
                    "image_name": result["image_name"],
                    "data_url": result["data_url"],
                }
                return [TextContent(type="text", text=json.dumps(response, ensure_ascii=False))]

            else:
                raise ValueError(f"Unknown tool: {name}")

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)


async def create_mcp_server(
    paragraphs: Optional[List[str]] = None,
    page_manager: Optional["PageManager"] = None,
    db_path: Optional[str] = None,
    extracted_dir: Optional[Path] = None
) -> MCPServer:
    """Create and return an MCP server instance.

    Args:
        paragraphs: List of text paragraphs (optional)
        page_manager: PageManager instance for advanced navigation (optional)
        db_path: Path to SQLite database for dataset storage (optional)
        extracted_dir: Path to extracted directory for image access (optional)

    Returns:
        Configured MCP server
    """
    return MCPServer(paragraphs=paragraphs, page_manager=page_manager, db_path=db_path, extracted_dir=extracted_dir)
