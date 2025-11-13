"""MCP server for paragraph and line/column navigation."""

from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

try:
    from ..utils.page_manager import PageManager
except ImportError:
    PageManager = None


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
        db_path: Optional[str] = None
    ) -> None:
        """Initialize MCP server.

        Args:
            paragraphs: List of text paragraphs (optional, for backward compatibility)
            page_manager: PageManager instance for advanced navigation (optional)
            db_path: Path to SQLite database for dataset storage (optional)
        """
        self.navigator = ParagraphNavigator(paragraphs) if paragraphs else None
        self.page_manager = page_manager
        self.db_path = db_path
        self.dataset_manager = None
        self.should_exit = False
        
        # Initialize dataset manager if db_path provided
        if self.db_path:
            try:
                from ..dataset import DatasetManager
                self.dataset_manager = DatasetManager(self.db_path)
            except ImportError:
                print("Warning: Could not import DatasetManager")
        
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
        
        return response

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
                Tool(
                    name="submit_dataset",
                    description="Submit a Q&A pair to the dataset. Use this to save question-answer pairs generated during document analysis.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "The input/question text",
                            },
                            "output": {
                                "type": "string",
                                "description": "The output/answer text",
                            }
                        },
                        "required": ["input", "output"],
                    },
                ),
                Tool(
                    name="exit",
                    description="Exit the MCP session after completing the required number of dataset submissions. Use this when you have submitted the target number of Q&A pairs.",
                    inputSchema={
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
            ]

            # Add page-based navigation tools if page_manager is available
            if self.page_manager:
                tools.extend([
                    Tool(
                        name="get_current_page",
                        description="Get the current page content with metadata (page number, total pages, etc.)",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                        },
                    ),
                    Tool(
                        name="next_page",
                        description="Move to the next page(s) and return the new page content",
                        inputSchema={
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
                    Tool(
                        name="previous_page",
                        description="Move to the previous page(s) and return the new page content",
                        inputSchema={
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
                    Tool(
                        name="jump_to_page",
                        description="Jump to a specific page by page number",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "page_number": {
                                    "type": "integer",
                                    "description": "Target page number",
                                }
                            },
                            "required": ["page_number"],
                        },
                    ),
                    Tool(
                        name="get_page_context",
                        description="Get current page with surrounding pages for context",
                        inputSchema={
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
                    Tool(
                        name="get_document_stats",
                        description="Get document statistics (total lines, pages, etc.)",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                        },
                    ),
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
                        name="search_text",
                        description="Search for text across the document",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query",
                                },
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
                
                response = self._format_response(
                    content=content,
                    line_number=line_num,
                    paragraph_number=paragraph_num,
                    page_number=page_num
                )
                return [TextContent(type="text", text=str(response))]
            
            elif name == "submit_dataset":
                prompt_text = arguments.get("input", "").strip()
                response_text = arguments.get("output", "").strip()
                
                # Validate that prompt and response are not empty
                if not prompt_text or not response_text:
                    response = {
                        "status": "error",
                        "message": "Both 'input' (question) and 'output' (answer) cannot be empty. Please provide valid content for both fields."
                    }
                # Save to database if dataset_manager is available
                elif self.dataset_manager:
                    entry_id = self.dataset_manager.add_entry(prompt_text, response_text)
                    total_entries = self.dataset_manager.count_entries()
                    
                    response = {
                        "status": "success",
                        "message": f"Dataset entry saved to database. Total entries: {total_entries}",
                        "entry_id": entry_id,
                        "entry": {
                            "prompt": prompt_text,
                            "response": response_text
                        }
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
            elif name == "get_current_page" and self.page_manager:
                page_info = self.page_manager.get_page_info()
                return [TextContent(type="text", text=str(page_info))]

            elif name == "next_page" and self.page_manager:
                steps = arguments.get("steps", 1)
                self.page_manager.next_page(steps)
                page_info = self.page_manager.get_page_info()
                return [TextContent(type="text", text=str(page_info))]

            elif name == "previous_page" and self.page_manager:
                steps = arguments.get("steps", 1)
                self.page_manager.previous_page(steps)
                page_info = self.page_manager.get_page_info()
                return [TextContent(type="text", text=str(page_info))]

            elif name == "jump_to_page" and self.page_manager:
                page_number = arguments["page_number"]
                content = self.page_manager.jump_to_page(page_number)
                if content is None:
                    return [TextContent(type="text", text=f"Page {page_number} not found")]
                page_info = self.page_manager.get_page_info()
                return [TextContent(type="text", text=str(page_info))]

            elif name == "get_page_context" and self.page_manager:
                before = arguments.get("before", 1)
                after = arguments.get("after", 1)
                current_page = self.page_manager.get_current_page_number()
                context = self.page_manager.get_context(current_page, before, after)
                return [TextContent(type="text", text=str(context))]

            # Line/column navigation tools
            elif name == "get_document_stats" and self.page_manager:
                stats = self.page_manager.get_statistics()
                return [TextContent(type="text", text=str(stats))]

            elif name == "get_line" and self.page_manager:
                line_num = arguments["line_number"]
                content = self.page_manager.get_line(line_num)
                if content is None:
                    return [TextContent(type="text", text=f"Line {line_num} not found")]
                
                # Get page and paragraph info
                page_num, _ = self.page_manager.line_to_page.get(line_num, (None, None))
                para_num = self.page_manager.get_paragraph_number(line_num)
                
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

            elif name == "search_text" and self.page_manager:
                query = arguments["query"]
                case_sensitive = arguments.get("case_sensitive", False)
                max_results = arguments.get("max_results", 100)
                results = self.page_manager.search_text(query, case_sensitive, max_results)
                
                # Each result already has line_number, page_number, paragraph info
                # Add paragraph_number to each result
                for result in results:
                    para_num = self.page_manager.get_paragraph_number(result["line_number"])
                    result["paragraph_number"] = para_num
                
                return [TextContent(type="text", text=str(results))]

            elif name == "get_page_info" and self.page_manager:
                page_num = arguments["page_number"]
                info = self.page_manager.get_page_line_info(page_num)
                if info is None:
                    return [TextContent(type="text", text=f"Page {page_num} not found")]
                return [TextContent(type="text", text=str(info))]

            else:
                raise ValueError(f"Unknown tool: {name}")

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)


async def create_mcp_server(
    paragraphs: Optional[List[str]] = None,
    page_manager: Optional["PageManager"] = None,
    db_path: Optional[str] = None
) -> MCPServer:
    """Create and return an MCP server instance.

    Args:
        paragraphs: List of text paragraphs (optional)
        page_manager: PageManager instance for advanced navigation (optional)
        db_path: Path to SQLite database for dataset storage (optional)

    Returns:
        Configured MCP server
    """
    return MCPServer(paragraphs=paragraphs, page_manager=page_manager, db_path=db_path)
