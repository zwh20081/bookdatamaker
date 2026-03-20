"""Search1API MCP client proxy for web search, news, crawl and other tools."""

import asyncio
import threading
from typing import Any, Dict, List, Optional

from tqdm import tqdm


class Search1APIMCPProxy:
    """Manages connections to Search1API remote MCP server.

    Uses Streamable HTTP transport to connect to https://mcp.search1api.com/mcp.
    Each generation thread gets its own MCP session.
    Servers are lazily started per-thread and cleaned up on close.
    """

    MCP_URL = "https://mcp.search1api.com/mcp"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._sessions: Dict[int, Any] = {}  # thread_id -> {session, cleanup, loop}
        self._tool_defs: Optional[List[dict]] = None
        self._lock = threading.Lock()

    async def _start_session(self):
        """Start a remote MCP session and return (session, cleanup_func)."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        # streamablehttp_client is an async context manager yielding (read, write, _)
        http_cm = streamablehttp_client(
            self.MCP_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        read_stream, write_stream, _ = await http_cm.__aenter__()

        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        await session.initialize()

        async def cleanup():
            await session.__aexit__(None, None, None)
            await http_cm.__aexit__(None, None, None)

        return session, cleanup

    def _get_or_create_loop(self, thread_id: int):
        """Get or create an event loop + session for a thread."""
        if thread_id not in self._sessions:
            loop = asyncio.new_event_loop()
            session, cleanup = loop.run_until_complete(self._start_session())
            self._sessions[thread_id] = {
                "session": session,
                "cleanup": cleanup,
                "loop": loop,
            }
        return self._sessions[thread_id]

    def get_openai_tool_defs(self, thread_id: int = 0) -> List[dict]:
        """Get Search1API MCP tools as OpenAI function-calling definitions.

        Connects to the remote MCP server to enumerate available tools,
        converts them to OpenAI format with ``search1api_`` prefix.
        Results are cached after the first call.
        """
        if self._tool_defs is not None:
            return self._tool_defs

        with self._lock:
            if self._tool_defs is not None:
                return self._tool_defs

            ctx = self._get_or_create_loop(thread_id)
            result = ctx["loop"].run_until_complete(
                ctx["session"].list_tools()
            )

            defs: List[dict] = []
            for tool in result.tools:
                defs.append({
                    "type": "function",
                    "function": {
                        "name": f"search1api_{tool.name}",
                        "description": tool.description or tool.name,
                        "parameters": tool.inputSchema,
                    },
                })
            self._tool_defs = defs
            tqdm.write(
                f"[Search1API MCP] Loaded {len(defs)} tools: "
                f"{[d['function']['name'] for d in defs]}"
            )
            return defs

    def call_tool(self, thread_id: int, function_name: str, arguments: dict) -> str:
        """Call a Search1API MCP tool from a sync thread context.

        Args:
            thread_id: Thread identifier (each gets its own MCP session)
            function_name: Tool name (with ``search1api_`` prefix)
            arguments: Tool arguments

        Returns:
            Result as string (text content concatenated)
        """
        real_name = function_name
        if real_name.startswith("search1api_"):
            real_name = real_name[len("search1api_"):]

        ctx = self._get_or_create_loop(thread_id)
        result = ctx["loop"].run_until_complete(
            ctx["session"].call_tool(real_name, arguments)
        )

        parts: List[str] = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            elif hasattr(content, "data"):
                mime = getattr(content, "mimeType", "application/octet-stream")
                parts.append(f"data:{mime};base64,{content.data}")

        if result.isError:
            return f"Error: {' '.join(parts)}"
        return "\n".join(parts) if parts else "No content returned"

    def is_search1api_tool(self, function_name: str) -> bool:
        """Check if a function name belongs to a Search1API MCP tool."""
        return function_name.startswith("search1api_")

    def close(self) -> None:
        """Shut down all MCP sessions."""
        for _thread_id, ctx in self._sessions.items():
            try:
                ctx["loop"].run_until_complete(ctx["cleanup"]())
                ctx["loop"].close()
            except Exception:
                pass
        self._sessions.clear()
        tqdm.write("[Search1API MCP] All sessions closed")
