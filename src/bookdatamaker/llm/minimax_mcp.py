"""MiniMax MCP client proxy for web_search and understand_image tools."""

import asyncio
import os
import shutil
import threading
from typing import Any, Dict, List, Optional

from tqdm import tqdm


class MiniMaxMCPProxy:
    """Manages connections to minimax-coding-plan-mcp servers.
    
    Each generation thread gets its own MCP server process (stdio is 1:1).
    Servers are lazily started per-thread and cleaned up on close.
    """

    def __init__(
        self,
        api_key: str,
        api_host: str = "https://api.minimaxi.com",
        uvx_path: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.api_host = api_host
        self.uvx_path = uvx_path or shutil.which("uvx") or "uvx"
        self._sessions: Dict[int, Any] = {}  # thread_id -> (session, cleanup)
        self._tool_defs: Optional[List[dict]] = None
        self._lock = threading.Lock()

    def _get_server_params(self):
        """Get MCP server parameters."""
        from mcp import StdioServerParameters
        return StdioServerParameters(
            command=self.uvx_path,
            args=["minimax-coding-plan-mcp", "-y"],
            env={
                **os.environ,
                "MINIMAX_API_KEY": self.api_key,
                "MINIMAX_API_HOST": self.api_host,
            },
        )

    async def _start_session(self):
        """Start an MCP server process and return (session, cleanup_func)."""
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        server_params = self._get_server_params()
        # stdio_client is an async context manager that yields (read, write)
        read_write_cm = stdio_client(server_params, errlog=open(os.devnull, "w"))
        read_stream, write_stream = await read_write_cm.__aenter__()
        
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        await session.initialize()
        
        async def cleanup():
            await session.__aexit__(None, None, None)
            await read_write_cm.__aexit__(None, None, None)
        
        return session, cleanup

    def _get_or_create_loop(self, thread_id: int):
        """Get or create an event loop for a thread."""
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
        """Get MiniMax MCP tools as OpenAI function calling definitions.
        
        Connects to the MCP server to list available tools, then converts
        them to OpenAI format. Results are cached after first call.
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

            defs = []
            for tool in result.tools:
                defs.append({
                    "type": "function",
                    "function": {
                        "name": f"minimax_{tool.name}",
                        "description": tool.description or tool.name,
                        "parameters": tool.inputSchema,
                    },
                })
            self._tool_defs = defs
            tqdm.write(f"[MiniMax MCP] Loaded {len(defs)} tools: {[d['function']['name'] for d in defs]}")
            return defs

    def call_tool(self, thread_id: int, function_name: str, arguments: dict) -> str:
        """Call a MiniMax MCP tool from a sync thread context.
        
        Args:
            thread_id: Thread identifier (each gets its own MCP server)
            function_name: Tool name (with 'minimax_' prefix)
            arguments: Tool arguments
            
        Returns:
            Result as string (text content concatenated)
        """
        # Strip the minimax_ prefix to get the real tool name
        real_name = function_name
        if real_name.startswith("minimax_"):
            real_name = real_name[len("minimax_"):]
        
        ctx = self._get_or_create_loop(thread_id)
        result = ctx["loop"].run_until_complete(
            ctx["session"].call_tool(real_name, arguments)
        )
        
        # Extract text from CallToolResult
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            elif hasattr(content, "data"):
                # Image content - return as data URL
                mime = getattr(content, "mimeType", "image/png")
                parts.append(f"data:{mime};base64,{content.data}")
        
        if result.isError:
            return f"Error: {' '.join(parts)}"
        return "\n".join(parts) if parts else "No content returned"

    def is_minimax_tool(self, function_name: str) -> bool:
        """Check if a function name belongs to a MiniMax MCP tool."""
        return function_name.startswith("minimax_")

    def close(self) -> None:
        """Shut down all MCP server processes."""
        for thread_id, ctx in self._sessions.items():
            try:
                ctx["loop"].run_until_complete(ctx["cleanup"]())
                ctx["loop"].close()
            except Exception:
                pass
        self._sessions.clear()
        tqdm.write("[MiniMax MCP] All sessions closed")
