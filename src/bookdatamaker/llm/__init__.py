"""LLM integration module.

Keep package-level imports lightweight so utility submodules can be imported
without pulling optional runtime dependencies (e.g. ``openai``/``vllm``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ParallelDatasetGenerator", "parse_distribution", "Search1APIMCPProxy"]


def __getattr__(name: str) -> Any:
    """Lazily expose heavy classes/functions from submodules."""
    if name in {"ParallelDatasetGenerator", "parse_distribution"}:
        mod = import_module("bookdatamaker.llm.parallel_generator")
        return getattr(mod, name)
    if name == "Search1APIMCPProxy":
        mod = import_module("bookdatamaker.llm.search1api_mcp")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

