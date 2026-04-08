"""Shared test helpers for langchain-router."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage


def make_human(content: str = "do something") -> HumanMessage:
    """Create a ``HumanMessage``."""
    return HumanMessage(content=content)


def make_tool(content: str = "result", call_id: str = "tc0") -> ToolMessage:
    """Create a ``ToolMessage``."""
    return ToolMessage(content=content, tool_call_id=call_id)


def make_tool_error(call_id: str = "tc0") -> ToolMessage:
    """Create a ``ToolMessage`` that looks like an error."""
    return ToolMessage(
        content="Error: FileNotFoundError: /src/missing.py",
        tool_call_id=call_id,
    )
