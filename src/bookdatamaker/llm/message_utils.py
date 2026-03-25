"""Message utility helpers for LLM tool-calling loops."""

from __future__ import annotations

import re
from typing import Any


def serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize messages to JSON-compatible format.

    Converts OpenAI tool_calls objects to dictionaries so they can be
    persisted in thread checkpoints.

    Args:
        messages: List of message dictionaries

    Returns:
        JSON-serializable list of messages
    """
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        msg_copy = msg.copy()

        # Convert tool_calls to dict if present
        if "tool_calls" in msg_copy and msg_copy["tool_calls"]:
            tool_calls_list: list[dict[str, Any]] = []
            for tool_call in msg_copy["tool_calls"]:
                # Check if already a dict (from restored state)
                if isinstance(tool_call, dict):
                    tool_calls_list.append(tool_call)
                # Convert ChatCompletionMessageToolCall to dict
                elif hasattr(tool_call, "model_dump"):
                    tool_calls_list.append(tool_call.model_dump())
                elif hasattr(tool_call, "dict"):
                    tool_calls_list.append(tool_call.dict())
                else:
                    # Manual conversion for objects
                    tool_calls_list.append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
            msg_copy["tool_calls"] = tool_calls_list

        serialized.append(msg_copy)

    return serialized


def extract_think(content: str) -> tuple[str, str]:
    """Extract ``<think>`` blocks and remaining visible text from content.

    Returns:
        A tuple of ``(think_text, visible_text)`` where think_text is the
        concatenated content inside ``<think>`` tags (or empty string).
    """
    think_parts = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
    visible = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    think_text = "\n".join(part.strip() for part in think_parts if part.strip())
    return think_text, visible


def sanitize_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove incomplete tool_call/tool_result groups from message list.

    Ensures every assistant message with tool_calls has all corresponding tool
    result messages immediately following it. Drops incomplete groups to avoid
    API validation errors.
    """
    result: list[dict[str, Any]] = []
    index = 0

    while index < len(messages):
        msg = messages[index]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Collect expected tool_call_ids
            tool_calls = msg["tool_calls"]
            expected_ids = {
                tool_call["id"] if isinstance(tool_call, dict) else tool_call.id
                for tool_call in tool_calls
            }

            # Collect following tool result messages
            end_index = index + 1
            found_ids: set[str] = set()
            while end_index < len(messages) and messages[end_index].get("role") == "tool":
                tool_call_id = messages[end_index].get("tool_call_id")
                if tool_call_id is not None:
                    found_ids.add(tool_call_id)
                end_index += 1

            if expected_ids <= found_ids:
                # Complete block — keep assistant + all tool results
                result.append(msg)
                result.extend(messages[index + 1 : end_index])
            # else: incomplete block — skip assistant + partial tool results

            index = end_index
            continue

        result.append(msg)
        index += 1

    return result


def safe_prune_messages(messages: list[dict[str, Any]], keep_last: int = 10) -> list[dict[str, Any]]:
    """Prune messages while preserving tool-call integrity.

    Keeps the system message (if present) plus the most recent messages, while
    ensuring assistant tool_calls are not orphaned from their tool responses.

    Args:
        messages: Full message list
        keep_last: Target number of recent messages to keep

    Returns:
        Pruned and sanitized message list
    """
    if len(messages) <= keep_last + 1:
        # Still sanitize even short lists (could have incomplete tool blocks)
        return sanitize_tool_pairs(messages)

    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    start_idx = 1 if system_msg else 0

    # Take the last keep_last messages
    candidate = messages[-keep_last:]

    # If the first message is a tool role, we need to include the preceding
    # assistant tool_call group.
    cut_point = len(messages) - keep_last
    while candidate and candidate[0].get("role") == "tool" and cut_point > start_idx:
        cut_point -= 1
        candidate = messages[cut_point:]

    result = ([system_msg] if system_msg else []) + candidate
    result = sanitize_tool_pairs(result)

    # Ensure the first non-system message is a user message.
    # After pruning, the history may start with tool/assistant messages,
    # which many APIs (OpenAI, MiniMax) reject with an empty response.
    non_system = [m for m in result if m.get("role") != "system"]
    if non_system and non_system[0].get("role") != "user":
        placeholder = {"role": "user", "content": "Continue the task."}
        insert_at = 1 if (result and result[0].get("role") == "system") else 0
        result.insert(insert_at, placeholder)

    return result
