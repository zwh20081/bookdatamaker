"""Tests for LLM message utility helpers."""

from bookdatamaker.llm.message_utils import (
    extract_think,
    safe_prune_messages,
    sanitize_tool_pairs,
    serialize_messages,
)


class _DummyFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _PlainToolCall:
    def __init__(self, tool_call_id: str, name: str, arguments: str) -> None:
        self.id = tool_call_id
        self.type = "function"
        self.function = _DummyFunction(name=name, arguments=arguments)


class _DumpableToolCall:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


def test_extract_think_splits_hidden_and_visible_parts() -> None:
    content = "hello <think>inner1</think> world <think>inner2</think> done"

    think_text, visible = extract_think(content)

    assert think_text == "inner1\ninner2"
    assert visible == "hello  world  done"


def test_sanitize_tool_pairs_keeps_complete_blocks() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        {"role": "assistant", "content": "next"},
    ]

    sanitized = sanitize_tool_pairs(messages)

    assert sanitized == messages


def test_sanitize_tool_pairs_drops_incomplete_blocks() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "another", "content": "wrong"},
        {"role": "assistant", "content": "next"},
    ]

    sanitized = sanitize_tool_pairs(messages)

    # Incomplete tool block should be removed entirely.
    assert sanitized == [{"role": "assistant", "content": "next"}]


def test_safe_prune_preserves_system_and_tool_group_boundaries() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool-res"},
        {"role": "assistant", "content": "done"},
    ]

    pruned = safe_prune_messages(messages, keep_last=2)

    assert pruned[0] == {"role": "system", "content": "sys"}
    # Assistant tool_call should still be present because we cut at a tool message boundary.
    assert any(m.get("tool_calls") for m in pruned)
    assert any(m.get("role") == "tool" for m in pruned)


def test_serialize_messages_supports_dict_dumpable_and_plain_tool_calls() -> None:
    dict_tool_call = {
        "id": "d1",
        "type": "function",
        "function": {"name": "jump_to_page", "arguments": "{\"page_number\": 1}"},
    }
    dumpable_tool_call = _DumpableToolCall(
        {
            "id": "d2",
            "type": "function",
            "function": {"name": "next_page", "arguments": "{\"steps\": 1}"},
        }
    )
    plain_tool_call = _PlainToolCall(
        tool_call_id="d3",
        name="search_text",
        arguments="{\"query\": \"topic\"}",
    )

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [dict_tool_call, dumpable_tool_call, plain_tool_call],
        }
    ]

    serialized = serialize_messages(messages)

    assert len(serialized) == 1
    tool_calls = serialized[0]["tool_calls"]
    assert [call["id"] for call in tool_calls] == ["d1", "d2", "d3"]
    assert tool_calls[2]["function"]["name"] == "search_text"
