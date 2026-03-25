import json
import pytest
import pandas as pd
from evals.frustration import build_conversation_text


class TestBuildConversationText:
    def test_simple_messages(self):
        raw_input = json.dumps({"messages": [{"content": "Find me a hotel in Paris"}]})
        raw_output = "Here are some hotels in Paris..."
        result = build_conversation_text(raw_input, raw_output)
        assert "Find me a hotel in Paris" in result
        assert "Here are some hotels in Paris" in result

    def test_plain_string_fallback(self):
        result = build_conversation_text("hello", "world")
        assert "hello" in result
        assert "world" in result

    def test_malformed_json_fallback(self):
        result = build_conversation_text("{bad json", "output")
        assert "{bad json" in result
        assert "output" in result


from evals.tool_selection import TOOL_SELECTION_TEMPLATE, build_tool_context


class TestBuildToolContext:
    def test_extracts_content_from_json(self):
        raw_input = json.dumps({"messages": [{"content": "Find flights to Tokyo"}]})
        result = build_tool_context(raw_input)
        assert "Find flights to Tokyo" in result

    def test_plain_text_passthrough(self):
        result = build_tool_context("Hello there")
        assert "Hello there" in result


class TestToolSelectionTemplate:
    def test_template_exists(self):
        assert TOOL_SELECTION_TEMPLATE is not None
