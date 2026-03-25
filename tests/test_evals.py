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
