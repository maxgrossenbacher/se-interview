import json
import pytest
from unittest.mock import patch
from tools.travel_tools import find_flight_options, find_hotel_options, find_nearby_attractions


MOCK_SEARCH_RESULT = "Mock search results for testing"


# ── find_flight_options ───────────────────────────────────────────────────

class TestFindFlightOptions:

    @patch("tools.travel_tools.DuckDuckGoSearchRun")
    def test_output_contains_required_fields(self, mock_search_class):
        mock_search_class.return_value.invoke.return_value = MOCK_SEARCH_RESULT
        result = find_flight_options.invoke({"origin": "Denver", "destination": "Tokyo", "departure_date": "June 5 2026"})
        parsed = json.loads(result)
        assert "origin" in parsed
        assert "destination" in parsed
        assert "results" in parsed
        assert "instructions" in parsed


# ── find_hotel_options ────────────────────────────────────────────────────

class TestFindHotelOptions:

    @patch("tools.travel_tools.DuckDuckGoSearchRun")
    def test_output_contains_required_fields(self, mock_search_class):
        mock_search_class.return_value.invoke.return_value = MOCK_SEARCH_RESULT
        result = find_hotel_options.invoke({"destination": "Tokyo"})
        parsed = json.loads(result)
        assert "destination" in parsed
        assert "results" in parsed
        assert "instructions" in parsed


# ── find_nearby_attractions ───────────────────────────────────────────────

class TestFindNearbyAttractions:

    @patch("tools.travel_tools.DuckDuckGoSearchRun")
    def test_output_contains_required_fields(self, mock_search_class):
        mock_search_class.return_value.invoke.return_value = MOCK_SEARCH_RESULT
        result = find_nearby_attractions.invoke({"destination": "Tokyo"})
        parsed = json.loads(result)
        assert "destination" in parsed
        assert "results" in parsed
        assert "instructions" in parsed
        assert "num_days" in parsed
        assert "interests" in parsed


# ── Agent construction ────────────────────────────────────────────────────

class TestAgentConstruction:

    def test_agent_builds_without_error(self):
        from agent import build_agent
        agent = build_agent()
        assert agent is not None

    def test_all_tools_registered(self):
        from agent import tools_by_name
        assert "duckduckgo_search" in tools_by_name
        assert "find_flight_options" in tools_by_name
        assert "find_hotel_options" in tools_by_name
        assert "find_nearby_attractions" in tools_by_name
