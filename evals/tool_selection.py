import json
import pandas as pd
from phoenix.evals import OpenAIModel, llm_classify, ClassificationTemplate


TOOL_SELECTION_RAILS = ["correct", "incorrect"]

TOOL_SELECTION_TEMPLATE = ClassificationTemplate(
    rails=TOOL_SELECTION_RAILS,
    template="""You are judging whether an AI travel assistant selected the correct tool(s) for the user's request.

Available tools:
- find_flight_options: Search for flights between cities. Requires origin and destination.
- find_hotel_options: Search for hotels in a destination city.
- find_nearby_attractions: Search for attractions and things to do in a destination city.
- duckduckgo_search: General web search for anything not covered by the specialized tools.

User query: {user_input}

Assistant response: {agent_output}

Judge tool selection correctness:
- "correct": The assistant used exactly the right tool(s) for the request, or correctly chose not to use a tool (e.g., asked clarifying questions first).
- "incorrect": The assistant used clearly wrong tools, failed to use any tool when one was needed, or used tools when it should have asked for clarification first.
""",
    scores=[1.0, 0.0],
)

def extract_message(raw):
    """Extract the last message content from a raw span attribute value."""
    try:
        if isinstance(raw, str):
            parsed = json.loads(raw)
            messages = parsed.get("messages", [])
            if messages:
                return messages[-1].get("content", raw)
        return str(raw)
    except Exception:
        return str(raw)


def build_tool_context(raw_input):
    """Extract user input text from raw span attribute."""
    return extract_message(raw_input)


def run_tool_selection_assessment(spans_df, model):
    """
    Run tool-selection-correctness classification on a spans dataframe.

    Args:
        spans_df: DataFrame with 'attributes.input.value' and 'attributes.output.value' columns.
                  Index must be context.span_id.
        model: An OpenAIModel instance for the LLM judge.

    Returns:
        DataFrame with columns: label, explanation, plus llm_classify metadata.
        Index matches spans_df (context.span_id).
    """
    assessment_df = spans_df[["attributes.input.value", "attributes.output.value"]].copy()
    assessment_df["user_input"] = assessment_df["attributes.input.value"].apply(extract_message)
    assessment_df["agent_output"] = assessment_df["attributes.output.value"].apply(extract_message)

    results = llm_classify(
        data=assessment_df,
        model=model,
        template=TOOL_SELECTION_TEMPLATE,
        rails=['correct', 'incorrect'],
        provide_explanation=True,
        include_exceptions=True,
    )
    return results
