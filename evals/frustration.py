import json
import pandas as pd
from phoenix.evals import (
    OpenAIModel,
    llm_classify,
    USER_FRUSTRATION_PROMPT_TEMPLATE,
)

# USER_FRUSTRATION_PROMPT_TEMPLATE = '''
# You are given a conversation where between a user and an assistant.
#   Here is the conversation:
#   [BEGIN DATA]
#   *****************
#   Conversation:
#   {conversation}
#   *****************
#   [END DATA]

#   Examine the conversation and determine whether or not the user got frustrated from the experience.
#   Frustration can range from midly frustrated to extremely frustrated. If the user seemed frustrated
#   at the beginning of the conversation but seemed satisfied at the end, they should not be deemed
#   as frustrated. Focus on how the user left the conversation.

#   Your response must be a single word, either "frustrated" or "ok", and should not
#   contain any text or characters aside from that word. "frustrated" means the user was left
#   frustrated as a result of the conversation. "ok" means that the user did not get frustrated
#   from the conversation.
# '''

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


def build_conversation_text(raw_input, raw_output):
    """Build a conversation string from raw span data."""
    user_msg = extract_message(raw_input)
    assistant_msg = extract_message(raw_output)
    return f"User: {user_msg}\nAssistant: {assistant_msg}"


def run_frustration_assessment(spans_df, model):
    """
    Run user-frustration classification on a spans dataframe.

    Args:
        spans_df: DataFrame with 'attributes.input.value' and 'attributes.output.value' columns.
                  Index must be context.span_id.
        model: An OpenAIModel instance for the LLM judge.

    Returns:
        DataFrame with columns: label, explanation, plus llm_classify metadata.
        Index matches spans_df (context.span_id).
    """
    assessment_df = spans_df[["attributes.input.value", "attributes.output.value"]].copy()
    assessment_df["conversation"] = assessment_df.apply(
        lambda r: build_conversation_text(
            r["attributes.input.value"], r["attributes.output.value"]
        ),
        axis=1,
    )

    results = llm_classify(
        data=assessment_df,
        model=model,
        template=USER_FRUSTRATION_PROMPT_TEMPLATE,
        rails=["frustrated", "ok"],
        provide_explanation=True,
        include_exceptions=True,
    )
    return results
