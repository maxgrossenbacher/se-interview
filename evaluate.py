"""
Step 6 — Assessment Pipeline

Fetches root spans from Phoenix, runs two LLM-judge assessments,
logs annotations back to Phoenix, creates a frustrated-interactions dataset,
and exports results to CSV.

Assessments:
  1. User Frustration — uses Phoenix built-in USER_FRUSTRATION_PROMPT_TEMPLATE
     to classify each conversation as "frustrated" or "ok".
  2. Tool Selection Correctness — custom template that judges whether the agent
     picked the right tool(s) for the user's request. Labels: correct / acceptable / incorrect.

Usage:
    Ensure Phoenix is running (docker compose up phoenix) and spans exist
    in the "travel-agent" project, then:

        python evaluate.py

    Results are written to:
        eval_frustration.csv
        eval_tool_selection.csv
    And logged as span annotations in Phoenix.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.evals import OpenAIModel

from evals.frustration import run_frustration_assessment
from evals.tool_selection import run_tool_selection_assessment

load_dotenv()

PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")


def fetch_spans(client: Client, project: str = "travel-agent") -> pd.DataFrame:
    """Fetch root spans from Phoenix and index by span_id."""
    spans_df = client.spans.get_spans_dataframe(
        project_identifier=project,
        root_spans_only=True,
        limit=1000,
    )
    if spans_df.empty:
        raise SystemExit(f"No spans found in project '{project}'. Run some queries first.")
    required_cols = ["context.span_id", "attributes.input.value", "attributes.output.value"]
    missing = [c for c in required_cols if c not in spans_df.columns]
    if missing:
        raise SystemExit(f"Spans missing required columns: {missing}")
    spans_df.index = spans_df["context.span_id"]
    print(f"Fetched {len(spans_df)} root spans from project '{project}'")
    return spans_df


SCORE_MAPS = {
    "user_frustration": {"frustrated": 1.0, "ok": 0.0},
    "tool_selection_correctness": {"correct": 1.0, "acceptable": 0.5, "incorrect": 0.0},
}


def log_annotations(
    client: Client,
    results_df: pd.DataFrame,
    annotation_name: str,
) -> None:
    """Log assessment results back to Phoenix as span annotations."""
    annot_df = results_df[["label", "explanation"]].copy()
    score_map = SCORE_MAPS.get(annotation_name, {})
    if score_map:
        annot_df["score"] = annot_df["label"].map(score_map)
    client.spans.log_span_annotations_dataframe(
        dataframe=annot_df,
        annotation_name=annotation_name,
        annotator_kind="LLM",
        sync=True,
    )
    print(f"Logged {len(annot_df)} '{annotation_name}' annotations to Phoenix")


def create_frustrated_dataset(
    client: Client,
    spans_df: pd.DataFrame,
    frustration_results: pd.DataFrame,
) -> None:
    """Filter frustrated interactions and create a Phoenix dataset."""
    frustrated_mask = frustration_results["label"] == "frustrated"
    frustrated_span_ids = frustrated_mask[frustrated_mask].index
    frustrated_spans = spans_df.loc[spans_df.index.intersection(frustrated_span_ids)]

    if frustrated_spans.empty:
        print("No frustrated interactions found — skipping dataset creation")
        return

    dataset_df = frustrated_spans[
        ["context.span_id", "attributes.input.value", "attributes.output.value"]
    ].copy()
    dataset_df = dataset_df.reset_index(drop=True)

    try:
        client.datasets.create_dataset(
            name="frustrated-interactions",
            dataframe=dataset_df,
            input_keys=["attributes.input.value"],
            output_keys=["attributes.output.value"],
            span_id_key="context.span_id",
            dataset_description="Conversations where the user was classified as frustrated.",
        )
        print(f"Created 'frustrated-interactions' dataset with {len(dataset_df)} examples")
    except Exception as e:
        print(f"Dataset 'frustrated-interactions' may already exist, skipping: {e}")


def main() -> None:
    client = Client(base_url=PHOENIX_ENDPOINT)
    model = OpenAIModel(model="gpt-4o", temperature=0)

    # 1. Fetch spans
    spans_df = fetch_spans(client)

    # 2. Run assessments
    print("\n--- User Frustration Assessment ---")
    frustration_results = run_frustration_assessment(spans_df, model)
    print(frustration_results[["label", "explanation"]].to_string())

    print("\n--- Tool Selection Correctness Assessment ---")
    tool_results = run_tool_selection_assessment(spans_df, model)
    print(tool_results[["label", "explanation"]].to_string())

    # 3. Log annotations to Phoenix
    log_annotations(client, frustration_results, "user_frustration")
    log_annotations(client, tool_results, "tool_selection_correctness")

    # 4. Create frustrated-interactions dataset
    create_frustrated_dataset(client, spans_df, frustration_results)

    # 5. Export to CSV
    frustration_results.to_csv("eval_frustration.csv")
    tool_results.to_csv("eval_tool_selection.csv")
    print("\nExported: eval_frustration.csv, eval_tool_selection.csv")


if __name__ == "__main__":
    main()
