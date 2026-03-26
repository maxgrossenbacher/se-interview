import os
import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.evals import OpenAIModel

from evals.frustration import run_frustration_assessment
from evals.tool_selection import run_tool_selection_assessment

load_dotenv()

PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

SCORE_MAPS = {
    "user_frustration": {"frustrated": 1.0, "ok": 0.0},
    "tool_selection_correctness": {"correct": 1.0, "incorrect": 0.0},
}

def fetch_spans(client, project="travel-agent"):
    """Fetch root spans from Phoenix and index by span_id."""
    spans_df = client.spans.get_spans_dataframe(
        project_identifier=project,
        root_spans_only=True,
        limit=1000,
    )
    return spans_df

def log_annotations(client, results_df, annotation_name):
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


def main():
    client = Client(base_url=PHOENIX_ENDPOINT)
    model = OpenAIModel(model="gpt-4o", temperature=0)

    # 1. Fetch spans
    spans_df = fetch_spans(client)
    print(f"Fetched {len(spans_df)} spans")

    # 2. Run assessments
    frustration_results = run_frustration_assessment(spans_df, model)
    tool_results = run_tool_selection_assessment(spans_df, model)

    # 3. Log annotations to Phoenix
    log_annotations(client, frustration_results, "user_frustration")
    log_annotations(client, tool_results, "tool_selection_correctness")

    # 4. Export to CSV
    frustration_results.to_csv("eval_frustration.csv")
    tool_results.to_csv("eval_tool_selection.csv")
    print("Exported: eval_frustration.csv, eval_tool_selection.csv")


if __name__ == "__main__":
    main()
