"""
STEP 15
Visualization of Recommendation System Evaluation

Reads:
    data/processed/hybrid_evaluation_summary.csv

Creates:
1. Precision comparison
2. Recall comparison
3. NDCG comparison
4. Overall metric comparison

All values come directly from the Step 14 experiment.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed"

INPUT_FILE = (
    DATA_PROCESSED
    / "hybrid_evaluation_summary.csv"
)

OUTPUT_DIR = (
    DATA_PROCESSED
    / "evaluation_plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD RESULTS
# ============================================================

print("=" * 70)
print("STEP 15 - EVALUATION VISUALIZATION")
print("=" * 70)

print("\nLoading evaluation results...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows loaded: {len(df)}"
)

print("\nEvaluation data:")
print(
    df.to_string(index=False)
)


# ============================================================
# 3. MODEL ORDER
# ============================================================

model_order = [
    "Popularity",
    "Content-Based",
    "KNN Collaborative",
    "Hybrid"
]

df["model"] = pd.Categorical(
    df["model"],
    categories=model_order,
    ordered=True
)

df = df.sort_values(
    ["k", "model"]
)


# ============================================================
# 4. PRECISION PLOT
# ============================================================

print("\nCreating Precision plot...")

precision_data = (
    df.pivot(
        index="model",
        columns="k",
        values="precision"
    )
    .reindex(model_order)
)

ax = precision_data.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Precision@K Comparison"
)

ax.set_xlabel(
    "Recommendation Model"
)

ax.set_ylabel(
    "Precision"
)

ax.legend(
    title="K"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

precision_file = (
    OUTPUT_DIR
    / "precision_comparison.png"
)

plt.savefig(
    precision_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {precision_file}"
)


# ============================================================
# 5. RECALL PLOT
# ============================================================

print("\nCreating Recall plot...")

recall_data = (
    df.pivot(
        index="model",
        columns="k",
        values="recall"
    )
    .reindex(model_order)
)

ax = recall_data.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Recall@K Comparison"
)

ax.set_xlabel(
    "Recommendation Model"
)

ax.set_ylabel(
    "Recall"
)

ax.legend(
    title="K"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

recall_file = (
    OUTPUT_DIR
    / "recall_comparison.png"
)

plt.savefig(
    recall_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {recall_file}"
)


# ============================================================
# 6. NDCG PLOT
# ============================================================

print("\nCreating NDCG plot...")

ndcg_data = (
    df.pivot(
        index="model",
        columns="k",
        values="ndcg"
    )
    .reindex(model_order)
)

ax = ndcg_data.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "NDCG@K Comparison"
)

ax.set_xlabel(
    "Recommendation Model"
)

ax.set_ylabel(
    "NDCG"
)

ax.legend(
    title="K"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

ndcg_file = (
    OUTPUT_DIR
    / "ndcg_comparison.png"
)

plt.savefig(
    ndcg_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {ndcg_file}"
)


# ============================================================
# 7. OVERALL METRIC PLOT
# ============================================================

print("\nCreating overall comparison plot...")

# Use K=10 because it represents the larger
# recommendation list and gives a broader comparison.

k10 = (
    df[df["k"] == 10]
    .set_index("model")
    .reindex(model_order)
)

overall_data = k10[
    [
        "precision",
        "recall",
        "ndcg"
    ]
]

ax = overall_data.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title(
    "Recommendation Model Comparison at K=10"
)

ax.set_xlabel(
    "Recommendation Model"
)

ax.set_ylabel(
    "Metric Value"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.legend(
    title="Metric"
)

plt.tight_layout()

overall_file = (
    OUTPUT_DIR
    / "overall_comparison_k10.png"
)

plt.savefig(
    overall_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {overall_file}"
)


# ============================================================
# 8. PRINT BEST NUMERICAL VALUES
# ============================================================

print("\n")
print("=" * 70)
print("K=10 METRIC SUMMARY")
print("=" * 70)

for metric in [
    "precision",
    "recall",
    "ndcg"
]:

    best_index = (
        k10[metric]
        .idxmax()
    )

    best_value = (
        k10.loc[
            best_index,
            metric
        ]
    )

    print(
        f"{metric.upper():10s}: "
        f"{best_index} "
        f"({best_value:.6f})"
    )


# ============================================================
# 9. COMPLETION
# ============================================================

print("\n")
print("=" * 70)
print("STEP 15 COMPLETE")
print("=" * 70)

print("\nGraphs saved in:")

print(
    OUTPUT_DIR
)

print("\nGenerated files:")

print(
    "1. precision_comparison.png"
)

print(
    "2. recall_comparison.png"
)

print(
    "3. ndcg_comparison.png"
)

print(
    "4. overall_comparison_k10.png"
)