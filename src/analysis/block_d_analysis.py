from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "final" / "block_d"
OUTPUT = INPUT


def pct(x):
    return round(float(x) * 100, 2)


def safe_mean(series):
    return float(series.mean()) if len(series) else np.nan


def safe_median(series):
    return float(series.median()) if len(series) else np.nan


def main():
    print("=" * 80)
    print("BLOCK D — RESEARCH METRICS & ANALYSIS")
    print("=" * 80)

    quality = pd.read_csv(INPUT / "quality_vs_k.csv")
    optimal = pd.read_csv(INPUT / "optimal_k_candidates.csv")

    controlled_files = sorted(INPUT.glob("EQ*_results.csv"))

    if not controlled_files:
        raise FileNotFoundError("No controlled experiment CSV files found.")

    controlled = pd.concat(
        [pd.read_csv(path) for path in controlled_files],
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # 1. QUALITY VS K
    # ------------------------------------------------------------------

    quality_by_k = (
        quality.groupby("K")
        .agg(
            Questions=("ID", "nunique"),
            Evidence_Coverage=("Evidence_Coverage", "mean"),
            Answer_Similarity=("Answer_Similarity", "mean"),
            Quality_Score=("Quality_Score", "mean"),
            Prompt_Tokens=("Prompt_Tokens", "mean"),
            Generated_Tokens=("Generated_Tokens", "mean"),
            Total_Tokens=("Total_Tokens", "mean"),
            Retrieval_Time_Seconds=("Retrieval_Time_Seconds", "mean"),
            Generation_Time_Seconds=("Generation_Time_Seconds", "mean"),
            Total_Latency_Seconds=("Total_Latency_Seconds", "mean"),
            Num_Retrieved_Chunks=("Num_Retrieved_Chunks", "mean"),
        )
        .reset_index()
        .sort_values("K")
    )

    quality_by_k["Quality_Improvement"] = (
        quality_by_k["Quality_Score"].diff()
    )

    quality_by_k["Evidence_Improvement"] = (
        quality_by_k["Evidence_Coverage"].diff()
    )

    quality_by_k["Answer_Similarity_Improvement"] = (
        quality_by_k["Answer_Similarity"].diff()
    )

    quality_by_k["Quality_per_1000_Tokens"] = (
        quality_by_k["Quality_Score"]
        / quality_by_k["Total_Tokens"]
        * 1000
    )

    quality_by_k.to_csv(
        OUTPUT / "D1_quality_vs_k.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. QUALITY-COST TRADEOFF
    # ------------------------------------------------------------------

    tradeoff = quality_by_k[
        [
            "K",
            "Quality_Score",
            "Evidence_Coverage",
            "Answer_Similarity",
            "Total_Tokens",
            "Total_Latency_Seconds",
            "Quality_per_1000_Tokens",
            "Quality_Improvement",
        ]
    ].copy()

    tradeoff["Additional_Tokens_vs_Previous_K"] = (
        tradeoff["Total_Tokens"].diff()
    )

    tradeoff["Additional_Latency_vs_Previous_K"] = (
        tradeoff["Total_Latency_Seconds"].diff()
    )

    tradeoff["Quality_Gain_per_1000_Additional_Tokens"] = (
        tradeoff["Quality_Improvement"]
        / tradeoff["Additional_Tokens_vs_Previous_K"]
        * 1000
    )

    tradeoff.to_csv(
        OUTPUT / "D2_quality_cost_tradeoff.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3. CANDIDATE OPTIMAL-K ANALYSIS
    # ------------------------------------------------------------------

    optimal_summary = pd.DataFrame(
        {
            "Metric": [
                "Questions",
                "Mean_Candidate_Optimal_K",
                "Median_Candidate_Optimal_K",
                "Minimum_Candidate_Optimal_K",
                "Maximum_Candidate_Optimal_K",
                "Mean_Best_K_Quality",
                "Mean_Quality_Threshold",
                "Mean_Candidate_Quality",
                "Mean_Maximum_K_Quality",
                "Mean_Estimated_Token_Savings",
                "Mean_Estimated_Latency_Savings",
            ],
            "Value": [
                optimal["ID"].nunique(),
                optimal["Candidate_Optimal_K"].mean(),
                optimal["Candidate_Optimal_K"].median(),
                optimal["Candidate_Optimal_K"].min(),
                optimal["Candidate_Optimal_K"].max(),
                optimal["Best_K_Quality"].mean(),
                optimal["Quality_Threshold"].mean(),
                optimal["Candidate_Quality"].mean(),
                optimal["Maximum_K_Quality"].mean(),
                optimal["Estimated_Token_Savings"].mean(),
                optimal["Estimated_Latency_Savings"].mean(),
            ],
        }
    )

    optimal_summary.to_csv(
        OUTPUT / "D3_optimal_k_summary.csv",
        index=False,
    )

    optimal_distribution = (
        optimal["Candidate_Optimal_K"]
        .value_counts()
        .sort_index()
        .rename_axis("Candidate_Optimal_K")
        .reset_index(name="Questions")
    )

    optimal_distribution.to_csv(
        OUTPUT / "D4_optimal_k_distribution.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. CONTROLLED EXPERIMENT SUMMARY
    # ------------------------------------------------------------------

    controlled["Mode"] = controlled["Mode"].astype(str)

    mode_summary = (
        controlled.groupby("Mode")
        .agg(
            Questions=("ID", "nunique"),
            Mean_Initial_K=("Initial_K", "mean"),
            Mean_Final_K=("Final_K", "mean"),
            Mean_Iterations=("Iterations", "mean"),
            Escalation_Rate=("K_Escalated", "mean"),
            Verification_Rate=("Verification_Performed", "mean"),
            Verification_Pass_Rate=("Verification_Result", "mean"),
            Mean_Chunks_Retrieved=("Chunks_Retrieved", "mean"),
            Mean_Chunks_Used=("Chunks_Used", "mean"),
            Mean_Prompt_Tokens=("Prompt_Tokens", "mean"),
            Mean_Generated_Tokens=("Generated_Tokens", "mean"),
            Mean_Total_Tokens=("Total_Tokens", "mean"),
            Median_Total_Tokens=("Total_Tokens", "median"),
            Mean_Total_Latency_ms=("Total_Latency_ms", "mean"),
            Median_Total_Latency_ms=("Total_Latency_ms", "median"),
        )
        .reset_index()
    )

    mode_summary["Escalation_Rate_pct"] = (
        mode_summary["Escalation_Rate"] * 100
    )

    mode_summary["Verification_Rate_pct"] = (
        mode_summary["Verification_Rate"] * 100
    )

    mode_summary["Verification_Pass_Rate_pct"] = (
        mode_summary["Verification_Pass_Rate"] * 100
    )

    mode_summary.to_csv(
        OUTPUT / "D5_controlled_mode_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 5. ADAPTIVE EFFICIENCY VS FIXED BASELINES
    # ------------------------------------------------------------------

    pivot = controlled.pivot_table(
        index="ID",
        columns="Mode",
        values=[
            "Total_Tokens",
            "Total_Latency_ms",
            "Chunks_Retrieved",
            "Chunks_Used",
        ],
        aggfunc="first",
    )

    rows = []

    for question_id in pivot.index:
        row = {"ID": question_id}

        for metric in [
            "Total_Tokens",
            "Total_Latency_ms",
            "Chunks_Retrieved",
            "Chunks_Used",
        ]:
            adaptive_value = pivot.loc[
                question_id, (metric, "adaptive")
            ]

            row[f"Adaptive_{metric}"] = adaptive_value

            for k in [3, 5, 10]:
                fixed_value = pivot.loc[
                    question_id, (metric, f"fixed_{k}")
                ]

                row[f"Fixed_{k}_{metric}"] = fixed_value

                row[f"Adaptive_vs_K{k}_{metric}_Savings_pct"] = (
                    (fixed_value - adaptive_value)
                    / fixed_value
                    * 100
                    if fixed_value not in [0, np.nan]
                    else np.nan
                )

        adaptive_initial = pivot.loc[
            question_id, ("Total_Tokens", "adaptive")
        ]

        rows.append(row)

    efficiency = pd.DataFrame(rows)

    efficiency.to_csv(
        OUTPUT / "D6_adaptive_efficiency_by_question.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 6. ADAPTIVE SUMMARY
    # ------------------------------------------------------------------

    adaptive_efficiency_summary = []

    for k in [3, 5, 10]:
        token_col = f"Adaptive_vs_K{k}_Total_Tokens_Savings_pct"
        latency_col = f"Adaptive_vs_K{k}_Total_Latency_ms_Savings_pct"
        retrieved_col = f"Adaptive_vs_K{k}_Chunks_Retrieved_Savings_pct"

        adaptive_efficiency_summary.append(
            {
                "Comparison": f"Adaptive_vs_Fixed_K{k}",
                "Mean_Token_Savings_pct": efficiency[token_col].mean(),
                "Median_Token_Savings_pct": efficiency[token_col].median(),
                "Questions_With_Token_Savings": (
                    efficiency[token_col] > 0
                ).sum(),
                "Mean_Latency_Savings_pct": efficiency[latency_col].mean(),
                "Median_Latency_Savings_pct": efficiency[latency_col].median(),
                "Questions_With_Latency_Savings": (
                    efficiency[latency_col] > 0
                ).sum(),
                "Mean_Chunk_Savings_pct": efficiency[retrieved_col].mean(),
                "Median_Chunk_Savings_pct": efficiency[retrieved_col].median(),
            }
        )

    adaptive_efficiency_summary = pd.DataFrame(
        adaptive_efficiency_summary
    )

    adaptive_efficiency_summary.to_csv(
        OUTPUT / "D7_adaptive_efficiency_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 7. ADAPTIVE K DISTRIBUTION
    # ------------------------------------------------------------------

    adaptive_only = controlled[
        controlled["Mode"] == "adaptive"
    ].copy()

    adaptive_k_distribution = (
        adaptive_only["Initial_K"]
        .value_counts()
        .sort_index()
        .rename_axis("Adaptive_Initial_K")
        .reset_index(name="Questions")
    )

    adaptive_k_distribution["Percentage"] = (
        adaptive_k_distribution["Questions"]
        / len(adaptive_only)
        * 100
    )

    adaptive_k_distribution.to_csv(
        OUTPUT / "D8_adaptive_k_distribution.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 8. ADAPTIVE BY COMPLEXITY
    # ------------------------------------------------------------------

    # Controlled experiment files do not necessarily carry
    # question metadata such as Complexity. Therefore, only
    # perform complexity analysis when that column exists.

    if "Complexity" in adaptive_only.columns:
        adaptive_complexity = (
            adaptive_only.groupby("Complexity", dropna=False)
            .agg(
                Questions=("ID", "nunique"),
                Mean_Initial_K=("Initial_K", "mean"),
                Mean_Final_K=("Final_K", "mean"),
                Mean_Total_Tokens=("Total_Tokens", "mean"),
                Median_Total_Tokens=("Total_Tokens", "median"),
                Mean_Total_Latency_ms=("Total_Latency_ms", "mean"),
                Escalation_Rate=("K_Escalated", "mean"),
            )
            .reset_index()
        )
    else:
        adaptive_complexity = pd.DataFrame(
            columns=[
                "Complexity",
                "Questions",
                "Mean_Initial_K",
                "Mean_Final_K",
                "Mean_Total_Tokens",
                "Median_Total_Tokens",
                "Mean_Total_Latency_ms",
                "Escalation_Rate",
            ]
        )

    adaptive_complexity.to_csv(
        OUTPUT / "D9_adaptive_by_complexity.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 9. ADAPTIVE BY CATEGORY
    # ------------------------------------------------------------------

    # Controlled experiment files may also omit category metadata.
    if {
        "Primary_Category",
        "Secondary_Category",
    }.issubset(adaptive_only.columns):

        adaptive_category = (
            adaptive_only.groupby(
                ["Primary_Category", "Secondary_Category"],
                dropna=False,
            )
            .agg(
                Questions=("ID", "nunique"),
                Mean_Initial_K=("Initial_K", "mean"),
                Mean_Final_K=("Final_K", "mean"),
                Mean_Total_Tokens=("Total_Tokens", "mean"),
                Mean_Total_Latency_ms=("Total_Latency_ms", "mean"),
            )
            .reset_index()
        )

    else:
        adaptive_category = pd.DataFrame(
            columns=[
                "Primary_Category",
                "Secondary_Category",
                "Questions",
                "Mean_Initial_K",
                "Mean_Final_K",
                "Mean_Total_Tokens",
                "Mean_Total_Latency_ms",
            ]
        )
    adaptive_category.to_csv(
        OUTPUT / "D10_adaptive_by_category.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 10. ESCALATION / VERIFICATION
    # ------------------------------------------------------------------

    verification = pd.DataFrame(
        [
            {
                "Metric": "Adaptive questions",
                "Value": len(adaptive_only),
            },
            {
                "Metric": "Verification performed",
                "Value": int(
                    adaptive_only["Verification_Performed"].sum()
                ),
            },
            {
                "Metric": "Verification pass",
                "Value": int(
                    adaptive_only["Verification_Result"]
                    .fillna(False)
                    .sum()
                ),
            },
            {
                "Metric": "K escalated",
                "Value": int(
                    adaptive_only["K_Escalated"].sum()
                ),
            },
            {
                "Metric": "Verification rate",
                "Value": float(
                    adaptive_only["Verification_Performed"].mean()
                ),
            },
            {
                "Metric": "Escalation rate",
                "Value": float(
                    adaptive_only["K_Escalated"].mean()
                ),
            },
        ]
    )

    verification.to_csv(
        OUTPUT / "D11_verification_escalation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 11. SUMMARY JSON
    # ------------------------------------------------------------------

    summary = {
        "block": "D",
        "controlled_questions": int(controlled["ID"].nunique()),
        "controlled_records": int(len(controlled)),
        "quality_questions": int(quality["ID"].nunique()),
        "quality_records": int(len(quality)),
        "candidate_optimal_k_mean": float(
            optimal["Candidate_Optimal_K"].mean()
        ),
        "candidate_optimal_k_median": float(
            optimal["Candidate_Optimal_K"].median()
        ),
        "adaptive_initial_k_mean": float(
            adaptive_only["Initial_K"].mean()
        ),
        "adaptive_initial_k_median": float(
            adaptive_only["Initial_K"].median()
        ),
        "adaptive_k_min": int(
            adaptive_only["Initial_K"].min()
        ),
        "adaptive_k_max": int(
            adaptive_only["Initial_K"].max()
        ),
        "adaptive_escalation_rate": float(
            adaptive_only["K_Escalated"].mean()
        ),
        "adaptive_verification_rate": float(
            adaptive_only["Verification_Performed"].mean()
        ),
        "quality_at_k_3": float(
            quality_by_k.loc[
                quality_by_k["K"] == 3,
                "Quality_Score",
            ].iloc[0]
        ),
        "quality_at_k_5": float(
            quality_by_k.loc[
                quality_by_k["K"] == 5,
                "Quality_Score",
            ].iloc[0]
        ),
        "quality_at_k_10": float(
            quality_by_k.loc[
                quality_by_k["K"] == 10,
                "Quality_Score",
            ].iloc[0]
        ),
    }

    (OUTPUT / "D12_block_d_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # CONSOLE SUMMARY
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("BLOCK D RESULTS")
    print("=" * 80)

    print(f"Quality records: {len(quality)}")
    print(f"Quality questions: {quality['ID'].nunique()}")

    print()
    print("QUALITY VS K")
    print(
        quality_by_k[
            [
                "K",
                "Evidence_Coverage",
                "Answer_Similarity",
                "Quality_Score",
                "Total_Tokens",
                "Total_Latency_Seconds",
            ]
        ].round(4).to_string(index=False)
    )

    print()
    print("CANDIDATE OPTIMAL K")
    print(
        optimal_distribution.to_string(index=False)
    )

    print()
    print("CONTROLLED EXPERIMENT")
    print(
        mode_summary[
            [
                "Mode",
                "Questions",
                "Mean_Initial_K",
                "Mean_Final_K",
                "Mean_Total_Tokens",
                "Median_Total_Tokens",
                "Mean_Total_Latency_ms",
                "Median_Total_Latency_ms",
            ]
        ].round(2).to_string(index=False)
    )

    print()
    print("ADAPTIVE K DISTRIBUTION")
    print(
        adaptive_k_distribution.round(2).to_string(index=False)
    )

    print()
    print("ADAPTIVE EFFICIENCY")
    print(
        adaptive_efficiency_summary.round(2).to_string(index=False)
    )

    print()
    print("=" * 80)
    print("BLOCK D ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()